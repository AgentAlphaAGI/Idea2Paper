# 中文注释: 本文件实现论文链路的多 Reviewer 评审与聚合逻辑（可接真实 LLM，也可离线运行）。
from __future__ import annotations

import hashlib
import json
import logging
import secrets
from dataclasses import dataclass
from statistics import mean
from typing import Dict, List, Optional, Sequence, Tuple

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from core.llm_utils import coerce_float, coerce_model, safe_content
from paper_kg.llm_schemas import PaperIssueType, PaperReviewDefect, PaperReviewerOutput, PaperReviewScores
from paper_kg.interfaces import IGraphStore
from paper_kg.models import StoryDraft, Trick
from paper_kg.prompts import load_paper_prompt


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PaperReviewerConfig:
    """
    功能：定义单个 Reviewer 的配置。
    参数：reviewer_id/persona/prompt_file/threshold。
    返回：PaperReviewerConfig。
    流程：以 dataclass 固化配置，便于在 controller 中注入/替换。
    说明：
    - prompt_file 指向 prompts_paper 下的 system prompt 文件名。
    - threshold 可用于“每个 reviewer 单独阈值”策略（可选启用）。
    """

    reviewer_id: str
    persona: str
    prompt_file: str
    threshold: float = 7.0


class PaperReviewResult(PaperReviewerOutput):
    """
    功能：对外暴露的 Reviewer 结果（在 LLM 输出基础上补充 reviewer_id/persona 等上下文）。
    参数：继承 PaperReviewerOutput，并额外包含 reviewer_id/persona/threshold。
    返回：PaperReviewResult。
    流程：字段校验 → 序列化。
    说明：该模型会被写入最终 report JSON。
    """

    reviewer_id: str = ""
    persona: str = ""
    threshold: float = 7.0


def _normalize_issue_type(raw: str, scores: PaperReviewScores) -> PaperIssueType:
    """
    功能：把 LLM 产出的 issue_type 归一化为固定枚举集合。
    参数：raw、scores。
    返回：PaperIssueType。
    流程：字符串清洗 → 映射 → 若失败则按分数推断。
    说明：用于降低 LLM 输出波动导致的解析/逻辑分叉风险。
    """
    cleaned = str(raw or "").strip().upper()
    mapping = {
        "PASS": "PASS",
        "OK": "PASS",
        "NONE": "PASS",
        "CLARITY": "CLARITY_ISSUE",
        "CLARITY_ISSUE": "CLARITY_ISSUE",
        "SOUNDNESS": "SOUNDNESS_RISK",
        "SOUNDNESS_RISK": "SOUNDNESS_RISK",
        "NOVELTY": "NOVELTY_LOW",
        "NOVELTY_LOW": "NOVELTY_LOW",
        "SIGNIFICANCE": "SIGNIFICANCE_LOW",
        "SIGNIFICANCE_LOW": "SIGNIFICANCE_LOW",
    }
    if cleaned in mapping:
        return mapping[cleaned]  # type: ignore[return-value]

    # 中文注释：兜底策略——根据四维评分推断主要问题类型。
    if scores.clarity < 5.0:
        return "CLARITY_ISSUE"
    if scores.soundness < 5.0:
        return "SOUNDNESS_RISK"
    if scores.novelty < 5.0:
        return "NOVELTY_LOW"
    if scores.significance < 5.0:
        return "SIGNIFICANCE_LOW"
    return "PASS"


def _clamp_score(value: float) -> float:
    """
    功能：把任意分值裁剪到 [0, 10]。
    参数：value。
    返回：裁剪后的浮点数。
    流程：max/min 限定范围。
    说明：用于对抗 LLM 输出越界。
    """
    return max(0.0, min(10.0, float(value)))


def _calc_overall(scores: PaperReviewScores) -> float:
    """
    功能：根据四维评分计算 overall_score（当 LLM 未提供或不可信时）。
    参数：scores。
    返回：overall_score（0-10）。
    流程：四维均值 → 裁剪。
    说明：保持与后续聚合一致，便于 trimmed mean。
    """
    return _clamp_score(mean([scores.novelty, scores.soundness, scores.clarity, scores.significance]))


def trimmed_mean(values: Sequence[float]) -> float:
    """
    功能：计算 trimmed mean（去掉最高与最低后求均值）。
    参数：values。
    返回：均值。
    流程：排序 → len>=3 时去头去尾 → mean。
    说明：用于降低单个 reviewer 极端值对决策的影响。
    """
    numbers = [float(v) for v in values]
    if not numbers:
        return 0.0
    if len(numbers) < 3:
        return float(mean(numbers))
    numbers.sort()
    trimmed = numbers[1:-1]
    return float(mean(trimmed)) if trimmed else float(mean(numbers))


def _truncate_text(text: str, max_chars: int) -> str:
    """
    功能：截断文本到指定长度。
    参数：text、max_chars。
    返回：截断后的文本。
    流程：长度判断 → 截断。
    说明：用于控制 reviewer 输入长度，避免超时。
    """
    if max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    return text[: max_chars].rstrip() + "…"


def _build_review_pack(draft: StoryDraft) -> Dict:
    """
    功能：构建 reviewer 输入包（避免塞整篇全文）。
    参数：draft。
    返回：review_pack 字典。
    流程：提取关键章节全文/摘要 → 拼接结构化 JSON。
    说明：Abstract/Method/Experiments/Conclusion 可给全文或裁剪，其余使用摘要。
    """
    pack: Dict[str, object] = {
        "idea": draft.idea,
        "domain": draft.domain,
        "patterns": [pattern.pattern_name for pattern in draft.patterns],
        "tricks": [trick.name for trick in draft.tricks],
        "story_skeleton": draft.story_skeleton.model_dump(),
    }

    if draft.paper_plan:
        pack["paper_plan"] = {
            "template_name": draft.paper_plan.template_name,
            "sections": [
                {
                    "section_id": section.section_id,
                    "title": section.title,
                    "goal": section.goal,
                    "required_points": section.required_points,
                    "required_tricks": section.required_tricks,
                }
                for section in draft.paper_plan.sections
            ],
        }

    if not draft.paper_sections:
        pack["final_story"] = _truncate_text(draft.final_story, 1200)
        return pack

    key_sections = {"abstract": 1500, "method": 1500, "experiments": 1500, "conclusion": 1200}
    section_payload = []
    for section in draft.paper_sections:
        content = section.content_markdown
        if section.section_id in key_sections:
            content = _truncate_text(content, key_sections[section.section_id])
        else:
            content = _truncate_text(section.summary, 600)
        section_payload.append(
            {
                "section_id": section.section_id,
                "title": section.title,
                "content": content,
                "summary": section.summary,
                "used_tricks": section.used_tricks,
                "covered_points": section.covered_points,
                "todo": section.todo,
            }
        )

    pack["sections"] = section_payload
    pack["paper_markdown_excerpt"] = _truncate_text(draft.paper_markdown, 2000)
    return pack


def aggregate_defects(reviews: Sequence[PaperReviewResult]) -> List[PaperReviewDefect]:
    """
    功能：聚合多个 reviewer 的 defects，并按“出现频次”排序。
    参数：reviews。
    返回：排序后的缺陷列表。
    流程：收集 defects → 统计 type 频次 → 按频次降序排序。
    说明：上层可用该列表去 KG 查询修复 Trick。
    """
    collected: List[PaperReviewDefect] = []
    counts: Dict[str, int] = {}
    for review in reviews:
        for defect in review.defects:
            defect_type = str(defect.type or "").strip()
            counts[defect_type] = counts.get(defect_type, 0) + 1
            collected.append(defect)

    collected.sort(key=lambda d: counts.get(str(d.type or "").strip(), 0), reverse=True)
    return collected


def decide_issue_type(reviews: Sequence[PaperReviewResult]) -> PaperIssueType:
    """
    功能：基于多个 reviewer 的 issue_type 产出“主问题类型”。
    参数：reviews。
    返回：PaperIssueType。
    流程：
    1) 若出现任何“硬问题”（soundness/novelty/significance）→ 优先返回对应类型；
    2) 否则若出现 clarity_issue → 返回 clarity_issue；
    3) 否则 PASS。
    说明：该规则对应 controller 的回退策略：硬问题回退 PLAN，清晰度回退 WRITE_DRAFT。
    """
    hard_order: List[PaperIssueType] = ["SOUNDNESS_RISK", "NOVELTY_LOW", "SIGNIFICANCE_LOW"]
    review_types = [review.issue_type for review in reviews]
    for hard in hard_order:
        if hard in review_types:
            return hard
    if "CLARITY_ISSUE" in review_types:
        return "CLARITY_ISSUE"
    return "PASS"


def aggregate_reviews(
    reviews: Sequence[PaperReviewResult],
    pass_threshold: float,
    require_individual_threshold: bool = False,
) -> Tuple[float, bool, PaperIssueType]:
    """
    功能：聚合 reviewer 评分并给出是否通过与主问题类型。
    参数：reviews、pass_threshold、require_individual_threshold。
    返回：(avg_score, pass_check, main_issue_type)。
    流程：
    - avg_score：对 overall_score 做 trimmed mean；
    - pass_check：avg_score >= pass_threshold，且（可选）每个 reviewer >= 其 threshold；
    - main_issue_type：用于回退决策。
    说明：当 reviews 为空时，返回 (0.0, False, PASS)。
    """
    if not reviews:
        return 0.0, False, "PASS"

    scores = [float(review.overall_score) for review in reviews]
    avg_score = trimmed_mean(scores)

    pass_check = avg_score >= float(pass_threshold)
    if require_individual_threshold:
        pass_check = pass_check and all(
            float(review.overall_score) >= float(review.threshold) for review in reviews
        )

    main_issue_type = decide_issue_type(reviews)
    return avg_score, pass_check, main_issue_type


class PaperReviewerAgent:
    """
    功能：论文 Reviewer Agent，支持真实 LLM（结构化输出）与离线启发式回退。
    参数：config、llm。
    返回：PaperReviewResult。
    流程：构造 prompt → 调用 LLM → 解析输出 → 归一化字段。
    说明：当 LLM 输出解析失败时，会记录 warning 并回退到启发式评分。
    """

    def __init__(self, config: PaperReviewerConfig, llm: Optional[BaseChatModel] = None) -> None:
        """
        功能：初始化 Reviewer Agent。
        参数：config、llm。
        返回：无。
        流程：保存配置 → 构建 structured_llm 与提示词模板。
        说明：llm=None 时不创建 prompt 模板，直接走离线评分。
        """
        self.config = config
        self.llm = llm
        if llm is None:
            self.prompt = None
            self.fix_prompt = None
            return

        system_prompt = load_paper_prompt(config.prompt_file)
        fix_system_prompt = load_paper_prompt("reviewer_fix_system.txt")
        self.structured_llm = llm.with_structured_output(PaperReviewerOutput, method="json_mode")
        self.fix_structured_llm = llm.with_structured_output(PaperReviewerOutput, method="json_mode")

        self.prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=system_prompt),
                (
                    "human",
                    "变体标识：{nonce}\n"
                    "Reviewer 人设：{persona}\n"
                    "论文草稿（结构化 JSON）：{draft_json}\n"
                    "请严格输出符合 schema 的 JSON。",
                ),
            ]
        )
        self.fix_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=fix_system_prompt),
                ("human", "原始输出：{raw_output}\n请修复为有效 JSON。"),
            ]
        )

    def review(self, draft: StoryDraft) -> PaperReviewResult:
        """
        功能：执行审稿流程并返回结构化结果。
        参数：draft（StoryDraft）。
        返回：PaperReviewResult。
        流程：优先 LLM → 失败则启发式 → 归一化字段。
        说明：该方法不会抛出解析异常（除非出现不可恢复的编程错误）。
        """
        logger.info("PAPER_REVIEWER_START: id=%s", self.config.reviewer_id)
        if self.llm is not None and self.prompt is not None and self.fix_prompt is not None:
            try:
                result = self._review_with_llm(draft)
                logger.info("PAPER_REVIEWER_OK: id=%s mode=llm", self.config.reviewer_id)
                return result
            except Exception as exc:  # noqa: BLE001
                logger.warning("Paper reviewer LLM failed, fallback to heuristic: %s", exc)

        result = self._review_with_heuristic(draft)
        logger.info("PAPER_REVIEWER_OK: id=%s mode=heuristic", self.config.reviewer_id)
        return result

    def _review_with_llm(self, draft: StoryDraft) -> PaperReviewResult:
        """
        功能：调用 LLM 进行结构化审稿。
        参数：draft。
        返回：PaperReviewResult。
        流程：structured invoke → coerce → normalize → 返回。
        说明：若 structured 解析失败，会尝试一次 fix prompt 修复。
        """
        nonce = secrets.token_hex(4)
        # 中文注释：只打包必要内容，避免把整篇全文直接塞给 reviewer。
        review_pack = _build_review_pack(draft)
        draft_json = json.dumps(review_pack, ensure_ascii=False)

        try:
            logger.info("PAPER_REVIEWER_LLM_CALL: id=%s", self.config.reviewer_id)
            response = self.structured_llm.invoke(
                self.prompt.format_messages(
                    nonce=nonce,
                    persona=self.config.persona,
                    draft_json=draft_json,
                )
            )
            output = coerce_model(response, PaperReviewerOutput)
        except Exception as exc:  # noqa: BLE001
            raw = self.llm.invoke(
                self.prompt.format_messages(
                    nonce=nonce,
                    persona=self.config.persona,
                    draft_json=draft_json,
                )
            )
            fix = self.fix_structured_llm.invoke(
                self.fix_prompt.format_messages(raw_output=safe_content(raw))
            )
            output = coerce_model(fix, PaperReviewerOutput)
            logger.info("Paper reviewer repaired output after error: %s", exc)

        scores = output.scores
        normalized_scores = PaperReviewScores(
            novelty=_clamp_score(coerce_float(scores.novelty)),
            soundness=_clamp_score(coerce_float(scores.soundness)),
            clarity=_clamp_score(coerce_float(scores.clarity)),
            significance=_clamp_score(coerce_float(scores.significance)),
        )
        overall = _clamp_score(coerce_float(output.overall_score, default=_calc_overall(normalized_scores)))
        issue_type = _normalize_issue_type(output.issue_type, normalized_scores)

        # 中文注释：strengths/weaknesses/defects 允许为空，但尽量保证为中文数组结构。
        strengths = [str(item).strip() for item in (output.strengths or []) if str(item).strip()]
        weaknesses = [str(item).strip() for item in (output.weaknesses or []) if str(item).strip()]
        defects = []
        for defect in output.defects or []:
            defects.append(
                PaperReviewDefect(
                    type=str(defect.type or "").strip(),
                    location=str(defect.location or "").strip(),
                    detail=str(defect.detail or "").strip(),
                )
            )

        return PaperReviewResult(
            reviewer_id=self.config.reviewer_id,
            persona=self.config.persona,
            threshold=float(self.config.threshold),
            overall_score=overall,
            scores=normalized_scores,
            strengths=strengths,
            weaknesses=weaknesses,
            defects=defects,
            decision=str(output.decision or "").strip(),
            issue_type=issue_type,
        )

    def _review_with_heuristic(self, draft: StoryDraft) -> PaperReviewResult:
        """
        功能：离线启发式审稿（无 LLM 时使用）。
        参数：draft。
        返回：PaperReviewResult。
        流程：基于文本特征生成稳定分数 → 推断 issue_type。
        说明：该实现的目标是“可跑、可测”，并非真实审稿能力。
        """
        text = f"{draft.idea}\n{draft.domain}\n{draft.paper_markdown or draft.final_story}"
        rand = self._rand(text)

        # 中文注释：粗略规则——长度越充分，soundness/clarity 越高；创新词出现则 novelty 越高。
        length = len(text)
        novelty = _clamp_score(5.0 + 4.0 * rand + (1.0 if "新颖" in text or "创新" in text else 0.0))
        soundness = _clamp_score(5.0 + 3.0 * rand + (1.0 if "实验" in text else 0.0))
        clarity = _clamp_score(4.5 + 3.5 * rand + min(2.0, length / 800))
        significance = _clamp_score(5.0 + 3.0 * rand + (1.0 if "应用" in text or "价值" in text else 0.0))
        scores = PaperReviewScores(novelty=novelty, soundness=soundness, clarity=clarity, significance=significance)
        overall = _calc_overall(scores)
        issue_type = _normalize_issue_type("", scores)

        return PaperReviewResult(
            reviewer_id=self.config.reviewer_id,
            persona=self.config.persona,
            threshold=float(self.config.threshold),
            overall_score=overall,
            scores=scores,
            strengths=["结构完整，具备基本叙事框架。"],
            weaknesses=["需要进一步结合领域细节强化说服力。"],
            defects=[],
            decision="borderline",
            issue_type=issue_type,
        )

    def _rand(self, text: str) -> float:
        """
        功能：生成与输入文本稳定相关的伪随机数（0-1）。
        参数：text。
        返回：浮点数。
        流程：md5 哈希 → 取前若干位 → 归一化。
        说明：保证同输入在离线模式下结果可复现，便于测试。
        """
        digest = hashlib.md5(f"{self.config.reviewer_id}:{text}".encode("utf-8")).hexdigest()
        return (int(digest[:6], 16) % 1000) / 1000.0


class PaperReviewerGroup:
    """
    功能：论文 Reviewer 组，负责批量调用多个 reviewer 并返回结果列表。
    参数：reviewers。
    返回：PaperReviewResult 列表。
    流程：遍历 reviewers → 调用 review → 汇总。
    说明：为了保持简单，本实现串行调用；后续可替换为并发实现。
    """

    def __init__(self, reviewers: Sequence[PaperReviewerAgent]) -> None:
        """
        功能：初始化 ReviewerGroup。
        参数：reviewers。
        返回：无。
        流程：保存 reviewers 列表。
        说明：reviewers 顺序可用于日志展示。
        """
        self.reviewers = list(reviewers)

    def review(self, draft: StoryDraft) -> List[PaperReviewResult]:
        """
        功能：对同一论文草稿执行多 reviewer 评审。
        参数：draft。
        返回：PaperReviewResult 列表。
        流程：逐个调用 reviewer.review → 汇总。
        说明：上层将对结果做聚合与决策。
        """
        return [reviewer.review(draft) for reviewer in self.reviewers]


def build_default_paper_reviewers(
    llm_map: Optional[Dict[str, Optional[BaseChatModel]]] = None,
) -> PaperReviewerGroup:
    """
    功能：构建默认论文 Reviewer 组（4 人设）。
    参数：llm_map（可选，用于注入不同 reviewer 的 LLM）。
    返回：PaperReviewerGroup。
    流程：定义 reviewer 配置 → 构建 agent → 组装 group。
    说明：
    - llm_map 的 key 建议为 reviewer_id，例如 "theory"/"experiment"/"novelty"/"practical"。
    - 若对应 llm 为空，则该 reviewer 自动回退到离线启发式。
    """
    llm_map = llm_map or {}
    configs = [
        PaperReviewerConfig(
            reviewer_id="theory",
            persona="理论严谨型（特别关注 soundness/theory/proof）",
            prompt_file="reviewer_theory_system.txt",
            threshold=7.0,
        ),
        PaperReviewerConfig(
            reviewer_id="experiment",
            persona="实验严谨型（特别关注 experiments/ablation/reproducibility）",
            prompt_file="reviewer_experiment_system.txt",
            threshold=7.0,
        ),
        PaperReviewerConfig(
            reviewer_id="novelty",
            persona="新颖性导向型（特别关注 novelty/innovation/impact）",
            prompt_file="reviewer_novelty_system.txt",
            threshold=7.5,
        ),
        PaperReviewerConfig(
            reviewer_id="practical",
            persona="实用性导向型（特别关注 significance/applicability/clarity）",
            prompt_file="reviewer_practical_system.txt",
            threshold=6.5,
        ),
    ]
    reviewers: List[PaperReviewerAgent] = []
    for cfg in configs:
        reviewers.append(PaperReviewerAgent(cfg, llm=llm_map.get(cfg.reviewer_id)))
    return PaperReviewerGroup(reviewers)


def pick_fix_tricks_for_defects(
    defects: Sequence[PaperReviewDefect],
    graph_store: IGraphStore,
    top_k_per_defect: int = 2,
) -> List[Trick]:
    """
    功能：根据 defects 从 KG 中挑选修复 Tricks。
    参数：defects、graph_store、top_k_per_defect。
    返回：Trick 列表（去重）。
    流程：按 defect.type 查询 fixed_by → 合并去重 → 返回。
    说明：
    - graph_store 期望实现 IGraphStore.query_tricks_for_defect。
    - 返回的 Trick.effectiveness 通常携带 success_rate（由底层实现决定）。
    """
    seen: set[str] = set()
    selected: List[Trick] = []
    for defect in defects:
        defect_type = str(defect.type or "").strip()
        if not defect_type:
            continue
        try:
            tricks = graph_store.query_tricks_for_defect(defect_type, top_k=top_k_per_defect)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Query fix tricks failed for defect=%s: %s", defect_type, exc)
            continue
        for trick in tricks:
            if trick.id in seen:
                continue
            seen.add(trick.id)
            selected.append(trick)
    return selected
