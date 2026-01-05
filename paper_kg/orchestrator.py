# 中文注释: 本文件提供基于图谱的故事初版生成编排器。
from __future__ import annotations

import json
import logging
import random
import re
import secrets
from typing import List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, ConfigDict

from paper_kg.interfaces import IEmbeddingModel, IGraphStore, IVectorStore
from paper_kg.llm_schemas import PaperReviewDefect
from paper_kg.models import (
    Domain,
    Idea,
    NoveltyCheckResult,
    Pattern,
    StoryDraft,
    StorySkeleton,
    Trick,
)
from paper_kg.novelty import check_novelty
from paper_kg.prompts import load_paper_prompt
from paper_kg.sampling import downweight_scores, select_patterns_diverse
from core.llm_utils import coerce_model, safe_content


logger = logging.getLogger(__name__)


class LLMStoryOutput(BaseModel):
    """
    功能：LLM 输出结构化模型。
    参数：story_skeleton、final_story。
    返回：LLMStoryOutput。
    流程：字段校验 → 序列化。
    说明：用于 structured output 解析。
    """

    model_config = ConfigDict(extra="ignore")

    story_skeleton: StorySkeleton
    final_story: str


class StoryOrchestrator:
    """
    功能：基于知识图谱生成故事初版。
    参数：graph_store、vector_store、embedding_model、rng、llm。
    返回：StoryDraft。
    流程：选套路 → 新颖性检查 → 构造故事。
    说明：支持 LLM 与模板两种模式。
    """

    def __init__(
        self,
        graph_store: IGraphStore,
        vector_store: IVectorStore,
        embedding_model: IEmbeddingModel,
        rng: Optional[random.Random] = None,
        llm: Optional[BaseChatModel] = None,
        max_resample: int = 3,
        penalty: float = 0.3,
        guide_limit: int = 2000,
        max_guides: int = 3,
    ) -> None:
        """
        功能：初始化编排器。
        参数：graph_store、vector_store、embedding_model、rng、llm、max_resample、penalty、guide_limit、max_guides。
        返回：无。
        流程：保存依赖 → 构建 LLM 提示词模板。
        说明：llm 为空时使用规则模板。
        """
        self.graph_store = graph_store
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.rng = rng or random.Random()
        self.max_resample = max_resample
        self.penalty = penalty
        self.llm = llm
        self.guide_limit = guide_limit
        self.max_guides = max_guides
        if llm is not None:
            self.structured_llm = llm.with_structured_output(LLMStoryOutput, method="json_mode")
            system_prompt = load_paper_prompt("generate_story_system.txt")
            self.prompt = ChatPromptTemplate.from_messages(
                [
                    SystemMessage(content=system_prompt),
                    (
                        "human",
                        """
变体标识：{nonce}
核心 Idea：{idea}
目标领域：{domain}
已选套路：{patterns}
已选 Tricks：{tricks}
写作指导（pattern_guides）：{pattern_guides}
请输出 JSON，包含 story_skeleton 与 final_story。
""",
                    ),
                ]
            )
        else:
            self.prompt = None

    def _slugify(self, text: str) -> str:
        """
        功能：生成简易 ID。
        参数：text。
        返回：slug 字符串。
        流程：小写 → 替换非字母数字。
        说明：用于自定义 idea/domain ID。
        """
        lower = text.strip().lower()
        slug = re.sub(r"[^a-z0-9]+", "_", lower).strip("_")
        return slug or "custom"

    def _resolve_idea(self, idea_text: str) -> Idea:
        """
        功能：从图谱解析或创建 Idea。
        参数：idea_text。
        返回：Idea。
        流程：匹配 core_idea → 回退创建。
        说明：避免找不到节点导致失败。
        """
        for node in self.graph_store.list_nodes("Idea"):
            desc = str(node.get("core_idea", ""))
            if idea_text in desc or desc in idea_text:
                return Idea.from_dict(node)
        return Idea(
            id=f"idea_{self._slugify(idea_text)}",
            core_idea=idea_text,
            tech_stack=[],
            input_type="",
            output_type="",
        )

    def _resolve_domain(self, domain_name: str) -> Domain:
        """
        功能：从图谱解析或创建 Domain。
        参数：domain_name。
        返回：Domain。
        流程：匹配 name → 回退创建。
        说明：若无该领域则生成临时节点。
        """
        for node in self.graph_store.list_nodes("Domain"):
            name = str(node.get("name", ""))
            if domain_name.lower() == name.lower():
                return Domain.from_dict(node)
        return Domain(id=f"domain_{self._slugify(domain_name)}", name=domain_name, paper_count=0)

    def _collect_patterns(self, idea: Idea, domain: Domain) -> List[Pattern]:
        """
        功能：收集候选套路。
        参数：idea、domain。
        返回：Pattern 列表。
        流程：查询 idea+domain → 回退全量套路。
        说明：保障采样阶段有候选集。
        """
        candidates = self.graph_store.query_patterns_for_idea_domain(idea.id, domain.id)
        if candidates:
            return candidates
        all_nodes = self.graph_store.list_nodes("Pattern")
        return [Pattern.from_dict(node) for node in all_nodes]

    def _format_pattern_guides(self, patterns: List[Pattern]) -> str:
        """
        功能：构建写作指导字符串。
        参数：patterns。
        返回：JSON 字符串。
        流程：截断 writing_guide → 组装为字典 → JSON 序列化。
        说明：用于注入 LLM 提示词。
        """
        guides = {}
        for pattern in patterns[: self.max_guides]:
            guide = pattern.writing_guide or ""
            if len(guide) > self.guide_limit:
                guide = guide[: self.guide_limit]
            guides[pattern.pattern_name] = guide
        return json.dumps(guides, ensure_ascii=False)

    def _build_skeleton(
        self,
        idea: Idea,
        domain: Domain,
        patterns: List[Pattern],
        tricks: List[Trick],
    ) -> StorySkeleton:
        """
        功能：构造规则化故事框架。
        参数：idea、domain、patterns、tricks。
        返回：StorySkeleton。
        流程：拼接中文模板。
        说明：不依赖 LLM 也能输出。
        """
        pattern_names = "、".join([p.pattern_name for p in patterns]) or "关键叙事套路"
        trick_names = "、".join([t.name for t in tricks]) or "关键技巧"
        return StorySkeleton(
            problem_framing=f"围绕 {idea.core_idea} 在 {domain.name} 领域的核心问题展开，强调实际痛点与研究价值。",
            gap_pattern=f"指出现有方法在 {domain.name} 中存在效率、稳定性或可解释性不足，形成研究空白。",
            method_story=f"提出 {idea.core_idea} 的解决方案，结合 {pattern_names} 与 {trick_names} 组织方法叙述。",
            experiments_story="设计对比实验、消融实验与跨域验证，强调可复现性与稳健性。",
        )

    def _build_final_story(self, skeleton: StorySkeleton) -> str:
        """
        功能：将故事框架合并为最终文案。
        参数：skeleton。
        返回：中文故事文本。
        流程：按段落拼接。
        说明：用于控制台最终稿。
        """
        return (
            f"问题定义：{skeleton.problem_framing}\n"
            f"研究缺口：{skeleton.gap_pattern}\n"
            f"方法叙述：{skeleton.method_story}\n"
            f"实验策略：{skeleton.experiments_story}"
        )

    def build_story(
        self,
        idea_text: str,
        domain_name: str,
        max_patterns: int = 3,
        temperature: float = 0.8,
        novelty_threshold: float = 0.85,
    ) -> StoryDraft:
        """
        功能：生成故事初版。
        参数：idea_text、domain_name、max_patterns、temperature、novelty_threshold。
        返回：StoryDraft。
        流程：采样套路 → 新颖性检查 → 生成故事。
        说明：包含降权重采样策略。
        """
        idea = self._resolve_idea(idea_text)
        domain = self._resolve_domain(domain_name)
        candidates = self._collect_patterns(idea, domain)

        pattern_ids = [p.id for p in candidates]
        scores = [self.graph_store.query_pattern_effectiveness(p.id, domain.id) for p in candidates]

        selected: List[Pattern] = []
        novelty_result = NoveltyCheckResult(
            is_novel=True, max_similarity=0.0, top_similar=[], suggestion=None
        )

        for _attempt in range(self.max_resample + 1):
            selected = select_patterns_diverse(
                candidates, scores, max_patterns, temperature, self.rng
            )
            novelty_result = check_novelty(
                selected,
                vector_store=self.vector_store,
                embedding_model=self.embedding_model,
                threshold=novelty_threshold,
                top_k=10,
            )
            if novelty_result.is_novel:
                break

            penalized_ids: List[str] = []
            for item in novelty_result.top_similar:
                for pid in item.metadata.get("pattern_ids", []):
                    penalized_ids.append(pid)
            if not penalized_ids:
                break
            scores = downweight_scores(pattern_ids, scores, penalized_ids, self.penalty)
            logger.info("Novelty hit, downweight patterns: %s", sorted(set(penalized_ids)))

        tricks: List[Trick] = []
        for pattern in selected:
            trick_nodes = self.graph_store.get_tricks_for_pattern(pattern.id)
            for node in trick_nodes:
                tricks.append(Trick.from_dict(node))

        if self.llm is not None and self.prompt is not None:
            nonce = secrets.token_hex(4)
            response = self.structured_llm.invoke(
                self.prompt.format_messages(
                    nonce=nonce,
                    idea=idea.core_idea,
                    domain=domain.name,
                    patterns="、".join([p.pattern_name for p in selected]),
                    tricks="、".join([t.name for t in tricks]),
                    pattern_guides=self._format_pattern_guides(selected),
                )
            )
            output = coerce_model(response, LLMStoryOutput)
            skeleton = output.story_skeleton
            final_story = output.final_story
        else:
            skeleton = self._build_skeleton(idea, domain, selected, tricks)
            final_story = self._build_final_story(skeleton)

        return StoryDraft(
            idea=idea.core_idea,
            domain=domain.name,
            patterns=selected,
            tricks=tricks,
            story_skeleton=skeleton,
            novelty=novelty_result,
            final_story=final_story,
        )


class PaperDraftWriter:
    """
    功能：论文草稿撰写器（生成初稿 + 基于缺陷重写）。
    参数：llm（可选）。
    返回：StorySkeleton + final_story。
    流程：优先 LLM 结构化输出；无 LLM 或失败时回退到模板拼接。
    说明：
    - 与 StoryOrchestrator 解耦：工作流可先“选套路/查重”，再调用本 writer 写稿与改稿。
    - prompt 使用 prompts_paper/ 目录（可用 PAPER_PROMPT_DIR 覆盖），不会影响其他模块。
    """

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        guide_limit: int = 2000,
        max_guides: int = 3,
    ) -> None:
        """
        功能：初始化 DraftWriter。
        参数：llm、guide_limit、max_guides。
        返回：无。
        流程：若有 llm 则构建 structured_llm 与提示词模板。
        说明：llm=None 时仅启用模板模式。
        """
        self.llm = llm
        self.guide_limit = guide_limit
        self.prompt = None
        self.rewrite_prompt = None
        self.fix_prompt = None
        self.max_guides = max_guides
        if llm is None:
            return

        self.structured_llm = llm.with_structured_output(LLMStoryOutput, method="json_mode")
        self.structured_fix_llm = llm.with_structured_output(LLMStoryOutput, method="json_mode")

        system_prompt = load_paper_prompt("generate_story_system.txt")
        rewrite_system_prompt = load_paper_prompt("rewrite_story_system.txt")
        fix_system_prompt = load_paper_prompt("story_fix_system.txt")

        self.prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=system_prompt),
                (
                    "human",
                    "变体标识：{nonce}\n"
                    "尝试次数：{attempt}\n"
                    "核心 Idea：{idea}\n"
                    "目标领域：{domain}\n"
                    "已选套路：{patterns}\n"
                    "已选 Tricks：{tricks}\n"
                    "写作指导（pattern_guides）：{pattern_guides}\n"
                    "请输出 JSON（story_skeleton + final_story）。",
                ),
            ]
        )
        self.rewrite_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=rewrite_system_prompt),
                (
                    "human",
                    "变体标识：{nonce}\n"
                    "尝试次数：{attempt}\n"
                    "核心 Idea：{idea}\n"
                    "目标领域：{domain}\n"
                    "已选套路：{patterns}\n"
                    "已选 Tricks：{tricks}\n"
                    "写作指导（pattern_guides）：{pattern_guides}\n"
                    "当前草稿：{current_draft}\n"
                    "需要修复的缺陷：{defects}\n"
                    "建议采用的修复 Tricks：{fix_tricks}\n"
                    "请输出修订后的 JSON（story_skeleton + final_story）。",
                ),
            ]
        )
        self.fix_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=fix_system_prompt),
                ("human", "原始输出：{raw_output}\n请修复为有效 JSON。"),
            ]
        )

    def _format_pattern_guides(self, patterns: List[Pattern], limit: int) -> str:
        """
        功能：构建写作指导字符串。
        参数：patterns、limit。
        返回：JSON 字符串。
        流程：截断 writing_guide → 组装为字典 → JSON 序列化。
        说明：用于注入 LLM 提示词。
        """
        guides = {}
        for pattern in patterns[: self.max_guides]:
            guide = pattern.writing_guide or ""
            if len(guide) > limit:
                guide = guide[:limit]
            guides[pattern.pattern_name] = guide
        return json.dumps(guides, ensure_ascii=False)

    def build_pattern_guides(self, patterns: List[Pattern]) -> str:
        """
        功能：生成写作指导的 JSON 字符串（对外公开方法）。
        参数：patterns。
        返回：JSON 字符串。
        流程：调用内部 _format_pattern_guides。
        说明：供 controller/Web UI 复用，避免重复实现截断逻辑。
        """
        return self._format_pattern_guides(patterns, self.guide_limit)

    def write_draft(
        self,
        idea: str,
        domain: str,
        patterns: List[Pattern],
        tricks: List[Trick],
        attempt: int = 1,
    ) -> tuple[StorySkeleton, str]:
        """
        功能：生成论文初稿（故事框架 + 整合稿）。
        参数：idea、domain、patterns、tricks、attempt。
        返回：(StorySkeleton, final_story)。
        流程：LLM → 失败则模板 → 返回。
        说明：attempt/nonce 用于增加输出多样性与可追踪性。
        """
        if self.llm is None or self.prompt is None:
            logger.info("PAPER_DRAFT_START: mode=template attempt=%s", attempt)
            skeleton = self._build_skeleton(idea, domain, patterns, tricks)
            logger.info("PAPER_DRAFT_OK: mode=template")
            return skeleton, self._build_final_story(skeleton)

        nonce = secrets.token_hex(4)
        payload = {
            "nonce": nonce,
            "attempt": attempt,
            "idea": idea,
            "domain": domain,
            "patterns": "、".join([p.pattern_name for p in patterns]),
            "tricks": "、".join([t.name for t in tricks]),
            "pattern_guides": self._format_pattern_guides(patterns, self.guide_limit),
        }
        try:
            logger.info("PAPER_DRAFT_START: mode=llm attempt=%s", attempt)
            response = self.structured_llm.invoke(self.prompt.format_messages(**payload))
            output = coerce_model(response, LLMStoryOutput)
            logger.info("PAPER_DRAFT_OK: mode=llm")
            return output.story_skeleton, output.final_story
        except Exception as exc:  # noqa: BLE001
            logger.warning("PAPER_DRAFT_FAIL: structured output failed, repair: %s", exc)
            raw = self.llm.invoke(self.prompt.format_messages(**payload))
            fix = self.structured_fix_llm.invoke(
                self.fix_prompt.format_messages(raw_output=safe_content(raw))
            )
            output = coerce_model(fix, LLMStoryOutput)
            logger.info("PAPER_DRAFT_OK: mode=repair")
            return output.story_skeleton, output.final_story

    def rewrite_draft(
        self,
        idea: str,
        domain: str,
        patterns: List[Pattern],
        tricks: List[Trick],
        current_skeleton: StorySkeleton,
        current_story: str,
        defects: List[PaperReviewDefect],
        fix_tricks: List[Trick],
        attempt: int = 1,
    ) -> tuple[StorySkeleton, str]:
        """
        功能：根据缺陷与修复建议重写论文草稿。
        参数：idea、domain、patterns、tricks、current_skeleton、current_story、defects、fix_tricks、attempt。
        返回：(StorySkeleton, final_story)。
        流程：LLM 重写 → 失败则规则修订 → 返回。
        说明：
        - 该方法用于 controller 的 CLARITY_ISSUE 回退（保持套路不变）。
        - 若 llm=None，则采用“在模板中补充/强调缺陷修复点”的简单方式。
        """
        if self.llm is None or self.rewrite_prompt is None:
            logger.info("PAPER_REWRITE_START: mode=template attempt=%s", attempt)
            skeleton = self._rewrite_skeleton_by_rules(current_skeleton, defects, fix_tricks)
            story = self._build_final_story(skeleton)
            logger.info("PAPER_REWRITE_OK: mode=template")
            return skeleton, story

        nonce = secrets.token_hex(4)
        payload = {
            "nonce": nonce,
            "attempt": attempt,
            "idea": idea,
            "domain": domain,
            "patterns": "、".join([p.pattern_name for p in patterns]),
            "tricks": "、".join([t.name for t in tricks]),
            "pattern_guides": self._format_pattern_guides(patterns, self.guide_limit),
            "current_draft": current_story,
            "defects": "；".join([f"{d.type}:{d.detail}" for d in defects if d.type or d.detail])
            or "无",
            "fix_tricks": "、".join([t.name for t in fix_tricks]) or "无",
        }
        try:
            logger.info("PAPER_REWRITE_START: mode=llm attempt=%s", attempt)
            response = self.structured_llm.invoke(self.rewrite_prompt.format_messages(**payload))
            output = coerce_model(response, LLMStoryOutput)
            logger.info("PAPER_REWRITE_OK: mode=llm")
            return output.story_skeleton, output.final_story
        except Exception as exc:  # noqa: BLE001
            logger.warning("PAPER_REWRITE_FAIL: fallback to rules: %s", exc)
            skeleton = self._rewrite_skeleton_by_rules(current_skeleton, defects, fix_tricks)
            return skeleton, self._build_final_story(skeleton)

    def _build_skeleton(
        self,
        idea: str,
        domain: str,
        patterns: List[Pattern],
        tricks: List[Trick],
    ) -> StorySkeleton:
        """
        功能：构造规则化故事框架（模板模式）。
        参数：idea、domain、patterns、tricks。
        返回：StorySkeleton。
        流程：拼接中文模板。
        说明：用于无 LLM 或兜底场景。
        """
        pattern_names = "、".join([p.pattern_name for p in patterns]) or "关键叙事套路"
        trick_names = "、".join([t.name for t in tricks]) or "关键技巧"
        return StorySkeleton(
            problem_framing=f"围绕 {idea} 在 {domain} 领域的核心问题展开，强调实际痛点与研究价值。",
            gap_pattern=f"指出现有方法在 {domain} 中存在效率、稳定性或可解释性不足，形成研究空白。",
            method_story=f"提出 {idea} 的解决方案，结合 {pattern_names} 与 {trick_names} 组织方法叙述。",
            experiments_story="设计对比实验、消融实验与跨域验证，强调可复现性与稳健性。",
        )

    def _rewrite_skeleton_by_rules(
        self,
        skeleton: StorySkeleton,
        defects: List[PaperReviewDefect],
        fix_tricks: List[Trick],
    ) -> StorySkeleton:
        """
        功能：在无 LLM 情况下做轻量“规则重写”。
        参数：skeleton、defects、fix_tricks。
        返回：新的 StorySkeleton。
        流程：根据 defect.type 追加/强化对应段落的表达 → 返回新骨架。
        说明：目标是让工作流在离线模式也能体现“回退改稿”的效果。
        """
        defect_types = {str(d.type or "").strip() for d in defects if str(d.type or "").strip()}
        fix_names = "、".join([t.name for t in fix_tricks]) if fix_tricks else ""

        new_problem = skeleton.problem_framing
        new_gap = skeleton.gap_pattern
        new_method = skeleton.method_story
        new_exp = skeleton.experiments_story

        if "clarity_low" in defect_types:
            new_problem += "（重写：用更短句、更明确的定义与符号说明关键概念）"
            new_method += "（重写：补充步骤列表与关键模块的直觉解释）"
        if "theory_missing" in defect_types:
            new_method += "（补充：加入形式化定义、复杂度分析或收敛性证明）"
        if "experiments_weak" in defect_types:
            new_exp += "（补充：加强基线对比、消融实验与复现细节）"
        if "novelty_low" in defect_types:
            new_gap += "（补充：明确与现有工作差异，并强调不可替代性）"
        if "significance_low" in defect_types:
            new_exp += "（补充：增加真实场景/指标，量化实际收益与影响）"

        if fix_names:
            new_exp += f"（建议采用 Tricks：{fix_names}）"

        return StorySkeleton(
            problem_framing=new_problem,
            gap_pattern=new_gap,
            method_story=new_method,
            experiments_story=new_exp,
        )

    def _build_final_story(self, skeleton: StorySkeleton) -> str:
        """
        功能：将故事框架合并为最终文案。
        参数：skeleton。
        返回：中文整合稿字符串。
        流程：按段落拼接。
        说明：与模板模式输出格式保持一致，便于控制台展示。
        """
        return (
            f"问题定义：{skeleton.problem_framing}\n"
            f"研究缺口：{skeleton.gap_pattern}\n"
            f"方法叙述：{skeleton.method_story}\n"
            f"实验策略：{skeleton.experiments_story}"
        )
