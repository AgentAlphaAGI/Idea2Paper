# 中文注释: 本文件实现逐章节写作（Markdown）与自检/局部重写逻辑。
from __future__ import annotations

import json
import logging
import re
import secrets
from typing import Dict, List, Optional, Tuple

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from core.config import PaperWorkflowConfig
from core.llm_utils import coerce_model, extract_json, log_llm_output, safe_content
from paper_kg.markdown_renderer import (
    inject_cite_markers,
    normalize_markdown_for_pandoc,
    render_section_markdown,
)
from paper_kg.models import (
    PaperPlan,
    PaperSectionOutput,
    PaperSectionPlan,
    SectionContentSpec,
)
from paper_kg.prompts import load_paper_prompt


logger = logging.getLogger(__name__)
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_MARKDOWN_FENCE_RE = re.compile(r"^```")


class PaperSectionWriter:
    """
    功能：论文分章节写作器（逐章生成 + 自检 + 局部重写 + LaTeX 格式保证）。
    参数：llm/config/pattern_guides。
    返回：PaperSectionOutput。
    流程：结构化输出 → 自检 → 缺失则重写 → 兜底模板。
    说明：LaTeX 由后续 Pandoc 渲染阶段统一生成。
    """

    def __init__(
        self,
        llm: Optional[BaseChatModel],
        config: PaperWorkflowConfig,
        pattern_guides: str,
    ) -> None:
        """
        功能：初始化 Section Writer。
        参数：llm、config、pattern_guides。
        返回：无。
        流程：构建 structured_llm 与提示词模板。
        说明：llm=None 时仅走兜底模板生成。
        """
        self.llm = llm
        self.config = config
        self.pattern_guides = pattern_guides
        self.write_prompt = None
        self.quality_rewrite_prompt = None
        self.fix_prompt = None

        if llm is None:
            return

        self.structured_llm = llm.with_structured_output(PaperSectionOutput, method="json_mode")
        markdown_backend = (self.config.markdown_writer_backend or "markdown").lower()
        if markdown_backend == "spec":
            write_system = load_paper_prompt("write_section_spec_system.txt")
        else:
            write_system = load_paper_prompt("write_section_system.txt")
        quality_rewrite_system = load_paper_prompt("rewrite_section_quality_system.txt")
        fix_system = load_paper_prompt("section_fix_system.txt")

        self.write_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=write_system),
                (
                    "human",
                    "变体标识：{nonce}\n"
                    "尝试次数：{attempt}\n"
                    "章节计划：{section_plan}\n"
                    "依赖摘要：{prior_summaries}\n"
                    "全局术语/主张：{global_terms}\n"
                    "写作指导（pattern_guides）：{pattern_guides}\n"
                    "请输出 JSON（PaperSectionOutput）。",
                ),
            ]
        )
        self.quality_rewrite_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=quality_rewrite_system),
                (
                    "human",
                    "变体标识：{nonce}\n"
                    "尝试次数：{attempt}\n"
                    "写作后端：{markdown_backend}\n"
                    "章节计划：{section_plan}\n"
                    "依赖摘要：{prior_summaries}\n"
                    "质量问题：{quality_issues}\n"
                    "原始章节：{current_section}\n"
                    "写作指导（pattern_guides）：{pattern_guides}\n"
                    "请输出修订后的 JSON（PaperSectionOutput）。",
                ),
            ]
        )
        self.fix_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=fix_system),
                ("human", "原始输出：{raw_output}\n请修复为有效 JSON。"),
            ]
        )

    def write_section(
        self,
        plan: PaperPlan,
        section_plan: PaperSectionPlan,
        prior_summaries: Dict[str, str],
        attempt: int = 1,
    ) -> PaperSectionOutput:
        """
        功能：生成单章节内容，并在必要时局部重写。
        参数：plan、section_plan、prior_summaries、attempt。
        返回：PaperSectionOutput。
        流程：LLM 生成 → 自检 → 缺失重写 → 兜底模板。
        说明：llm=None 时会使用兜底输出。
        """
        if self.llm is None or self.write_prompt is None:
            logger.warning(
                "SECTION_WRITE_FALLBACK: id=%s reason=%s",
                section_plan.section_id,
                "llm_missing_or_prompt_missing",
            )
            return self._fallback_section(section_plan, "离线模式兜底输出")

        nonce = secrets.token_hex(4)
        payload = {
            "nonce": nonce,
            "attempt": attempt,
            "section_plan": json.dumps(section_plan.model_dump(), ensure_ascii=False),
            "prior_summaries": json.dumps(prior_summaries, ensure_ascii=False),
            "global_terms": json.dumps(plan.terms, ensure_ascii=False),
            "pattern_guides": self.pattern_guides,
        }

        try:
            logger.info(
                "SECTION_WRITE_LLM_START: id=%s attempt=%s reason=initial_write",
                section_plan.section_id,
                attempt,
            )
            response = self.structured_llm.invoke(self.write_prompt.format_messages(**payload))
            if bool(self.config.log_llm_outputs):
                log_llm_output(
                    logger,
                    f"SECTION_WRITE:{section_plan.section_id}",
                    response,
                    int(self.config.log_llm_output_max_chars),
                )
            output = coerce_model(response, PaperSectionOutput)
        except Exception as exc:  # noqa: BLE001
            logger.warning("SECTION_WRITE_FAIL: %s", exc)
            output = self._repair_output(
                section_plan.section_id,
                payload,
                reason="structured_output_invalid",
            )
            if output is None:
                return self._fallback_section(section_plan, "结构化输出失败，兜底输出")

        output = self._normalize_output(section_plan, output)
        output = self._normalize_markdown_heading(section_plan, output)

        quality_issues = self._validate_markdown_output(section_plan, output)
        if quality_issues:
            stats = self._markdown_quality_stats(section_plan, output)
            logger.warning(
                "SECTION_MARKDOWN_INVALID: id=%s issues=%s stats=%s",
                section_plan.section_id,
                quality_issues,
                stats,
            )
            output = self._attempt_markdown_repair(
                section_plan,
                prior_summaries,
                output,
                payload,
                quality_issues,
                attempt=attempt,
            )

        return output

    def _repair_output(
        self,
        section_id: str,
        payload: Dict[str, object],
        reason: str,
    ) -> Optional[PaperSectionOutput]:
        """
        功能：当结构化输出失败时进行修复。
        参数：payload（写作提示参数）。
        返回：PaperSectionOutput 或 None。
        流程：直接解析 JSON → 不行则使用修复提示词 → 再解析。
        说明：仅在 LLM 输出不符合 schema 时使用。
        """
        if self.llm is None or self.write_prompt is None:
            return None

        try:
            logger.info(
                "SECTION_STRUCTURED_REPAIR_START: id=%s reason=%s",
                section_id,
                reason,
            )
            response = self.llm.invoke(self.write_prompt.format_messages(**payload))
            if bool(self.config.log_llm_outputs):
                log_llm_output(
                    logger,
                    f"SECTION_REPAIR_RAW:{section_id}",
                    response,
                    int(self.config.log_llm_output_max_chars),
                )
            raw_output = safe_content(response)
            data = extract_json(raw_output)
            return PaperSectionOutput.model_validate(data)
        except Exception as exc:  # noqa: BLE001
            logger.warning("SECTION_REPAIR_PARSE_FAIL: id=%s error=%s", section_id, exc)
            pass

        if self.fix_prompt is None:
            return None
        try:
            logger.info(
                "SECTION_STRUCTURED_REPAIR_JSON_FIX_START: id=%s reason=raw_json_invalid",
                section_id,
            )
            response = self.llm.invoke(self.fix_prompt.format_messages(raw_output=raw_output))
            if bool(self.config.log_llm_outputs):
                log_llm_output(
                    logger,
                    f"SECTION_REPAIR_FIX:{section_id}",
                    response,
                    int(self.config.log_llm_output_max_chars),
                )
            fixed_raw = safe_content(response)
            data = extract_json(fixed_raw)
            return PaperSectionOutput.model_validate(data)
        except Exception as exc:  # noqa: BLE001
            logger.warning("SECTION_STRUCTURED_REPAIR_JSON_FIX_FAIL: id=%s error=%s", section_id, exc)
            return None

    def _rewrite_for_markdown_quality(
        self,
        section_plan: PaperSectionPlan,
        prior_summaries: Dict[str, str],
        current_output: PaperSectionOutput,
        quality_issues: List[str],
        attempt: int,
    ) -> PaperSectionOutput:
        """
        功能：基于 Markdown 质量问题的修复重写。
        参数：section_plan/prior_summaries/current_output/quality_issues/attempt。
        返回：PaperSectionOutput。
        说明：用于修复字数/格式等问题，避免混入缺失要点逻辑。
        """
        if self.llm is None or self.quality_rewrite_prompt is None:
            return self._fallback_section(section_plan, "Markdown 质量修复失败，兜底输出")

        for retry in range(1, int(self.config.section_max_retries) + 1):
            logger.info(
                "SECTION_QUALITY_REWRITE_START: id=%s attempt=%s",
                section_plan.section_id,
                f"{attempt}-{retry}",
            )
            nonce = secrets.token_hex(4)
            payload = {
                "nonce": nonce,
                "attempt": f"{attempt}-{retry}",
                "markdown_backend": (self.config.markdown_writer_backend or "markdown"),
                "section_plan": json.dumps(section_plan.model_dump(), ensure_ascii=False),
                "prior_summaries": json.dumps(prior_summaries, ensure_ascii=False),
                "quality_issues": "；".join(quality_issues) or "无",
                "current_section": self._select_content(current_output),
                "pattern_guides": self.pattern_guides,
            }
            try:
                logger.info(
                    "SECTION_QUALITY_REWRITE_LLM_START: id=%s attempt=%s",
                    section_plan.section_id,
                    f"{attempt}-{retry}",
                )
                response = self.structured_llm.invoke(
                    self.quality_rewrite_prompt.format_messages(**payload)
                )
                if bool(self.config.log_llm_outputs):
                    log_llm_output(
                        logger,
                        f"SECTION_QUALITY_REWRITE:{section_plan.section_id}:{attempt}-{retry}",
                        response,
                        int(self.config.log_llm_output_max_chars),
                    )
                output = coerce_model(response, PaperSectionOutput)
            except Exception as exc:  # noqa: BLE001
                logger.warning("SECTION_QUALITY_REWRITE_FAIL: %s", exc)
                return self._fallback_section(section_plan, "Markdown 质量修复失败，兜底输出")

            output = self._normalize_output(section_plan, output)
            issues = self._validate_markdown_output(section_plan, output)
            if not issues:
                logger.info("SECTION_QUALITY_REWRITE_OK: id=%s", section_plan.section_id)
                return output
            logger.info(
                "SECTION_QUALITY_REWRITE_VALIDATE_FAIL: id=%s issues=%s",
                section_plan.section_id,
                issues,
            )

        logger.warning("SECTION_QUALITY_REWRITE_FAIL: exceed retries")
        return self._fallback_section(section_plan, "Markdown 质量修复失败，兜底输出")

    def _attempt_markdown_repair(
        self,
        section_plan: PaperSectionPlan,
        prior_summaries: Dict[str, str],
        output: PaperSectionOutput,
        payload: Dict[str, object],
        issues: List[str],
        attempt: int,
    ) -> PaperSectionOutput:
        if not issues:
            return output

        if self.llm is None or self.write_prompt is None:
            if bool(self.config.strict_markdown_validation):
                raise RuntimeError(
                    f"Section markdown invalid and llm unavailable: {section_plan.section_id}"
                )
            return self._fallback_section(section_plan, "Markdown 质量校验失败")

        backend = (self.config.markdown_writer_backend or "markdown").lower()
        needs_structured_repair = False
        if backend == "spec":
            if output.content_spec is None or not _has_spec_blocks(output.content_spec):
                needs_structured_repair = True
        else:
            if not (output.content_markdown or "").strip():
                needs_structured_repair = True

        if needs_structured_repair:
            repaired = self._repair_output(
                section_plan.section_id,
                payload,
                reason="content_missing",
            )
            if repaired is not None:
                output = self._normalize_output(section_plan, repaired)
                output = self._normalize_markdown_heading(section_plan, output)
                issues = self._validate_markdown_output(section_plan, output)
                if not issues:
                    return output

        max_retries = int(self.config.section_min_quality_max_retries)
        for retry in range(max_retries):
            output = self._rewrite_for_markdown_quality(
                section_plan,
                prior_summaries,
                output,
                issues,
                attempt=attempt + retry,
            )
            output = self._normalize_output(section_plan, output)
            output = self._normalize_markdown_heading(section_plan, output)
            issues = self._validate_markdown_output(section_plan, output)
            if not issues:
                return output

        if bool(self.config.strict_markdown_validation):
            raise RuntimeError(
                f"Section markdown invalid after repair: {section_plan.section_id}, issues={issues}"
            )
        return self._fallback_section(section_plan, "Markdown 质量校验失败")

    def _normalize_output(
        self, section_plan: PaperSectionPlan, output: PaperSectionOutput
    ) -> PaperSectionOutput:
        """
        功能：规范化 LLM 输出，补齐字段。
        参数：section_plan、output。
        返回：PaperSectionOutput。
        流程：修正标题/字数/摘要长度。
        说明：避免 LLM 输出缺字段导致下游异常。
        """
        output.section_id = section_plan.section_id
        if not output.title:
            output.title = section_plan.title or section_plan.section_id
        output = self._render_markdown_if_needed(section_plan, output)
        content = self._select_content(output)
        output.word_count = _count_words(content)
        output.summary = _limit_text(
            output.summary or _summarize_text(content),
            int(self.config.section_summary_max_chars),
        )
        return output

    def _render_markdown_if_needed(
        self, section_plan: PaperSectionPlan, output: PaperSectionOutput
    ) -> PaperSectionOutput:
        backend = (self.config.markdown_writer_backend or "markdown").lower()
        if backend != "spec":
            return output
        spec = output.content_spec
        if spec is None or not _has_spec_blocks(spec):
            if self._should_fail_fast():
                raise RuntimeError(
                    f"Section content_spec missing: {section_plan.section_id}"
                )
            return output
        title = output.title or section_plan.title or section_plan.section_id
        md = render_section_markdown(
            section_id=section_plan.section_id,
            title=title,
            spec=spec,
            cite_token=self.config.spec_cite_token,
        )
        md = inject_cite_markers(
            md,
            section_id=section_plan.section_id,
            cite_token=self.config.spec_cite_token,
            require_cite_ids=bool(self.config.spec_render_require_cite_ids),
        )
        output.content_markdown = normalize_markdown_for_pandoc(md)
        return output

    def _markdown_quality_stats(
        self, section_plan: PaperSectionPlan, output: PaperSectionOutput
    ) -> Dict[str, object]:
        md = output.content_markdown or ""
        body_text = _strip_markdown_for_count(md)
        word_count = _count_words(body_text)
        min_ratio = float(self.config.section_min_word_ratio)
        min_target = int(section_plan.target_words_min or 0)
        min_required = int(min_target * min_ratio) if min_target > 0 and min_ratio > 0 else 0
        first_lines = [line for line in md.splitlines() if line.strip()][:3]
        return {
            "computed_word_count": word_count,
            "min_required": min_required,
            "md_chars": len(md),
            "first_3_lines": first_lines,
        }

    def _should_enforce_english(self) -> bool:
        if not bool(self.config.enforce_english):
            return False
        return str(self.config.paper_language or "").lower().startswith("en")

    def _normalize_markdown_heading(
        self, section_plan: PaperSectionPlan, output: PaperSectionOutput
    ) -> PaperSectionOutput:
        if not output.content_markdown:
            return output

        title = output.title or section_plan.title or section_plan.section_id
        output.title = title
        lines = output.content_markdown.splitlines()
        first_idx = _first_non_empty_index(lines)
        if first_idx is None:
            return output

        first_line = lines[first_idx].lstrip()
        if _MARKDOWN_FENCE_RE.match(first_line):
            return output

        if first_line.startswith("#"):
            heading = first_line.lstrip("#").strip()
            if heading:
                output.title = heading
                lines[first_idx] = f"# {output.title}"
                output.content_markdown = "\n".join(lines)
                return output

        lines.insert(first_idx, f"# {output.title}")
        output.content_markdown = "\n".join(lines)
        return output

    def _validate_markdown_output(
        self, section_plan: PaperSectionPlan, output: PaperSectionOutput
    ) -> List[str]:
        if not self._should_validate_markdown():
            return []
        md = output.content_markdown or ""
        issues: List[str] = []

        if not md.strip():
            issues.append("content_markdown_empty")
            return issues

        if _is_wrapped_in_markdown_fence(md):
            issues.append("content_markdown_wrapped_in_code_fence")

        heading = _extract_markdown_heading(md)
        if not heading:
            issues.append("missing_top_level_heading")
        elif output.title and _canonicalize_title(heading) != _canonicalize_title(output.title):
            issues.append("heading_title_mismatch")

        min_ratio = float(self.config.section_min_word_ratio)
        min_target = int(section_plan.target_words_min or 0)
        if min_target > 0 and min_ratio > 0:
            min_required = int(min_target * min_ratio)
            body_text = _strip_markdown_for_count(md)
            word_count = _count_words(body_text)
            if word_count < min_required:
                issues.append(f"word_count_below_min:{word_count}<{min_required}")

        if self._should_enforce_english():
            body_text = _strip_markdown_for_count(md)
            ratio = _calc_cjk_ratio(body_text)
            max_ratio = float(self.config.max_cjk_ratio)
            if ratio > max_ratio:
                issues.append(f"non_english_ratio_high:{ratio:.3f}>{max_ratio}")

        return issues

    def _should_validate_markdown(self) -> bool:
        return True

    def _should_fail_fast(self) -> bool:
        if not bool(self.config.quality_fail_fast):
            return False
        return str(self.config.paper_language or "").lower().startswith("en")

    def _fallback_section(self, section_plan: PaperSectionPlan, reason: str) -> PaperSectionOutput:
        """
        功能：生成兜底章节内容，保证流程不中断。
        参数：section_plan、reason。
        返回：PaperSectionOutput。
        流程：模板拼接 + 标记 todo。
        说明：离线或 LLM 失败时使用。
        """
        if self._should_fail_fast():
            raise RuntimeError(
                f"Section fallback triggered: {section_plan.section_id} reason={reason}"
            )
        title = section_plan.title or section_plan.section_id
        language = str(self.config.paper_language or "").lower()
        if language.startswith("en"):
            content_markdown = "\n".join(
                [
                    f"# {title}",
                    "",
                    f"> Fallback output: {reason}",
                    "",
                    "## Required points",
                    "\n".join([f"- {p}" for p in section_plan.required_points]) or "- None",
                    "",
                    "## Required tricks",
                    "\n".join([f"- {t}" for t in section_plan.required_tricks]) or "- None",
                    "",
                    "(Content pending.)",
                ]
            )
            todo = ["Content pending", "Check required points"]
        else:
            content_markdown = "\n".join(
                [
                    f"# {title}",
                    "",
                    f"> 兜底输出：{reason}",
                    "",
                    "## 需要覆盖的要点",
                    "\n".join([f"- {p}" for p in section_plan.required_points]) or "- 无",
                    "",
                    "## 需要使用的技巧",
                    "\n".join([f"- {t}" for t in section_plan.required_tricks]) or "- 无",
                    "",
                    "（正文待补充）",
                ]
            )
            todo = ["待补充正文", "检查要点/技巧"]
        content_latex = _latex_placeholder_section(section_plan.section_id, title, reason)
        return PaperSectionOutput(
            section_id=section_plan.section_id,
            title=title,
            content_markdown=content_markdown,
            content_latex=content_latex,
            used_tricks=[],
            covered_points=[],
            todo=todo,
            word_count=_count_words(content_markdown),
            summary=_limit_text(content_markdown, int(self.config.section_summary_max_chars)),
        )

    def _select_content(self, output: PaperSectionOutput) -> str:
        """
        功能：根据输出格式选择内容字段。
        参数：output。
        返回：文本内容。
        说明：优先使用 Markdown，必要时回退到 LaTeX。
        """
        return output.content_markdown or output.content_latex or ""


def _first_non_empty_index(lines: List[str]) -> Optional[int]:
    for idx, line in enumerate(lines):
        if line.strip():
            return idx
    return None


def _is_wrapped_in_markdown_fence(text: str) -> bool:
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return False
    if lines[0].lstrip().startswith("```") and lines[-1].strip() == "```":
        return True
    return False


def _extract_markdown_heading(text: str) -> str:
    lines = text.splitlines()
    first_idx = _first_non_empty_index(lines)
    if first_idx is None:
        return ""
    first_line = lines[first_idx].lstrip()
    if first_line.startswith("#"):
        return first_line.lstrip("#").strip()
    return ""


def _strip_markdown_for_count(text: str) -> str:
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"<!--\s*CITE_CANDIDATE.*?-->", "", text)
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    return text


def _has_spec_blocks(spec: Optional[SectionContentSpec]) -> bool:
    return bool(spec and spec.blocks)


def _canonicalize_title(text: str) -> str:
    value = (text or "").lower()
    value = re.sub(r"[:;,.\(\)\[\]\{\}\-—]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _calc_cjk_ratio(text: str) -> float:
    if not text:
        return 0.0
    cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
    total = len([ch for ch in text if not ch.isspace()])
    if total == 0:
        return 0.0
    return cjk / total


def _count_words(text: str) -> int:
    """
    功能：粗略统计字数/词数。
    参数：text。
    返回：整数。
    流程：中文按字符计数，英文按空白分词计数。
    说明：用于质量门槛与日志统计。
    """
    if not text:
        return 0
    if _CJK_RE.search(text):
        return len("".join(text.split()))
    return len([w for w in re.split(r"\s+", text.strip()) if w])


def _latex_placeholder_section(section_id: str, title: str, reason: str) -> str:
    """
    功能：生成可编译的 LaTeX 占位章节。
    参数：section_id/title/reason。
    返回：LaTeX 文本。
    说明：用于修复失败的最终兜底。
    """
    if section_id == "abstract":
        return f"TODO: {reason}.\n"
    safe_title = title or "Section"
    return f"\\section{{{safe_title}}}\nTODO: {reason}.\n"


def _summarize_text(text: str, max_chars: int = 200) -> str:
    """
    功能：生成简单摘要。
    参数：text、max_chars。
    返回：摘要字符串。
    流程：取前 N 字符。
    说明：用于无 LLM 时的简易摘要。
    """
    return _limit_text(text.replace("\n", " "), max_chars)


def _limit_text(text: str, max_chars: int) -> str:
    """
    功能：截断文本到指定长度。
    参数：text、max_chars。
    返回：截断文本。
    流程：长度判断 → 截断。
    说明：保证摘要输入不会过长。
    """
    if max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    return text[: max_chars].rstrip() + "…"


def _extract_trick_name(instruction: str) -> str:
    """
    功能：从写作指令中提取 Trick 名称。
    参数：instruction。
    返回：名称字符串。
    流程：优先按“名称：”切分 → 回退到提取前半句。
    说明：用于 required_tricks 与 used_tricks 的比对。
    """
    if "：" in instruction:
        return instruction.split("：", 1)[0].strip()
    # 中文注释：回退策略，取前 10 个字以降低误判。
    return instruction[:10].strip()
