# 中文注释: Pandoc LaTeX 渲染 + LLM 修复代理。
from __future__ import annotations

import difflib
import logging
import re
from typing import Dict, List, Optional, Set, Tuple

from langchain_core.language_models.chat_models import BaseChatModel

from paper_kg.latex_utils import normalize_latex_newlines
from paper_kg.latex_renderer import render_sections_to_latex
from paper_kg.latex_renderer_agent import LatexRendererAgent
from paper_kg.models import PaperSectionOutput
from paper_kg.pandoc_renderer import (
    IPandocRunner,
    replace_cite_candidates,
    sanitize_for_pandoc,
    sanitize_pandoc_latex_fragment,
    strip_top_heading,
    unwrap_outer_markdown_fence,
)


logger = logging.getLogger(__name__)


class PandocLatexRendererAgent:
    """Pandoc 渲染 + compile_overleaf 失败后的 LLM 修复。"""

    def __init__(
        self,
        pandoc_runner: IPandocRunner,
        fix_llm: Optional[BaseChatModel] = None,
        fallback_renderer: Optional[LatexRendererAgent] = None,
        consistency_threshold: float = 0.65,
        fail_on_content_loss: bool = True,
    ) -> None:
        self.pandoc_runner = pandoc_runner
        self.fix_agent = LatexRendererAgent(fix_llm=fix_llm) if fix_llm else None
        self.fallback_renderer = fallback_renderer
        self.consistency_threshold = consistency_threshold
        self.fail_on_content_loss = fail_on_content_loss

    def render_sections(
        self, sections: List[PaperSectionOutput], citation_map: Dict[str, str]
    ) -> Tuple[List[PaperSectionOutput], str]:
        logger.info(
            "PANDOC_RENDER_SECTIONS_START: sections=%s citations=%s",
            len(sections),
            len(citation_map),
        )
        if not self._pandoc_available():
            logger.warning("PANDOC_RENDER_SKIP: pandoc 不可用，走 fallback 渲染")
            return self._render_with_fallback(sections, citation_map)

        updated: List[PaperSectionOutput] = []
        parts: List[str] = []
        for section in sections:
            section_id = section.section_id
            md_body = strip_top_heading(section.content_markdown or "")
            md_body = unwrap_outer_markdown_fence(md_body)
            md_body = replace_cite_candidates(md_body, citation_map)
            md_body = sanitize_for_pandoc(md_body)
            latex_body = self.pandoc_runner.to_latex(md_body)
            if not self._pandoc_available():
                logger.warning("PANDOC_RENDER_ABORT: pandoc 运行失败，走 fallback 渲染")
                return self._render_with_fallback(sections, citation_map)
            latex_body = sanitize_pandoc_latex_fragment(latex_body)
            if md_body.strip() and not latex_body.strip():
                message = f"Pandoc returned empty output for section={section_id}"
                if self.fail_on_content_loss:
                    raise RuntimeError(message)
                logger.warning("PANDOC_RENDER_EMPTY: %s", message)
                return self._render_with_fallback(sections, citation_map)
            if not md_body.strip() and not latex_body.strip():
                latex_body = "TODO: Content pending.\n"
            if md_body.strip() and not _content_consistent(
                md_body,
                latex_body,
                self.consistency_threshold,
            ):
                message = f"Pandoc output appears inconsistent for section={section_id}"
                if self.fail_on_content_loss:
                    raise RuntimeError(message)
                logger.warning("PANDOC_RENDER_INCONSISTENT: %s", message)
                return self._render_with_fallback(sections, citation_map)
            if section_id == "abstract":
                section_tex = latex_body.strip() + "\n"
            else:
                title = section.title or section.section_id
                section_tex = f"\\section{{{_escape_latex_title(title)}}}\n{latex_body}".strip() + "\n"
            section_tex = normalize_latex_newlines(section_tex)
            updated.append(section.model_copy(update={"content_latex": section_tex}))
            parts.append(section_tex)
            logger.info(
                "PANDOC_RENDER_SECTION_OK: section=%s md_chars=%s latex_chars=%s",
                section_id,
                len(md_body),
                len(section_tex),
            )
        return updated, "\n\n".join([p for p in parts if p.strip()])

    def fix_sections(
        self,
        sections: List[PaperSectionOutput],
        compile_log: str,
        citation_map: Dict[str, str],
        targets: Optional[Set[str]] = None,
    ) -> Tuple[List[PaperSectionOutput], str]:
        if self.fix_agent is None:
            logger.info("PANDOC_FIX_SKIP: 未配置修复 LLM，直接返回现有 LaTeX")
            return self._join_sections(sections)
        logger.info(
            "PANDOC_FIX_START: sections=%s targets=%s log_chars=%s",
            len(sections),
            len(targets or []),
            len(compile_log or ""),
        )
        return self.fix_agent.fix_sections(
            sections,
            compile_log=compile_log,
            citation_map=citation_map,
            targets=targets,
        )

    def _pandoc_available(self) -> bool:
        return bool(getattr(self.pandoc_runner, "available", True))

    def _render_with_fallback(
        self,
        sections: List[PaperSectionOutput],
        citation_map: Dict[str, str],
    ) -> Tuple[List[PaperSectionOutput], str]:
        if self.fallback_renderer is not None:
            logger.info("PANDOC_RENDER_FALLBACK: 使用备用渲染器回退")
            return self.fallback_renderer.render_sections(sections, citation_map)
        logger.info("PANDOC_RENDER_FALLBACK: 使用内置确定性渲染回退")
        return render_sections_to_latex(sections, citation_map)

    def _join_sections(
        self, sections: List[PaperSectionOutput]
    ) -> Tuple[List[PaperSectionOutput], str]:
        parts = [s.content_latex for s in sections if s.content_latex and s.content_latex.strip()]
        return sections, "\n\n".join(parts)


def _escape_latex_title(text: str) -> str:
    if not text:
        return text
    replacements = [
        ("{", r"\\{"),
        ("}", r"\\}"),
        ("_", r"\\_"),
        ("%", r"\\%"),
        ("&", r"\\&"),
        ("#", r"\\#"),
        ("$", r"\\$"),
    ]
    result = text
    for char, escaped in replacements:
        result = re.sub(rf"(?<!\\){re.escape(char)}", escaped, result)
    return result


def _strip_markdown_for_compare(text: str) -> str:
    text = re.sub(r"<!--\\s*CITE_CANDIDATE.*?-->", "", text)
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"^#+\\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\\cite[a-zA-Z]*\{[^}]*\}", "", text)
    text = re.sub(r"[*_`]", "", text)
    return text


def _strip_latex_for_compare(text: str) -> str:
    text = re.sub(r"%.*", "", text)
    text = re.sub(r"\\cite[a-zA-Z]*\{[^}]*\}", "", text)
    text = re.sub(r"\\[A-Za-z]+(\[[^\\]]*\\])?\{([^}]*)\}", r"\\2", text)
    text = re.sub(r"\\[A-Za-z]+", "", text)
    text = text.replace("{", "").replace("}", "")
    return text


def _content_consistent(markdown: str, latex: str, threshold: float) -> bool:
    md = re.sub(r"\\s+", "", _strip_markdown_for_compare(markdown))
    lx = re.sub(r"\\s+", "", _strip_latex_for_compare(latex))
    if not md:
        return True
    if md in lx:
        return True
    ratio = difflib.SequenceMatcher(None, md, lx).ratio()
    return ratio >= threshold
