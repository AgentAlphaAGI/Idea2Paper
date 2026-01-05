# 中文注释: LLM LaTeX 渲染与格式修复 Agent。
from __future__ import annotations

import difflib
import json
import re
from typing import Dict, List, Optional, Set, Tuple

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from paper_kg.latex_utils import normalize_latex_newlines
from paper_kg.models import PaperSectionOutput
from paper_kg.prompts import load_paper_prompt
from paper_kg.latex_renderer import render_sections_to_latex


_MARKER_RE = re.compile(r"<!--\s*CITE_CANDIDATE(?::([A-Za-z0-9:_-]+))?\s*-->")
_CITE_RE = re.compile(r"\\cite[a-zA-Z]*\{([^}]+)\}")


class LatexRendererAgent:
    """LLM 渲染 + LaTeX-only 修复。"""

    def __init__(
        self,
        llm: Optional[BaseChatModel],
        fix_llm: Optional[BaseChatModel] = None,
        similarity_threshold: float = 0.6,
    ) -> None:
        self.llm = llm
        self.fix_llm = fix_llm or llm
        self.similarity_threshold = similarity_threshold
        self.render_prompt: Optional[ChatPromptTemplate] = None
        self.fix_prompt: Optional[ChatPromptTemplate] = None
        if llm is not None:
            system = load_paper_prompt("latex_renderer_system.txt")
            self.render_prompt = ChatPromptTemplate.from_messages(
                [SystemMessage(content=system), ("human", "输入：{payload}")]
            )
        if self.fix_llm is not None:
            system = load_paper_prompt("latex_format_fix_system.txt")
            self.fix_prompt = ChatPromptTemplate.from_messages(
                [SystemMessage(content=system), ("human", "输入：{payload}")]
            )

    def render_sections(
        self, sections: List[PaperSectionOutput], citation_map: Dict[str, str]
    ) -> Tuple[List[PaperSectionOutput], str]:
        if self.llm is None or self.render_prompt is None:
            return render_sections_to_latex(sections, citation_map)

        updated: List[PaperSectionOutput] = []
        parts: List[str] = []
        for section in sections:
            section_cites = _extract_section_citations(section.content_markdown, citation_map)
            md_body = _strip_section_heading(section.content_markdown or "")
            md_body = _replace_markers_with_cites(md_body, section_cites)
            payload = {
                "section_id": section.section_id,
                "title": section.title or section.section_id,
                "markdown": md_body,
                "citations": section_cites,
            }
            response = self.llm.invoke(self.render_prompt.format_messages(payload=json.dumps(payload, ensure_ascii=False)))
            latex_body = getattr(response, "content", "") or ""
            latex_body = normalize_latex_newlines(latex_body)
            latex_body = _sanitize_latex_body(latex_body, set(section_cites.values()))
            if not latex_body.strip():
                latex_body = "TODO: 内容待补充。\n"
            if section.section_id == "abstract":
                section_tex = latex_body
            else:
                title = section.title or section.section_id
                section_tex = f"\\section{{{_escape_latex_title(title)}}}\n{latex_body}"
            updated.append(section.model_copy(update={"content_latex": section_tex}))
            parts.append(section_tex)
        return updated, "\n\n".join([p for p in parts if p.strip()])

    def fix_sections(
        self,
        sections: List[PaperSectionOutput],
        compile_log: str,
        citation_map: Dict[str, str],
        targets: Optional[Set[str]] = None,
    ) -> Tuple[List[PaperSectionOutput], str]:
        if self.fix_llm is None or self.fix_prompt is None:
            return render_sections_to_latex(sections, citation_map)

        updated: List[PaperSectionOutput] = []
        parts: List[str] = []
        for section in sections:
            if targets and section.section_id not in targets:
                updated.append(section)
                if section.content_latex:
                    parts.append(section.content_latex)
                continue

            latex = section.content_latex or ""
            payload = {
                "section_id": section.section_id,
                "latex": latex,
                "compile_log": compile_log,
            }
            response = self.fix_llm.invoke(self.fix_prompt.format_messages(payload=json.dumps(payload, ensure_ascii=False)))
            fixed = getattr(response, "content", "") or ""
            fixed = normalize_latex_newlines(fixed)
            fixed = _sanitize_latex_body(fixed, set(citation_map.values()))
            if not fixed.strip():
                fixed = latex
            if section.section_id == "abstract":
                fixed = re.sub(r"^\\s*\\section\{.*?\}\\s*", "", fixed, count=1).lstrip()
            elif "\\section{" not in fixed:
                title = section.title or section.section_id
                fixed = f"\\section{{{_escape_latex_title(title)}}}\n{fixed}"

            if not _content_consistent(section.content_markdown or "", fixed, self.similarity_threshold):
                fixed = latex

            updated.append(section.model_copy(update={"content_latex": fixed}))
            if fixed.strip():
                parts.append(fixed)
        return updated, "\n\n".join([p for p in parts if p.strip()])


def _replace_markers_with_cites(text: str, citation_map: Dict[str, str]) -> str:
    def _replace(match: re.Match) -> str:
        cand_id = (match.group(1) or "").strip()
        bibkey = citation_map.get(cand_id, "")
        return f"\\cite{{{bibkey}}}" if bibkey else ""

    return _MARKER_RE.sub(_replace, text)


def _extract_section_citations(markdown: str, citation_map: Dict[str, str]) -> Dict[str, str]:
    if not markdown:
        return {}
    found: Dict[str, str] = {}
    for match in _MARKER_RE.finditer(markdown):
        cand_id = (match.group(1) or "").strip()
        if not cand_id:
            continue
        bibkey = citation_map.get(cand_id, "")
        if bibkey:
            found[cand_id] = bibkey
    return found


def _strip_section_heading(markdown: str) -> str:
    lines = markdown.splitlines()
    if lines and lines[0].lstrip().startswith("#"):
        return "\n".join(lines[1:]).lstrip()
    return markdown


def _sanitize_latex_body(latex: str, allowed_cites: Set[str]) -> str:
    latex = _MARKER_RE.sub("", latex)
    if "\\documentclass" in latex:
        latex = latex.replace("\\documentclass", "")
    if "\\begin{document}" in latex:
        latex = latex.replace("\\begin{document}", "")
    if "\\end{document}" in latex:
        latex = latex.replace("\\end{document}", "")

    for match in _CITE_RE.finditer(latex):
        keys = [k.strip() for k in match.group(1).split(",") if k.strip()]
        if any(k not in allowed_cites for k in keys):
            latex = latex.replace(match.group(0), "")
    return latex


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


def _strip_markdown(text: str) -> str:
    text = _MARKER_RE.sub("", text)
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"^#+\\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"[*_`]", "", text)
    return text


def _strip_latex(text: str) -> str:
    text = re.sub(r"%.*", "", text)
    text = re.sub(r"\\cite[a-zA-Z]*\{[^}]*\}", "", text)
    text = re.sub(r"\\[A-Za-z]+(\[[^\\]]*\\])?\{([^}]*)\}", r"\\2", text)
    text = re.sub(r"\\[A-Za-z]+", "", text)
    text = text.replace("{", "").replace("}", "")
    return text


def _content_consistent(markdown: str, latex: str, threshold: float) -> bool:
    md = re.sub(r"\\s+", "", _strip_markdown(markdown))
    lx = re.sub(r"\\s+", "", _strip_latex(latex))
    if not md:
        return True
    if md in lx:
        return True
    ratio = difflib.SequenceMatcher(None, md, lx).ratio()
    return ratio >= threshold
