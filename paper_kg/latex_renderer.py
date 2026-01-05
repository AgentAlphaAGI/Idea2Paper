# 中文注释: 将 Markdown 渲染为 LaTeX（不改内容，只做格式转换与引用插入）。
from __future__ import annotations

import re
from typing import Dict, List, Tuple

from paper_kg.latex_utils import normalize_latex_newlines
from paper_kg.models import PaperSectionOutput


_MARKER_RE = re.compile(r"<!--\s*CITE_CANDIDATE(?::([A-Za-z0-9:_-]+))?\s*-->")


def render_sections_to_latex(
    sections: List[PaperSectionOutput],
    citation_map: Dict[str, str],
) -> Tuple[List[PaperSectionOutput], str]:
    """
    功能：批量渲染章节 Markdown 为 LaTeX。
    参数：sections、citation_map(candidate_id->bibkey)。
    返回：更新后的 sections + 拼接后的 paper_latex。
    """
    updated: List[PaperSectionOutput] = []
    parts: List[str] = []
    for section in sections:
        latex_body = render_markdown_to_latex(section.content_markdown, citation_map)
        if not latex_body.strip():
            latex_body = "TODO: 内容待补充。\n"
        if section.section_id == "abstract":
            section_tex = latex_body
        else:
            title = section.title or section.section_id
            section_tex = f"\\section{{{_escape_latex_text(title)}}}\n{latex_body}"
        section_tex = normalize_latex_newlines(section_tex)
        updated.append(section.model_copy(update={"content_latex": section_tex}))
        parts.append(section_tex)
    return updated, "\n\n".join([p for p in parts if p.strip()])


def render_markdown_to_latex(markdown: str, citation_map: Dict[str, str]) -> str:
    """
    功能：将 Markdown 文本转为 LaTeX 正文。
    参数：markdown、citation_map。
    返回：LaTeX 文本。
    说明：仅处理常见结构，不改变语义。
    """
    if not markdown:
        return ""

    text, token_map = _replace_markers_with_tokens(markdown)
    lines = text.splitlines()
    out_lines: List[str] = []
    in_list = False
    in_code = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            if not in_code:
                out_lines.append("\\begin{verbatim}")
            else:
                out_lines.append("\\end{verbatim}")
            in_code = not in_code
            continue

        if in_code:
            out_lines.append(line)
            continue

        if stripped.startswith("#"):
            level = len(stripped) - len(stripped.lstrip("#"))
            heading = stripped.lstrip("#").strip()
            if level >= 2:
                cmd = "\\subsection" if level == 2 else "\\subsubsection"
                out_lines.append(f"{cmd}{{{_escape_latex_text(heading)}}}")
            # level 1 heading 忽略（由外层 \\section 提供）
            continue

        if stripped.startswith(("- ", "* ")):
            if not in_list:
                out_lines.append("\\begin{itemize}")
                in_list = True
            item_text = stripped[2:].strip()
            out_lines.append(f"  \\item {_escape_latex_text(item_text)}")
            continue

        if in_list:
            out_lines.append("\\end{itemize}")
            in_list = False

        out_lines.append(_escape_latex_text(line))

    if in_list:
        out_lines.append("\\end{itemize}")

    latex = "\n".join(out_lines)
    latex = _restore_citation_tokens(latex, token_map, citation_map)
    latex = _MARKER_RE.sub("", latex)
    return latex


def _replace_markers_with_tokens(markdown: str) -> Tuple[str, Dict[str, str]]:
    token_map: Dict[str, str] = {}
    pieces: List[str] = []
    cursor = 0
    index = 0
    for match in _MARKER_RE.finditer(markdown):
        pieces.append(markdown[cursor : match.start()])
        cand_id = (match.group(1) or "").strip()
        if not cand_id:
            # 未带 id 的 marker 直接移除（应在 extractor 阶段补齐）
            cursor = match.end()
            continue
        token = f"[[CITETOKEN{index}]]"
        token_map[token] = cand_id
        pieces.append(token)
        cursor = match.end()
        index += 1
    pieces.append(markdown[cursor:])
    return "".join(pieces), token_map


def _restore_citation_tokens(
    latex: str, token_map: Dict[str, str], citation_map: Dict[str, str]
) -> str:
    for token, cand_id in token_map.items():
        bibkey = citation_map.get(cand_id, "")
        replacement = f"\\cite{{{bibkey}}}" if bibkey else ""
        latex = latex.replace(token, replacement)
    return latex


def _escape_latex_text(text: str) -> str:
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
