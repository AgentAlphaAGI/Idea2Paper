# Deterministic Markdown renderer for section content specs.
from __future__ import annotations

from typing import Iterable, List

from paper_kg.models import (
    MarkdownBlock,
    MarkdownBullets,
    MarkdownCode,
    MarkdownEquation,
    MarkdownHeading,
    MarkdownParagraph,
    SectionContentSpec,
)


def render_section_markdown(
    section_id: str,
    title: str,
    spec: SectionContentSpec,
    cite_token: str = "[[CITE]]",
) -> str:
    """Render a section spec into Markdown without relying on LLM formatting."""
    lines: List[str] = [f"# {title}".strip()]
    for block in spec.blocks:
        block_lines = _render_block(block)
        if not block_lines:
            continue
        if lines and lines[-1].strip() != "":
            lines.append("")
        lines.extend(block_lines)
    rendered = "\n".join(lines)
    return rendered if cite_token in rendered else rendered


def inject_cite_markers(
    markdown: str,
    section_id: str,
    cite_token: str = "[[CITE]]",
    require_cite_ids: bool = True,
) -> str:
    """Replace cite tokens with stable CITE_CANDIDATE markers (skip code fences)."""
    if not markdown or cite_token not in markdown:
        return markdown

    lines = markdown.splitlines()
    in_code = False
    counter = 1
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            continue
        if in_code or cite_token not in line:
            continue
        parts = line.split(cite_token)
        if len(parts) == 1:
            continue
        rebuilt = parts[0]
        for tail in parts[1:]:
            marker = (
                f"<!-- CITE_CANDIDATE:{section_id}_{counter:03d} -->"
                if require_cite_ids
                else "<!-- CITE_CANDIDATE -->"
            )
            counter += 1
            if rebuilt and not rebuilt.endswith((" ", "\t")):
                rebuilt += " "
            rebuilt += marker + tail
        lines[idx] = rebuilt
    return "\n".join(lines)


def normalize_markdown_for_pandoc(markdown: str) -> str:
    """Normalize Markdown for Pandoc conversion (no outer fences, stable blanks)."""
    if not markdown:
        return ""
    text = markdown.replace("\r\n", "\n").replace("\r", "\n")
    text = _unwrap_outer_markdown_fence(text)
    lines = [line.rstrip() for line in text.splitlines()]
    normalized: List[str] = []
    blank = False
    for line in lines:
        if line.strip() == "":
            if blank:
                continue
            blank = True
            normalized.append("")
            continue
        blank = False
        normalized.append(line)
    return "\n".join(normalized).strip() + "\n"


def _render_block(block: MarkdownBlock) -> List[str]:
    if isinstance(block, MarkdownParagraph):
        return _render_paragraph(block.text)
    if isinstance(block, MarkdownHeading):
        return _render_heading(block.level, block.text)
    if isinstance(block, MarkdownBullets):
        return _render_bullets(block.items)
    if isinstance(block, MarkdownEquation):
        return _render_equation(block.latex)
    if isinstance(block, MarkdownCode):
        return _render_code(block.code, block.language)
    return []


def _render_paragraph(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    return text.splitlines() or [text]


def _render_heading(level: int, text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    prefix = "##" if level <= 2 else "###"
    return [f"{prefix} {text}"]


def _render_bullets(items: Iterable[str]) -> List[str]:
    lines = []
    for item in items or []:
        value = str(item or "").strip()
        if not value:
            continue
        lines.append(f"- {value}")
    return lines


def _render_equation(latex: str) -> List[str]:
    latex = (latex or "").strip()
    if not latex:
        return []
    return ["$$", latex, "$$"]


def _render_code(code: str, language: str) -> List[str]:
    code = code or ""
    if not code.strip():
        return []
    lang = (language or "").strip()
    lines = [f"```{lang}".rstrip()]
    lines.extend(code.splitlines())
    lines.append("```")
    return lines


def _unwrap_outer_markdown_fence(markdown: str) -> str:
    stripped = markdown.strip()
    if not stripped.startswith("```"):
        return markdown
    lines = stripped.splitlines()
    if len(lines) < 2:
        return markdown
    if not lines[0].strip().startswith("```"):
        return markdown
    if lines[-1].strip() != "```":
        return markdown
    inner = "\n".join(lines[1:-1]).strip()
    return inner if inner else markdown
