# Deterministic Markdown renderer tests.

from paper_kg.markdown_renderer import (
    inject_cite_markers,
    normalize_markdown_for_pandoc,
    render_section_markdown,
)
from paper_kg.models import (
    MarkdownBullets,
    MarkdownCode,
    MarkdownEquation,
    MarkdownHeading,
    MarkdownParagraph,
    SectionContentSpec,
)


def test_markdown_renderer_injects_citations() -> None:
    spec = SectionContentSpec(
        blocks=[
            MarkdownParagraph(
                text="We introduce the setting and motivate the problem. [[CITE]]"
            ),
            MarkdownHeading(level=2, text="Background"),
            MarkdownBullets(items=["First point.", "Second point. [[CITE]]"]),
            MarkdownEquation(latex="E = mc^2"),
            MarkdownCode(language="python", code="print('hello')"),
        ]
    )
    md = render_section_markdown("introduction", "Introduction", spec, cite_token="[[CITE]]")
    md = inject_cite_markers(md, "introduction", cite_token="[[CITE]]", require_cite_ids=True)
    normalized = normalize_markdown_for_pandoc(md)

    assert normalized.splitlines()[0] == "# Introduction"
    assert "<!-- CITE_CANDIDATE:introduction_001 -->" in normalized
    assert "<!-- CITE_CANDIDATE:introduction_002 -->" in normalized
    assert "[[CITE]]" not in normalized
    assert "## Background" in normalized
