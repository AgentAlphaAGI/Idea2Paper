# 中文注释: 测试 Pandoc 渲染器的预/后处理与渲染代理。

from paper_kg.models import PaperSectionOutput
from paper_kg.pandoc_latex_renderer_agent import PandocLatexRendererAgent
import pytest

from paper_kg.pandoc_renderer import (
    replace_cite_candidates,
    sanitize_pandoc_latex_fragment,
    strip_top_heading,
    unwrap_outer_markdown_fence,
)


class _FakePandocRunner:
    def __init__(self, output: str) -> None:
        self.output = output
        self.last_markdown = ""

    def to_latex(self, markdown: str) -> str:
        self.last_markdown = markdown
        return self.output


def test_strip_top_heading() -> None:
    md = "# 引言\n\n正文内容"
    stripped = strip_top_heading(md)
    assert stripped == "正文内容"


def test_replace_cite_candidates() -> None:
    md = "内容 <!-- CITE_CANDIDATE:sec_x_001 --> 继续"
    replaced = replace_cite_candidates(md, {"sec_x_001": "Smith2020"})
    assert "\\cite{Smith2020}" in replaced
    assert "CITE_CANDIDATE" not in replaced


def test_sanitize_pandoc_latex_fragment_removes_tightlist() -> None:
    tex = "hello\n\\tightlist\nworld"
    sanitized = sanitize_pandoc_latex_fragment(tex)
    assert "\\tightlist" not in sanitized
    assert "hello" in sanitized
    assert "world" in sanitized


def test_pandoc_renderer_agent_render_sections() -> None:
    section = PaperSectionOutput(
        section_id="introduction",
        title="引言",
        content_markdown="# 引言\n\n内容 <!-- CITE_CANDIDATE:sec_x_001 -->",
    )
    citation_map = {"sec_x_001": "Smith2020"}
    runner = _FakePandocRunner("内容\n\\tightlist\n\\cite{Smith2020}")
    agent = PandocLatexRendererAgent(pandoc_runner=runner, fix_llm=None)

    updated, _ = agent.render_sections([section], citation_map)
    rendered = updated[0].content_latex or ""

    assert "\\section{引言}" in rendered
    assert "\\tightlist" not in rendered
    assert "\\cite{Smith2020}" in rendered


def test_unwrap_outer_markdown_fence() -> None:
    md = "```markdown\n# Title\nBody\n```"
    assert unwrap_outer_markdown_fence(md).startswith("# Title")


def test_pandoc_content_loss_raises() -> None:
    section = PaperSectionOutput(
        section_id="introduction",
        title="Introduction",
        content_markdown="# Introduction\n\nThis is a long body of text.",
    )
    runner = _FakePandocRunner("UNRELATED OUTPUT")
    agent = PandocLatexRendererAgent(
        pandoc_runner=runner,
        fix_llm=None,
        consistency_threshold=0.95,
        fail_on_content_loss=True,
    )
    with pytest.raises(RuntimeError):
        agent.render_sections([section], {})
