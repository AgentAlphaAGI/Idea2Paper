# 中文注释: 测试 LaTeX-only 修复 agent 的基础行为。

from paper_kg.latex_renderer_agent import LatexRendererAgent
from paper_kg.models import PaperSectionOutput


class _FakeLLM:
    def __init__(self, content: str) -> None:
        self.content = content

    def invoke(self, _messages):  # noqa: ANN001
        class _Resp:
            def __init__(self, content):
                self.content = content

        return _Resp(self.content)


def test_latex_fix_adds_math_mode() -> None:
    section = PaperSectionOutput(
        section_id="method",
        title="方法",
        content_markdown="# 方法\n\n多维注意力机制的计算复杂度为 O(n^3)。",
        content_latex="\\section{方法}\n多维注意力机制的计算复杂度为 O(n^3)。",
    )
    fix_llm = _FakeLLM("\\section{方法}\n多维注意力机制的计算复杂度为 $O(n^3)$。")
    agent = LatexRendererAgent(llm=None, fix_llm=fix_llm)

    updated, _ = agent.fix_sections(
        [section],
        compile_log="Missing $ inserted",
        citation_map={},
        targets={"method"},
    )

    assert "$O(n^3)$" in (updated[0].content_latex or "")
