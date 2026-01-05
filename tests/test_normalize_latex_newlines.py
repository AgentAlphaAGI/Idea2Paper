# 中文注释: 测试 \\n 字面量归一化逻辑。

from paper_kg.latex_utils import normalize_latex_newlines


def test_normalize_latex_newlines_converts_literal() -> None:
    text = r"\section{相关工作}\n本章概述"
    fixed = normalize_latex_newlines(text)
    assert "\\n" not in fixed
    assert "\n" in fixed


def test_normalize_latex_newlines_preserves_commands() -> None:
    text = r"\newcommand{\foo}{bar} \nabla"
    fixed = normalize_latex_newlines(text)
    assert "\\newcommand" in fixed
    assert "\\nabla" in fixed
