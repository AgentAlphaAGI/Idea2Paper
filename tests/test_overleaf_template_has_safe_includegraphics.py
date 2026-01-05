from pathlib import Path


def test_overleaf_template_has_safe_includegraphics() -> None:
    template_path = Path("paper_kg/latex_templates/article_ctex/main.tex.template")
    content = template_path.read_text(encoding="utf-8")
    assert "\\renewcommand{\\includegraphics}" in content
