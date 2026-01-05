from pathlib import Path

from paper_kg.latex_toolchain import _build_placeholder_bib_entries, _extract_cite_keys_from_tex
from paper_kg.overleaf_project import _build_bib_entries, _collect_cite_keys


def test_extract_cite_keys_from_tex() -> None:
    text = "See \\\\cite{TODO_a, TODO_b} and \\\\citet{TODO_c}."
    keys = _extract_cite_keys_from_tex(text)
    assert keys == ["TODO_a", "TODO_b", "TODO_c"]
    bib = _build_placeholder_bib_entries(keys)
    assert "@misc{TODO_a," in bib


def test_collect_cite_keys(tmp_path: Path) -> None:
    sections_dir = tmp_path / "sections"
    sections_dir.mkdir()
    tex = "\\\\section{Intro}\\nSome text \\\\cite{TODO_a, TODO_b} and \\\\citep{TODO_c}.\\n"
    (sections_dir / "intro.tex").write_text(tex, encoding="utf-8")
    keys = _collect_cite_keys(sections_dir)
    assert keys == ["TODO_a", "TODO_b", "TODO_c"]
    bib = _build_bib_entries(keys)
    assert "@misc{TODO_a," in bib
