from paper_kg.latex_toolchain import (
    _extract_nonfatal_ref_cite_warnings,
    is_nonfatal_minimal_compile_issue,
)


def test_nonfatal_minimal_compile_issue_detected() -> None:
    log = (
        "Latexmk: Missing bbl file 'main_min.bbl' in following:\n"
        "No file main_min.bbl.\n"
        "Latexmk: ====Undefined refs and citations with line #s in .tex file:\n"
        "Reference `sec:method' on page 1 undefined on input line 7\n"
        "Citation `TODO_x' on page 1 undefined on input line 8\n"
    )
    assert is_nonfatal_minimal_compile_issue(log) is True
    warnings = _extract_nonfatal_ref_cite_warnings(log)
    assert any("Missing bbl file" in warning for warning in warnings)


def test_fatal_minimal_compile_issue_rejected() -> None:
    log = (
        "! LaTeX Error: File `foo.sty' not found.\n"
        "Emergency stop.\n"
        "! Undefined control sequence.\n"
    )
    assert is_nonfatal_minimal_compile_issue(log) is False
