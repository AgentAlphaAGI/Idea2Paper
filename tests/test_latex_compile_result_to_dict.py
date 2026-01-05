from paper_kg.latex_toolchain import CompileResult


def test_compile_result_to_dict() -> None:
    result = CompileResult(
        ok=True,
        stdout="ok",
        stderr="",
        log_excerpt="",
        errors=[],
        cmd=["latexmk"],
    )
    payload = result.to_dict()
    assert payload["ok"] is True
    assert payload["stdout"] == "ok"
    assert payload["errors"] == []
