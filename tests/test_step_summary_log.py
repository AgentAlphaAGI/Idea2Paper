import logging
from pathlib import Path

from core.logging import add_step_summary_file_handler


def _attach_handler(path: Path) -> tuple[str, logging.Handler]:
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    handler_path = add_step_summary_file_handler(str(path))
    handler = next(
        h
        for h in root.handlers
        if getattr(h, "name", "") == "paper_workflow_step_summary"
    )
    return handler_path, handler


def _detach_handler(handler: logging.Handler) -> None:
    root = logging.getLogger()
    root.removeHandler(handler)
    handler.close()


def test_step_summary_filters_llm_output(tmp_path: Path) -> None:
    log_path = tmp_path / "steps.log"
    handler_path, handler = _attach_handler(log_path)
    logger = logging.getLogger("test_step_summary")
    logger.setLevel(logging.INFO)
    logger.propagate = True
    try:
        logger.info("LLM_OUTPUT: SECTION_WRITE:intro ...")
        logger.info(
            "PAPER_EXPORT_OVERLEAF_START: run_id=paper_1 dir=./outputs/overleaf"
        )
        logger.warning("PAPER_EXPORT_OVERLEAF_FAIL: boom")
        handler.flush()
        content = Path(handler_path).read_text(encoding="utf-8")
        assert "PAPER_EXPORT_OVERLEAF_START" in content
        assert "PAPER_EXPORT_OVERLEAF_FAIL" in content
        assert "LLM_OUTPUT" not in content
    finally:
        _detach_handler(handler)


def test_step_summary_includes_traceback(tmp_path: Path) -> None:
    log_path = tmp_path / "steps.log"
    handler_path, handler = _attach_handler(log_path)
    logger = logging.getLogger("test_step_summary_error")
    logger.setLevel(logging.INFO)
    logger.propagate = True
    try:
        try:
            1 / 0
        except ZeroDivisionError:
            logger.exception("PAPER_WORKFLOW_FAIL")
        handler.flush()
        content = Path(handler_path).read_text(encoding="utf-8")
        assert "PAPER_WORKFLOW_FAIL" in content
        assert "ZeroDivisionError" in content
    finally:
        _detach_handler(handler)
