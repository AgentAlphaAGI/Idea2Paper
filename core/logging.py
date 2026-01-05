# 中文注释: 本文件包含可运行代码与流程说明。
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional


class _StepSummaryFilter(logging.Filter):
    _PATTERN = re.compile(
        r"("
        r"PAPER_WORKFLOW_START|"
        r"PAPER_WORKFLOW_OK|"
        r"PAPER_WORKFLOW_FAIL|"
        r"PAPER_STEP_LOG_PATH|"
        r"PAPER_PARSE|"
        r"PAPER_BLUEPRINT|"
        r"PAPER_SECTION|"
        r"PAPER_ASSEMBLE_|"
        r"PAPER_CITATION_|"
        r"PAPER_RENDER_LATEX_|"
        r"PAPER_EXPORT_OVERLEAF_|"
        r"PAPER_COMPILE_LATEX_|"
        r"SECTION_MARKDOWN_INVALID|"
        r"SECTION_QUALITY_REWRITE_|"
        r"SECTION_REWRITE_|"
        r"SECTION_STRUCTURED_REPAIR_|"
        r"PANDOC_|"
        r"OVERLEAF_"
        r")"
    )

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.ERROR:
            return True
        message = record.getMessage()
        if "LLM_OUTPUT" in message:
            return False
        return bool(self._PATTERN.search(message))


def _default_step_log_path() -> Path:
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return Path("outputs") / "logs" / f"paper_workflow_steps_{timestamp}.log"


def add_step_summary_file_handler(
    path: str, level: int = logging.INFO
) -> str:
    target = Path(path) if path else _default_step_log_path()
    if target.suffix != ".log":
        target.mkdir(parents=True, exist_ok=True)
        target = target / _default_step_log_path().name
    else:
        target.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    handler_name = "paper_workflow_step_summary"
    for existing in root.handlers:
        if getattr(existing, "name", "") == handler_name:
            return str(target)

    handler = logging.FileHandler(target, encoding="utf-8")
    handler.name = handler_name
    handler.setLevel(level)
    handler.addFilter(_StepSummaryFilter())
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")
    )
    root.addHandler(handler)
    return str(target)


def setup_logging(level: int = logging.INFO) -> None:
    """
    功能：setup_logging 的核心流程封装，负责处理输入并输出结果。
    参数：level。
    返回：依据实现返回结果或产生副作用。
    流程：接收输入 → 核心逻辑处理 → 输出或触发副作用。
    说明：如遇异常或边界情况，调用侧需关注返回值或日志信息。
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
