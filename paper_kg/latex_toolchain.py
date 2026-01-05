# 中文注释: 本文件提供 LaTeX 工具链的本地实现（编译）。
from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Protocol


logger = logging.getLogger(__name__)


@dataclass
class CompileResult:
    """
    功能：LaTeX 编译结果结构。
    参数：ok/stdout/stderr/log_excerpt/errors/cmd。
    返回：CompileResult。
    流程：统一保存编译输出与错误摘要。
    说明：errors 用于定位问题章节并触发修复。
    """

    ok: bool
    stdout: str
    stderr: str
    log_excerpt: str
    errors: List[Dict[str, object]]
    cmd: List[str]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class ILatexToolchain(Protocol):
    """
    功能：LaTeX 工具链接口（便于测试注入 FakeToolchain）。
    参数：无。
    返回：compile 结果。
    说明：tests 中不应依赖真实 latexmk。
    """

    def compile_project(self, project_dir: Path, engine: str, timeout_sec: int) -> CompileResult:
        """编译完整工程。"""


class LocalLatexToolchain:
    """
    功能：本地 LaTeX 工具链实现（基于 latexmk）。
    参数：require_tools。
    返回：实例对象。
    流程：检查工具可用性 → 提供 compile 方法。
    说明：require_tools 为 True 且工具缺失时应报错。
    """

    def __init__(self, require_tools: bool = True) -> None:
        self.require_tools = require_tools

        self._latexmk = shutil.which("latexmk")

        if self.require_tools and self._latexmk is None:
            raise RuntimeError("未找到 latexmk，请安装 TeXLive/latexmk 或关闭 latex_require_tools")

    def compile_project(self, project_dir: Path, engine: str, timeout_sec: int) -> CompileResult:
        """
        功能：编译完整工程。
        参数：project_dir/engine/timeout_sec。
        返回：CompileResult。
        流程：调用 latexmk。
        说明：使用 -interaction=nonstopmode 与 -file-line-error 便于错误定位。
        """
        if self._latexmk is None:
            return CompileResult(
                ok=False,
                stdout="",
                stderr="latexmk not found",
                log_excerpt="latexmk not found",
                errors=[{"message": "latexmk not found"}],
                cmd=[],
            )

        cmd = self._build_latexmk_cmd(engine, "main.tex")
        return self._run(cmd, timeout_sec=timeout_sec, cwd=project_dir)

    def _build_latexmk_cmd(self, engine: str, tex_file: str, jobname: Optional[str] = None) -> List[str]:
        """
        功能：构建 latexmk 命令行参数。
        参数：engine/tex_file/jobname。
        返回：命令列表。
        说明：根据 engine 选择 xelatex 或 pdflatex。
        """
        cmd = [self._latexmk or "latexmk"]
        if engine.lower() == "xelatex":
            cmd.append("-xelatex")
        else:
            cmd.append("-pdf")
        cmd += ["-interaction=nonstopmode", "-file-line-error"]
        if jobname:
            cmd += [f"-jobname={jobname}"]
        cmd.append(tex_file)
        return cmd

    def _run(
        self,
        cmd: List[str],
        timeout_sec: int,
        cwd: Optional[Path] = None,
    ) -> CompileResult:
        """
        功能：执行外部命令并收集输出。
        参数：cmd/timeout_sec/cwd。
        返回：CompileResult。
        说明：即使失败也返回结构化结果。
        """
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(cwd) if cwd else None,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
            stdout = proc.stdout or ""
            stderr = proc.stderr or ""
            ok = proc.returncode == 0
            log_excerpt = (stderr or stdout)[-2000:]
            errors = _parse_file_line_errors(stderr or stdout)
            return CompileResult(ok=ok, stdout=stdout, stderr=stderr, log_excerpt=log_excerpt, errors=errors, cmd=cmd)
        except subprocess.TimeoutExpired:
            return CompileResult(
                ok=False,
                stdout="",
                stderr="timeout",
                log_excerpt="timeout",
                errors=[{"message": "timeout"}],
                cmd=cmd,
            )


def _parse_file_line_errors(text: str) -> List[Dict[str, object]]:
    """
    功能：解析 latexmk 的 -file-line-error 输出。
    参数：text。
    返回：错误列表。
    流程：匹配 file:line: message 格式 → 返回结构化列表。
    说明：仅做粗解析，足以定位章节文件。
    """
    import re

    errors: List[Dict[str, object]] = []
    pattern = re.compile(r"^([^:\n]+):(\d+):\s*(.*)$", re.MULTILINE)
    for match in pattern.finditer(text or ""):
        errors.append({"file": match.group(1), "line": int(match.group(2)), "message": match.group(3)})
    return errors
