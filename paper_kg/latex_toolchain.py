# 中文注释: 本文件提供 LaTeX 工具链的本地实现（lint + 编译）。
from __future__ import annotations

import logging
import re
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Protocol


logger = logging.getLogger(__name__)


@dataclass
class LintResult:
    """
    功能：LaTeX 静态检查结果结构。
    参数：ok/output/warnings。
    返回：LintResult。
    流程：作为 lint 的统一输出载体。
    说明：lint 失败不一定阻断流程，但需要记录日志。
    """

    ok: bool
    output: str
    warnings: List[str]


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
    返回：lint/compile 结果。
    说明：tests 中不应依赖真实 latexmk/chktex/lacheck。
    """

    def lint_tex(self, target: Path) -> LintResult:  # noqa: D401
        """执行 lint 检查（可对文件或目录）。"""

    def compile_project(self, project_dir: Path, engine: str, timeout_sec: int) -> CompileResult:
        """编译完整工程。"""

    def compile_minimal_section(
        self,
        project_dir: Path,
        section_relpath: str,
        engine: str,
        timeout_sec: int,
    ) -> CompileResult:
        """编译仅包含单章的最小工程。"""


class LocalLatexToolchain:
    """
    功能：本地 LaTeX 工具链实现（基于 latexmk/chktex/lacheck）。
    参数：require_tools/use_chktex/use_lacheck。
    返回：实例对象。
    流程：检查工具可用性 → 提供 lint/compile 方法。
    说明：require_tools 为 True 且工具缺失时应报错。
    """

    def __init__(self, require_tools: bool = True, use_chktex: bool = True, use_lacheck: bool = True) -> None:
        self.require_tools = require_tools
        self.use_chktex = use_chktex
        self.use_lacheck = use_lacheck

        self._latexmk = shutil.which("latexmk")
        self._chktex = shutil.which("chktex")
        self._lacheck = shutil.which("lacheck")

        if self.require_tools and self._latexmk is None:
            raise RuntimeError("未找到 latexmk，请安装 TeXLive/latexmk 或关闭 latex_require_tools")
        if self.require_tools and self.use_chktex and self._chktex is None:
            raise RuntimeError("未找到 chktex，请安装 chktex 或关闭 latex_use_chktex")
        if self.require_tools and self.use_lacheck and self._lacheck is None:
            raise RuntimeError("未找到 lacheck，请安装 lacheck 或关闭 latex_use_lacheck")

    def lint_tex(self, target: Path) -> LintResult:
        """
        功能：对指定 tex 文件或目录执行 lint（chktex + lacheck）。
        参数：target（文件或目录）。
        返回：LintResult。
        流程：顺序执行可用工具 → 合并输出。
        说明：lint 失败不一定阻断流程，但会记录 warnings。
        """
        outputs: List[str] = []
        warnings: List[str] = []
        ok = True

        if self.use_chktex and self._chktex:
            cmd = [self._chktex, "-q", "-n1", "-n2", "-n8", str(target)]
            result = self._run(cmd, timeout_sec=30)
            outputs.append(result.stdout + result.stderr)
            if not result.ok:
                ok = False
                warnings.append("chktex 返回非零")
        elif self.use_chktex:
            warnings.append("chktex 未找到，已跳过")

        if self.use_lacheck and self._lacheck:
            cmd = [self._lacheck, str(target)]
            result = self._run(cmd, timeout_sec=30)
            outputs.append(result.stdout + result.stderr)
            if not result.ok:
                ok = False
                warnings.append("lacheck 返回非零")
        elif self.use_lacheck:
            warnings.append("lacheck 未找到，已跳过")

        output = "\n".join([o.strip() for o in outputs if o.strip()])
        return LintResult(ok=ok, output=output, warnings=warnings)

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

    def compile_minimal_section(
        self,
        project_dir: Path,
        section_relpath: str,
        engine: str,
        timeout_sec: int,
    ) -> CompileResult:
        """
        功能：编译仅包含单章节的最小工程。
        参数：project_dir/section_relpath/engine/timeout_sec。
        返回：CompileResult。
        流程：创建临时 main_min.tex → latexmk 编译。
        说明：有助于定位具体章节错误。
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

        section_relpath = section_relpath.replace("\\", "/")
        section_relpath_clean = section_relpath
        while section_relpath_clean.startswith("../"):
            section_relpath_clean = section_relpath_clean[3:]
        section_path = project_dir / section_relpath_clean
        if not section_relpath.startswith("../"):
            section_relpath = f"../{section_relpath}"
        tmp_dir = Path(tempfile.mkdtemp(prefix="latex_min_", dir=str(project_dir)))
        main_path = tmp_dir / "main_min.tex"
        main_path.write_text(
            self._minimal_main_tex(section_relpath),
            encoding="utf-8",
        )
        cite_keys: List[str] = []
        if section_path.exists():
            cite_keys = _extract_cite_keys_from_tex(section_path.read_text(encoding="utf-8"))
        (tmp_dir / "refs.bib").write_text(_build_placeholder_bib_entries(cite_keys), encoding="utf-8")
        cmd = self._build_latexmk_cmd(engine, str(main_path.name), jobname="main_min")
        result = self._run(cmd, timeout_sec=timeout_sec, cwd=tmp_dir)
        return result

    def _minimal_main_tex(self, section_relpath: str) -> str:
        """
        功能：生成最小编译入口（仅包含单章节）。
        参数：section_relpath。
        返回：tex 字符串。
        说明：采用 ctex + article 以保证中文兼容。
        """
        return (
            "\\documentclass{article}\n"
            "\\usepackage{ctex}\n"
            "\\usepackage{amsmath}\n"
            "\\usepackage{graphicx}\n"
            "% --- Safe includegraphics: do not fail compilation if figure files are missing ---\n"
            "\\let\\origincludegraphics\\includegraphics\n"
            "\\newcommand{\\MissingFigureBox}[1]{%\n"
            "  \\fbox{\\parbox[c][0.18\\textheight][c]{0.9\\linewidth}{\\centering Missing figure: \\texttt{#1}}}%\n"
            "}\n"
            "\\renewcommand{\\includegraphics}[2][]{%\n"
            "  \\IfFileExists{#2}{\\origincludegraphics[#1]{#2}}{%\n"
            "    \\IfFileExists{#2.pdf}{\\origincludegraphics[#1]{#2.pdf}}{%\n"
            "      \\IfFileExists{#2.png}{\\origincludegraphics[#1]{#2.png}}{%\n"
            "        \\IfFileExists{#2.jpg}{\\origincludegraphics[#1]{#2.jpg}}{%\n"
            "          \\IfFileExists{#2.jpeg}{\\origincludegraphics[#1]{#2.jpeg}}{%\n"
            "            \\MissingFigureBox{\\detokenize{#2}}%\n"
            "          }%\n"
            "        }%\n"
            "      }%\n"
            "    }%\n"
            "  }%\n"
            "}\n"
            "% -------------------------------------------------------------------------------\n"
            "\\usepackage{booktabs}\n"
            "\\usepackage{natbib}\n"
            "\\begin{document}\n"
            f"\\input{{{section_relpath}}}\n"
            "\\bibliographystyle{plainnat}\n"
            "\\bibliography{refs}\n"
            "\\end{document}\n"
        )

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


def _extract_cite_keys_from_tex(text: str) -> List[str]:
    cite_pattern = re.compile(r"\\cite[a-zA-Z]*\{([^}]*)\}")
    keys: List[str] = []
    for match in cite_pattern.finditer(text or ""):
        raw_keys = match.group(1)
        for key in raw_keys.split(","):
            key = key.strip()
            if key:
                keys.append(key)
    return sorted(set(keys))


def _build_placeholder_bib_entries(keys: List[str]) -> str:
    entries = []
    for key in keys:
        entries.append(
            (
                f"@misc{{{key},\n"
                "  title = {TODO Placeholder},\n"
                "  year = {2026},\n"
                "  note = {Auto-generated placeholder}\n"
                "}\n"
            )
        )
    return "\n".join(entries)


_FATAL_TEX_PATTERNS = [
    re.compile(r"^! LaTeX Error:", re.MULTILINE),
    re.compile(r"^! Package .* Error:", re.MULTILINE),
    re.compile(r"^! Undefined control sequence", re.MULTILINE),
    re.compile(r"Emergency stop", re.MULTILINE),
    re.compile(r"Fatal error occurred", re.MULTILINE),
    re.compile(r"Runaway argument", re.MULTILINE),
]

_NONFATAL_REF_CITE_PATTERNS = [
    re.compile(r"Missing bbl file", re.IGNORECASE),
    re.compile(r"No file .*\\.bbl", re.IGNORECASE),
    re.compile(r"Undefined refs and citations", re.IGNORECASE),
    re.compile(r"^Reference `.*' .*undefined", re.IGNORECASE),
    re.compile(r"^Citation `.*' .*undefined", re.IGNORECASE),
    re.compile(r"There were undefined references", re.IGNORECASE),
    re.compile(r"There were undefined citations", re.IGNORECASE),
]


def _contains_fatal_tex_error(text: str) -> bool:
    for pattern in _FATAL_TEX_PATTERNS:
        if pattern.search(text or ""):
            return True
    return False


def _extract_nonfatal_ref_cite_warnings(text: str) -> List[str]:
    if not text:
        return []
    warnings: List[str] = []
    seen = set()
    for line in text.splitlines():
        for pattern in _NONFATAL_REF_CITE_PATTERNS:
            if pattern.search(line):
                clean = line.strip()
                if clean and clean not in seen:
                    warnings.append(clean)
                    seen.add(clean)
                break
    return warnings


def is_nonfatal_minimal_compile_issue(text: str) -> bool:
    if not text:
        return False
    if _contains_fatal_tex_error(text):
        return False
    return bool(_extract_nonfatal_ref_cite_warnings(text))
