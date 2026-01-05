# 中文注释: Pandoc 渲染器与 Markdown 预/后处理工具。
from __future__ import annotations

import logging
import os
import re
import subprocess
from typing import Dict, Protocol


logger = logging.getLogger(__name__)


_MARKER_RE = re.compile(r"<!--\s*CITE_CANDIDATE(?::([A-Za-z0-9:_-]+))?\s*-->")
_INLINE_CODE_RE = re.compile(r"`([^`]+)`")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


class IPandocRunner(Protocol):
    def to_latex(self, markdown: str) -> str: ...


class SubprocessPandocRunner:
    def __init__(self, pandoc_bin: str = "pandoc", require: bool = True) -> None:
        env_bin = os.getenv("PANDOC_BIN")
        self.pandoc_bin = env_bin or pandoc_bin
        self.require = require
        self.available = True

    def to_latex(self, markdown: str) -> str:
        if not markdown:
            logger.info("PANDOC_RENDER_SKIP: 输入为空，直接返回空文本")
            return ""

        cmd = [
            self.pandoc_bin,
            "--from",
            "markdown+raw_tex+tex_math_dollars",
            "--to",
            "latex",
            "--wrap=none",
            "--no-highlight",
        ]
        logger.info(
            "PANDOC_RENDER_START: bin=%s input_chars=%s",
            self.pandoc_bin,
            len(markdown),
        )
        try:
            result = subprocess.run(
                cmd,
                input=markdown,
                text=True,
                capture_output=True,
                check=True,
            )
        except FileNotFoundError as exc:
            self.available = False
            logger.warning("PANDOC_RENDER_FAIL: 未找到 pandoc 可执行文件")
            if self.require:
                raise RuntimeError("请安装 pandoc 或配置 PANDOC_BIN") from exc
            return ""
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            stdout = (exc.stdout or "").strip()
            message = f"pandoc failed with exit code {exc.returncode}"
            if stderr or stdout:
                message = f"{message}: {stderr or stdout}"
            logger.error("PANDOC_RENDER_FAIL: %s", message)
            raise RuntimeError(message) from exc

        output = result.stdout or ""
        logger.info("PANDOC_RENDER_OK: output_chars=%s", len(output))
        return output


def strip_top_heading(md: str) -> str:
    lines = md.splitlines()
    if not lines:
        return md

    first = lines[0].lstrip()
    if re.match(r"#(\s+|$)", first):
        lines = lines[1:]
        while lines and not lines[0].strip():
            lines.pop(0)
        return "\n".join(lines)
    return md


def replace_cite_candidates(md: str, citation_map: Dict[str, str]) -> str:
    def _replace(match: re.Match) -> str:
        cand_id = (match.group(1) or "").strip()
        if not cand_id:
            return ""
        bibkey = citation_map.get(cand_id, "")
        return f"\\cite{{{bibkey}}}" if bibkey else ""

    return _MARKER_RE.sub(_replace, md)


def sanitize_for_pandoc(md: str) -> str:
    if not md:
        return md

    lines = md.splitlines()
    out_lines = []
    in_code = False
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("```"):
            in_code = not in_code
            out_lines.append(line)
            continue
        if in_code:
            out_lines.append(line)
            continue

        def _inline_replace(match: re.Match) -> str:
            content = match.group(1)
            if _CJK_RE.search(content):
                return content
            return match.group(0)

        out_lines.append(_INLINE_CODE_RE.sub(_inline_replace, line))
    return "\n".join(out_lines)


def unwrap_outer_markdown_fence(md: str) -> str:
    if not md:
        return md
    lines = md.splitlines()
    first_idx = None
    last_idx = None
    for i, line in enumerate(lines):
        if line.strip():
            first_idx = i
            break
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip():
            last_idx = i
            break
    if first_idx is None or last_idx is None or first_idx >= last_idx:
        return md
    first_line = lines[first_idx].lstrip()
    last_line = lines[last_idx].strip()
    if not first_line.startswith("```") or last_line != "```":
        return md
    inner = lines[first_idx + 1 : last_idx]
    return "\n".join(inner)


def sanitize_pandoc_latex_fragment(tex: str) -> str:
    if not tex:
        return tex

    out_lines = []
    for line in tex.splitlines():
        stripped = line.strip()
        if stripped == "\\tightlist":
            continue
        if stripped.startswith("\\documentclass"):
            continue
        if stripped in {"\\begin{document}", "\\end{document}"}:
            continue
        out_lines.append(line)
    return "\n".join(out_lines)
