# 中文注释: LaTeX 文本处理工具（换行归一化等）。
from __future__ import annotations

import re


def normalize_latex_newlines(tex: str) -> str:
    """
    功能：把字面量 \\n 转为真实换行，避免 LaTeX 误解析为 \\n 命令。
    参数：tex。
    返回：归一化后的文本。
    说明：避免误伤 \\newcommand、\\nabla 等合法命令。
    """
    if not tex:
        return tex

    # 先处理字面量 \\r\\n，避免残留 \\r。
    tex = re.sub(r"\\r\\n(?![A-Za-z])", "\n", tex)
    # 处理字面量 \\n，但不替换 \\newcommand/\\nabla 等命令。
    tex = re.sub(r"\\n(?![A-Za-z])", "\n", tex)
    return tex
