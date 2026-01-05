# 中文注释: 本文件提供论文链路专用的提示词加载工具，避免与其他模块的提示词冲突。
from __future__ import annotations

import os
from pathlib import Path


def _paper_prompt_dir() -> Path:
    """
    功能：获取论文链路的提示词目录。
    参数：无。
    返回：提示词目录 Path。
    流程：读取 PAPER_PROMPT_DIR 环境变量 → 回退到项目 prompts_paper 目录。
    说明：
    - 之所以不用 core/prompts.py 的 PROMPT_DIR，是为了保证不同模块的提示词互不覆盖。
    - PAPER_PROMPT_DIR 支持指向自定义目录（例如用于 E2E 强制回退场景的提示词集合）。
    """
    env_dir = os.getenv("PAPER_PROMPT_DIR")
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    return Path(__file__).resolve().parents[1] / "prompts_paper"


def load_paper_prompt(filename: str) -> str:
    """
    功能：读取论文链路提示词文件内容。
    参数：filename。
    返回：提示词文本。
    流程：拼接目录 → 读取文件 → 返回文本。
    说明：文件不存在时抛出 FileNotFoundError，便于上层快速定位配置问题。
    """
    path = _paper_prompt_dir() / filename
    if not path.exists():
        raise FileNotFoundError(f"Paper prompt file not found: {path}")
    return path.read_text(encoding="utf-8")
