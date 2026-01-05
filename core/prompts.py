# 中文注释: 本文件包含可运行代码与流程说明。
from __future__ import annotations

import os
from pathlib import Path


def _prompt_dir() -> Path:
    """
    功能：获取提示词目录路径。
    参数：无。
    返回：提示词目录 Path。
    流程：读取 PROMPT_DIR 环境变量 → 回退到项目 prompts 目录。
    说明：PROMPT_DIR 可用于自定义提示词文件位置。
    """
    env_dir = os.getenv("PROMPT_DIR")
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    return Path(__file__).resolve().parents[1] / "prompts"


def load_prompt(filename: str) -> str:
    """
    功能：读取指定提示词文件内容。
    参数：filename。
    返回：提示词文本内容。
    流程：拼接目录 → 读取文件 → 返回文本。
    说明：文件不存在时抛出 FileNotFoundError。
    """
    path = _prompt_dir() / filename
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8")
