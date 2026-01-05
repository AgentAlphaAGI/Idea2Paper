# 中文注释: 本文件包含可运行代码与流程说明。
from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


def load_env() -> None:
    """
    功能：加载 .env 文件中的环境变量，便于后续读取 API 配置。
    参数：无。
    返回：无，使用副作用写入进程环境变量。
    流程：读取 .env → 合并到 os.environ。
    说明：若环境变量已存在，默认不会覆盖已有值。
    """
    load_dotenv()


def _get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    """
    功能：统一获取环境变量，提供默认值回退。
    参数：name、default。
    返回：环境变量值或默认值。
    流程：读取 os.environ → 判空 → 返回。
    说明：返回 None 表示未设置且无默认值。
    """
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value


def _resolve_env(prefix: str, name: str, default: Optional[str] = None) -> Optional[str]:
    """
    功能：按代理前缀优先解析环境变量。
    参数：prefix、name、default。
    返回：环境变量值或默认值。
    流程：优先读取 PREFIX_NAME → 回退到 NAME → 返回。
    说明：当 prefix 为空时直接读取 NAME。
    """
    if prefix:
        value = _get_env(f"{prefix}_{name}")
        if value is not None:
            return value
    return _get_env(name, default)


def create_chat_model(agent_name: str, required: bool = False) -> Optional[ChatOpenAI]:
    """
    功能：基于环境变量构建可用的 ChatOpenAI 模型实例。
    参数：agent_name、required。
    返回：ChatOpenAI 模型对象或 None。
    流程：读取环境变量 → 组装参数 → 初始化模型。
    说明：required=True 且未配置 API Key 会抛出异常。
    """
    prefix = agent_name.upper()
    api_key = _resolve_env(prefix, "OPENAI_API_KEY")
    if not api_key:
        if required:
            raise ValueError(
                f"OPENAI_API_KEY is required for agent={agent_name} in .env or environment"
            )
        return None

    model = _resolve_env(prefix, "OPENAI_MODEL", "gpt-4o-mini")
    base_url = _resolve_env(prefix, "OPENAI_BASE_URL")
    temperature = float(_resolve_env(prefix, "OPENAI_TEMPERATURE", "0.7") or "0.7")
    timeout = _resolve_env(prefix, "OPENAI_TIMEOUT")

    kwargs = {}
    if base_url:
        kwargs["base_url"] = base_url
    if timeout:
        kwargs["timeout"] = float(timeout)

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        temperature=temperature,
        **kwargs,
    )
