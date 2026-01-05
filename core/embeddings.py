# 中文注释: 本文件提供 Embedding 模型的创建与封装。
from __future__ import annotations

import logging
from typing import List, Optional

from langchain_openai import OpenAIEmbeddings

from core.llm import _resolve_env, load_env
from paper_kg.interfaces import IEmbeddingModel

logger = logging.getLogger(__name__)


class OpenAIEmbeddingModel(IEmbeddingModel):
    """
    功能：OpenAI-compatible embedding 模型封装。
    参数：model/api_key/base_url/timeout。
    返回：提供 embed(text) 方法的实例。
    流程：内部调用 OpenAIEmbeddings.embed_query。
    说明：用于替换 MockEmbeddingModel。
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> None:
        """
        功能：初始化 embedding 模型。
        参数：model/api_key/base_url/timeout。
        返回：无。
        流程：组装 OpenAIEmbeddings。
        说明：base_url 可对接 OpenAI-compatible 服务。
        """
        # 中文注释：关闭 tiktoken，将输入保持为原始字符串，避免兼容服务拒收 token 列表。
        kwargs = {
            "model": model,
            "api_key": api_key,
            "tiktoken_enabled": False,
            "check_embedding_ctx_length": False,
        }
        if base_url:
            kwargs["base_url"] = base_url
        if timeout is not None:
            kwargs["timeout"] = timeout
        self._model = OpenAIEmbeddings(**kwargs)

    def embed(self, text: str) -> List[float]:
        """
        功能：生成文本向量。
        参数：text。
        返回：向量列表。
        流程：调用 OpenAIEmbeddings.embed_query。
        说明：异常由调用方处理。
        """
        # 中文注释：记录每次 embedding 调用的开始与结果，便于排查耗时与失败原因。
        logger.info("EMBEDDING_CALL_START: chars=%s", len(text))
        try:
            vector = self._model.embed_query(text)
        except Exception as exc:  # noqa: BLE001
            logger.warning("EMBEDDING_CALL_FAIL: err=%s", exc)
            raise
        logger.info("EMBEDDING_CALL_OK: dims=%s", len(vector))
        return vector


def create_embedding_model(agent_name: str, required: bool = False) -> Optional[IEmbeddingModel]:
    """
    功能：基于环境变量创建 Embedding 模型。
    参数：agent_name、required。
    返回：IEmbeddingModel 或 None。
    流程：读取环境变量 → 初始化 OpenAIEmbeddingModel。
    说明：required=True 且缺少 key 将抛出异常。
    """
    load_env()
    prefix = agent_name.upper()
    api_key = _resolve_env(prefix, "OPENAI_API_KEY")
    if not api_key:
        if required:
            raise ValueError(
                f"OPENAI_API_KEY is required for embedding agent={agent_name} in .env or environment"
            )
        return None

    model = _resolve_env(prefix, "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    base_url = _resolve_env(prefix, "OPENAI_BASE_URL")
    timeout_raw = _resolve_env(prefix, "OPENAI_TIMEOUT")
    timeout = float(timeout_raw) if timeout_raw else None

    return OpenAIEmbeddingModel(
        model=model,
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
    )
