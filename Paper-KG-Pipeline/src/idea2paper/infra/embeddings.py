import time
from typing import Optional, List

import requests

from idea2paper.config import LLM_API_KEY
from idea2paper.infra.run_context import get_logger


EMBEDDING_URL = "https://api.siliconflow.cn/v1/embeddings"
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"


def get_embedding(text: str, logger=None, timeout: int = 120) -> Optional[List[float]]:
    """Get embedding for text using SiliconFlow embeddings API.

    Returns None on failure (no exception thrown).
    """
    if logger is None:
        logger = get_logger()
    start_ts = time.time()

    if not LLM_API_KEY:
        if logger:
            logger.log_embedding_call(
                request={
                    "provider": "siliconflow",
                    "url": EMBEDDING_URL,
                    "model": EMBEDDING_MODEL,
                    "input_preview": text,
                    "timeout": timeout,
                    "simulated": True
                },
                response={
                    "ok": False,
                    "latency_ms": int((time.time() - start_ts) * 1000),
                    "error": "SILICONFLOW_API_KEY not configured"
                }
            )
        return None

    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": EMBEDDING_MODEL,
        "input": text
    }

    try:
        resp = requests.post(EMBEDDING_URL, headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        emb = data["data"][0]["embedding"]
        if logger:
            logger.log_embedding_call(
                request={
                    "provider": "siliconflow",
                    "url": EMBEDDING_URL,
                    "model": EMBEDDING_MODEL,
                    "input_preview": text,
                    "timeout": timeout,
                    "simulated": False
                },
                response={
                    "ok": True,
                    "latency_ms": int((time.time() - start_ts) * 1000)
                }
            )
        return emb
    except Exception as e:
        if logger:
            logger.log_embedding_call(
                request={
                    "provider": "siliconflow",
                    "url": EMBEDDING_URL,
                    "model": EMBEDDING_MODEL,
                    "input_preview": text,
                    "timeout": timeout,
                    "simulated": False
                },
                response={
                    "ok": False,
                    "latency_ms": int((time.time() - start_ts) * 1000),
                    "error": str(e)
                }
            )
        return None
