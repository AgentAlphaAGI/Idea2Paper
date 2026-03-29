import time
from typing import Dict, List, Optional

import requests

from idea2paper.config import (
    EMBEDDING_API_KEY,
    EMBEDDING_API_URL,
    EMBEDDING_MODEL,
    EMBEDDING_PROVIDER,
)
from idea2paper.infra.run_context import get_logger


def _effective_provider() -> str:
    return (EMBEDDING_PROVIDER or "openai_compatible").strip().lower()


def _is_gemini() -> bool:
    return _effective_provider() == "gemini"


# ---------------------------------------------------------------------------
# Gemini helpers
# ---------------------------------------------------------------------------

def _gemini_embed_url() -> str:
    """Return the embedContent URL for Gemini."""
    url = (EMBEDDING_API_URL or "").strip()
    if ":embedContent" in url:
        return url
    base = url.rstrip("/")
    if not base:
        base = "https://generativelanguage.googleapis.com/v1beta"
    return f"{base}/models/{EMBEDDING_MODEL}:embedContent"


def _gemini_batch_url() -> str:
    """Return the batchEmbedContents URL for Gemini."""
    return _gemini_embed_url().replace(":embedContent", ":batchEmbedContents")


def _gemini_headers() -> Dict[str, str]:
    return {
        "x-goog-api-key": EMBEDDING_API_KEY,
        "Content-Type": "application/json",
    }


def _gemini_single(text: str, logger, timeout: int) -> Optional[List[float]]:
    start_ts = time.time()
    url = _gemini_embed_url()
    payload = {
        "model": f"models/{EMBEDDING_MODEL}",
        "content": {"parts": [{"text": text}]},
    }
    try:
        resp = requests.post(url, headers=_gemini_headers(), json=payload, timeout=timeout)
        resp.raise_for_status()
        emb = resp.json()["embedding"]["values"]
        if logger:
            logger.log_embedding_call(
                request={"provider": "gemini", "url": url, "model": EMBEDDING_MODEL,
                         "input_preview": text, "timeout": timeout, "simulated": False},
                response={"ok": True, "latency_ms": int((time.time() - start_ts) * 1000)},
            )
        return emb
    except Exception as e:
        if logger:
            logger.log_embedding_call(
                request={"provider": "gemini", "url": url, "model": EMBEDDING_MODEL,
                         "input_preview": text, "timeout": timeout, "simulated": False},
                response={"ok": False, "latency_ms": int((time.time() - start_ts) * 1000), "error": str(e)},
            )
        return None


def _gemini_batch(texts: List[str], logger, timeout: int) -> Optional[List[List[float]]]:
    start_ts = time.time()
    url = _gemini_batch_url()
    model_ref = f"models/{EMBEDDING_MODEL}"
    payload = {
        "requests": [
            {"model": model_ref, "content": {"parts": [{"text": t}]}}
            for t in texts
        ]
    }
    try:
        resp = requests.post(url, headers=_gemini_headers(), json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        embs = [item["values"] for item in data.get("embeddings", [])]
        if len(embs) != len(texts):
            raise ValueError(f"embedding batch size mismatch: got {len(embs)} expected {len(texts)}")
        if logger:
            logger.log_embedding_call(
                request={"provider": "gemini", "url": url, "model": EMBEDDING_MODEL,
                         "input_preview": _preview_texts(texts), "timeout": timeout,
                         "simulated": False, "batch_size": len(texts)},
                response={"ok": True, "latency_ms": int((time.time() - start_ts) * 1000)},
            )
        return embs
    except Exception as e:
        if logger:
            logger.log_embedding_call(
                request={"provider": "gemini", "url": url, "model": EMBEDDING_MODEL,
                         "input_preview": _preview_texts(texts), "timeout": timeout,
                         "simulated": False, "batch_size": len(texts)},
                response={"ok": False, "latency_ms": int((time.time() - start_ts) * 1000), "error": str(e)},
            )
        return None


# ---------------------------------------------------------------------------
# OpenAI-compatible helpers
# ---------------------------------------------------------------------------

def _openai_single(text: str, logger, timeout: int) -> Optional[List[float]]:
    start_ts = time.time()
    provider_tag = "openai_compatible"
    headers = {"Authorization": f"Bearer {EMBEDDING_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": EMBEDDING_MODEL, "input": text}
    try:
        resp = requests.post(EMBEDDING_API_URL, headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
        emb = resp.json()["data"][0]["embedding"]
        if logger:
            logger.log_embedding_call(
                request={"provider": provider_tag, "url": EMBEDDING_API_URL, "model": EMBEDDING_MODEL,
                         "input_preview": text, "timeout": timeout, "simulated": False},
                response={"ok": True, "latency_ms": int((time.time() - start_ts) * 1000)},
            )
        return emb
    except Exception as e:
        if logger:
            logger.log_embedding_call(
                request={"provider": provider_tag, "url": EMBEDDING_API_URL, "model": EMBEDDING_MODEL,
                         "input_preview": text, "timeout": timeout, "simulated": False},
                response={"ok": False, "latency_ms": int((time.time() - start_ts) * 1000), "error": str(e)},
            )
        return None


def _openai_batch(texts: List[str], logger, timeout: int) -> Optional[List[List[float]]]:
    start_ts = time.time()
    provider_tag = "openai_compatible"
    headers = {"Authorization": f"Bearer {EMBEDDING_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": EMBEDDING_MODEL, "input": texts}
    try:
        resp = requests.post(EMBEDDING_API_URL, headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        embs = [item["embedding"] for item in data.get("data", [])]
        if len(embs) != len(texts):
            raise ValueError(f"embedding batch size mismatch: got {len(embs)} expected {len(texts)}")
        if logger:
            logger.log_embedding_call(
                request={"provider": provider_tag, "url": EMBEDDING_API_URL, "model": EMBEDDING_MODEL,
                         "input_preview": _preview_texts(texts), "timeout": timeout,
                         "simulated": False, "batch_size": len(texts)},
                response={"ok": True, "latency_ms": int((time.time() - start_ts) * 1000)},
            )
        return embs
    except Exception as e:
        if logger:
            logger.log_embedding_call(
                request={"provider": provider_tag, "url": EMBEDDING_API_URL, "model": EMBEDDING_MODEL,
                         "input_preview": _preview_texts(texts), "timeout": timeout,
                         "simulated": False, "batch_size": len(texts)},
                response={"ok": False, "latency_ms": int((time.time() - start_ts) * 1000), "error": str(e)},
            )
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_embedding(text: str, logger=None, timeout: int = 120) -> Optional[List[float]]:
    """Get embedding for text. Dispatches to Gemini or OpenAI-compatible based on EMBEDDING_PROVIDER.

    Returns None on failure (no exception thrown).
    """
    if logger is None:
        logger = get_logger()

    if not EMBEDDING_API_KEY:
        start_ts = time.time()
        prov = _effective_provider()
        url = _gemini_embed_url() if _is_gemini() else EMBEDDING_API_URL
        if logger:
            logger.log_embedding_call(
                request={"provider": prov, "url": url, "model": EMBEDDING_MODEL,
                         "input_preview": text, "timeout": timeout, "simulated": True},
                response={"ok": False, "latency_ms": int((time.time() - start_ts) * 1000),
                           "error": "EMBEDDING_API_KEY not configured"},
            )
        return None

    if _is_gemini():
        return _gemini_single(text, logger, timeout)
    return _openai_single(text, logger, timeout)


def _preview_texts(texts: List[str], max_chars: int = 200) -> List[str]:
    previews = []
    for t in texts:
        if t is None:
            previews.append("")
            continue
        s = str(t)
        if len(s) > max_chars:
            previews.append(s[:max_chars] + "...(truncated)")
        else:
            previews.append(s)
    return previews


def get_embeddings_batch(texts: List[str], logger=None, timeout: int = 120) -> Optional[List[List[float]]]:
    """Get embeddings for a batch of texts. Returns None on failure."""
    if logger is None:
        logger = get_logger()

    if not EMBEDDING_API_KEY:
        start_ts = time.time()
        prov = _effective_provider()
        url = _gemini_batch_url() if _is_gemini() else EMBEDDING_API_URL
        if logger:
            logger.log_embedding_call(
                request={"provider": prov, "url": url, "model": EMBEDDING_MODEL,
                         "input_preview": _preview_texts(texts), "timeout": timeout,
                         "simulated": True, "batch_size": len(texts)},
                response={"ok": False, "latency_ms": int((time.time() - start_ts) * 1000),
                           "error": "EMBEDDING_API_KEY not configured"},
            )
        return None

    if _is_gemini():
        return _gemini_batch(texts, logger, timeout)
    return _openai_batch(texts, logger, timeout)
