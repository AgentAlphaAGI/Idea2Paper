import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import requests

from idea2paper.config import (
    EMBEDDING_API_URL,
    EMBEDDING_MODEL,
    EMBEDDING_PROVIDER,
    LLM_API_URL,
    LLM_BASE_URL,
    LLM_MODEL,
    LLM_PROVIDER,
    LLM_ANTHROPIC_VERSION,
    LLM_EXTRA_BODY,
    LLM_EXTRA_HEADERS,
    NOVELTY_INDEX_DIR,
    PipelineConfig,
)
from idea2paper.infra.llm_providers import anthropic, gemini, openai_compatible, openai_responses
from idea2paper.infra.llm_providers.common import parse_extra, redact_mapping
from idea2paper.infra.run_context import get_logger


@dataclass
class PreflightResult:
    ok: bool
    error: str = ""
    llm_endpoint: str = ""
    embedding_endpoint: str = ""
    embedding_dim: Optional[int] = None


def _bool_env(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip() in ("1", "true", "True", "yes", "YES")


def _int_env(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None or not str(raw).strip():
        return default
    try:
        return int(str(raw).strip())
    except Exception:
        return default


def _sleep_backoff(attempt: int, base: float = 1.0, cap: float = 8.0) -> None:
    time.sleep(min(cap, base * (2 ** attempt)))


def _parse_extra_or_error(name: str, value) -> Tuple[Dict[str, Any], str]:
    data, err = parse_extra(value)
    if err:
        return {}, f"{name} parse failed: {err}"
    return data or {}, ""


def _llm_ping_once(timeout: int) -> Tuple[bool, str]:
    """
    Real LLM ping (fail-fast). Do NOT fallback to simulated output.
    """
    api_key = os.getenv("LLM_API_KEY", "")
    if not api_key:
        return False, "LLM_API_KEY not configured"

    provider = (LLM_PROVIDER or "openai_compatible_chat").strip().lower()
    extra_headers, err_h = _parse_extra_or_error("LLM_EXTRA_HEADERS_JSON", LLM_EXTRA_HEADERS)
    if err_h:
        return False, err_h
    extra_body, err_b = _parse_extra_or_error("LLM_EXTRA_BODY_JSON", LLM_EXTRA_BODY)
    if err_b:
        return False, err_b

    prompt = "ping"
    try:
        if provider in ("openai_compatible_chat", "openai_compatible"):
            r = openai_compatible.call_openai_compatible_chat(
                prompt,
                model=LLM_MODEL,
                api_key=api_key,
                base_url=LLM_BASE_URL,
                api_url=LLM_API_URL,
                temperature=0.0,
                max_tokens=8,
                timeout=timeout,
                extra_headers=extra_headers,
                extra_body=extra_body,
            )
        elif provider in ("openai_responses", "responses"):
            r = openai_responses.call_openai_responses(
                prompt,
                model=LLM_MODEL,
                api_key=api_key,
                base_url=LLM_BASE_URL,
                api_url=LLM_API_URL,
                temperature=0.0,
                max_tokens=8,
                timeout=timeout,
                extra_headers=extra_headers,
                extra_body=extra_body,
            )
        elif provider == "anthropic":
            r = anthropic.call_anthropic(
                prompt,
                model=LLM_MODEL,
                api_key=api_key,
                base_url=LLM_BASE_URL,
                api_url=LLM_API_URL,
                anthropic_version=LLM_ANTHROPIC_VERSION,
                temperature=0.0,
                max_tokens=8,
                timeout=timeout,
                extra_headers=extra_headers,
                extra_body=extra_body,
            )
        elif provider == "gemini":
            r = gemini.call_gemini(
                prompt,
                model=LLM_MODEL,
                api_key=api_key,
                base_url=LLM_BASE_URL,
                api_url=LLM_API_URL,
                temperature=0.0,
                max_tokens=8,
                timeout=timeout,
                extra_headers=extra_headers,
                extra_body=extra_body,
            )
        else:
            return False, f"unknown LLM_PROVIDER: {LLM_PROVIDER}"

        if isinstance(r, dict) and r.get("ok"):
            return True, ""
        if isinstance(r, dict):
            return False, r.get("error", "") or "llm call failed"
        return False, "llm call failed"
    except Exception as e:
        return False, str(e)


def _is_gemini_embedding() -> bool:
    return (EMBEDDING_PROVIDER or "").strip().lower() == "gemini"


def _embedding_ping_once(timeout: int) -> Tuple[bool, Optional[int], str]:
    """
    Real embedding ping (fail-fast) and infer embedding_dim.
    Supports both OpenAI-compatible and Gemini native APIs.
    """
    api_key = os.getenv("EMBEDDING_API_KEY", "")
    if not api_key:
        return False, None, "EMBEDDING_API_KEY not configured"

    if not EMBEDDING_API_URL:
        return False, None, "EMBEDDING_API_URL not configured"

    if _is_gemini_embedding():
        return _embedding_ping_gemini(api_key, timeout)
    return _embedding_ping_openai(api_key, timeout)


def _embedding_ping_gemini(api_key: str, timeout: int) -> Tuple[bool, Optional[int], str]:
    url = (EMBEDDING_API_URL or "").strip()
    if ":embedContent" not in url:
        base = url.rstrip("/") or "https://generativelanguage.googleapis.com/v1beta"
        url = f"{base}/models/{EMBEDDING_MODEL}:embedContent"

    headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
    payload = {
        "model": f"models/{EMBEDDING_MODEL}",
        "content": {"parts": [{"text": "ping"}]},
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        emb = data.get("embedding", {}).get("values")
        if not isinstance(emb, list):
            return False, None, "invalid Gemini embedding response: 'embedding.values' is not a list"
        dim = len(emb)
        if dim <= 0:
            return False, None, "embedding_dim is 0"
        return True, dim, ""
    except Exception as e:
        return False, None, str(e)


def _embedding_ping_openai(api_key: str, timeout: int) -> Tuple[bool, Optional[int], str]:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": EMBEDDING_MODEL, "input": "ping"}
    try:
        resp = requests.post(EMBEDDING_API_URL, headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        emb = data["data"][0]["embedding"]
        if not isinstance(emb, list):
            return False, None, "invalid embedding response: embedding is not a list"
        dim = len(emb)
        if dim <= 0:
            return False, None, "embedding_dim is 0"
        return True, dim, ""
    except Exception as e:
        return False, None, str(e)


def _read_npy_dim(path: Path) -> Optional[int]:
    try:
        arr = np.load(path, mmap_mode="r")
        if getattr(arr, "ndim", None) != 2:
            return None
        return int(arr.shape[1])
    except Exception:
        return None


def _check_index_dims(online_dim: int) -> None:
    """
    Compare online embedding dim with existing local index .npy dims.
    If index does not exist yet (first run), skip that check.
    """
    # novelty index
    novelty_emb = NOVELTY_INDEX_DIR / "paper_emb.npy"
    if novelty_emb.exists():
        d = _read_npy_dim(novelty_emb)
        if d is None:
            raise RuntimeError(f"cannot read novelty index dim: {novelty_emb}")
        if d != online_dim:
            raise RuntimeError(f"novelty index dim mismatch: index_dim={d} != online_dim={online_dim}")

    # recall offline index
    if PipelineConfig.RECALL_USE_OFFLINE_INDEX:
        idx = PipelineConfig.RECALL_INDEX_DIR
        for p in (idx / "idea_emb.npy", idx / "paper_emb.npy"):
            if p.exists():
                d = _read_npy_dim(p)
                if d is None:
                    raise RuntimeError(f"cannot read recall index dim: {p}")
                if d != online_dim:
                    raise RuntimeError(f"recall index dim mismatch: file={p.name} index_dim={d} != online_dim={online_dim}")


def run_startup_preflight() -> PreflightResult:
    """
    Startup preflight (fail-fast by default):
    - LLM connectivity (requires env LLM_API_KEY)
    - Embedding connectivity (requires env EMBEDDING_API_KEY)
    - (Optional) Embedding dim consistency with existing local indexes
    """
    if not _bool_env("I2P_PREFLIGHT_ENABLE", True):
        print("[preflight] disabled: I2P_PREFLIGHT_ENABLE=0")
        return PreflightResult(ok=True)

    llm_endpoint = (LLM_API_URL or LLM_BASE_URL or "").strip()
    emb_endpoint = (EMBEDDING_API_URL or "").strip()
    logger = get_logger()

    llm_retries = _int_env("I2P_PREFLIGHT_LLM_RETRIES", 3)
    llm_timeout = _int_env("I2P_PREFLIGHT_LLM_TIMEOUT", 20)
    emb_retries = _int_env("I2P_PREFLIGHT_EMB_RETRIES", 3)
    emb_timeout = _int_env("I2P_PREFLIGHT_EMB_TIMEOUT", 20)
    check_dim = _bool_env("I2P_PREFLIGHT_CHECK_DIM", True)

    # 1) LLM ping
    print(f"[preflight] start: llm_provider={LLM_PROVIDER} llm_model={LLM_MODEL} llm_endpoint={llm_endpoint or '(empty)'}")
    print(f"[preflight] start: embedding_model={EMBEDDING_MODEL} embedding_endpoint={emb_endpoint or '(empty)'} check_dim={int(bool(check_dim))}")
    print(f"[preflight][llm] checking connectivity (retries={llm_retries}, timeout={llm_timeout}s)...")
    last_err = ""
    for attempt in range(max(1, llm_retries)):
        ok, err = _llm_ping_once(timeout=llm_timeout)
        if ok:
            last_err = ""
            break
        last_err = err
        print(f"[preflight][llm] attempt {attempt + 1}/{max(1, llm_retries)} failed: {err}")
        _sleep_backoff(attempt)
    if last_err:
        msg = f"LLM preflight failed after {llm_retries} attempts: {last_err}"
        print(f"[preflight][llm] FAILED: {msg}")
        if logger:
            # avoid logging secrets
            extra_headers, _ = parse_extra(LLM_EXTRA_HEADERS)
            extra_body, _ = parse_extra(LLM_EXTRA_BODY)
            logger.log_event("startup_preflight_failed", {
                "kind": "llm",
                "error": msg,
                "endpoint": llm_endpoint,
                "provider": LLM_PROVIDER,
                "model": LLM_MODEL,
                "extra_headers": redact_mapping(extra_headers or {}),
                "extra_body": redact_mapping(extra_body or {}),
            })
        return PreflightResult(ok=False, error=msg, llm_endpoint=llm_endpoint, embedding_endpoint=emb_endpoint)

    # 2) Embedding ping + dim
    print("[preflight][llm] OK")
    print(f"[preflight][embedding] checking connectivity (retries={emb_retries}, timeout={emb_timeout}s)...")
    online_dim: Optional[int] = None
    last_err = ""
    for attempt in range(max(1, emb_retries)):
        ok, dim, err = _embedding_ping_once(timeout=emb_timeout)
        if ok:
            online_dim = dim
            last_err = ""
            break
        last_err = err
        print(f"[preflight][embedding] attempt {attempt + 1}/{max(1, emb_retries)} failed: {err}")
        _sleep_backoff(attempt)
    if last_err:
        msg = f"Embedding preflight failed after {emb_retries} attempts: {last_err}"
        print(f"[preflight][embedding] FAILED: {msg}")
        if logger:
            logger.log_event("startup_preflight_failed", {
                "kind": "embedding",
                "error": msg,
                "endpoint": emb_endpoint,
                "model": EMBEDDING_MODEL,
            })
        return PreflightResult(ok=False, error=msg, llm_endpoint=llm_endpoint, embedding_endpoint=emb_endpoint)

    # 3) Dim consistency with local indexes (if any)
    print(f"[preflight][embedding] OK (online_dim={online_dim})")
    if check_dim and online_dim is not None:
        print("[preflight][dim] checking local index dims against online_dim...")
        try:
            _check_index_dims(online_dim)
            print("[preflight][dim] OK")
        except Exception as e:
            msg = f"Embedding dim preflight failed: {e}"
            print(f"[preflight][dim] FAILED: {msg}")
            if logger:
                logger.log_event("startup_preflight_failed", {
                    "kind": "embedding_dim",
                    "error": msg,
                    "online_dim": online_dim,
                    "novelty_index_dir": str(NOVELTY_INDEX_DIR),
                    "recall_index_dir": str(PipelineConfig.RECALL_INDEX_DIR),
                    "recall_use_offline_index": bool(PipelineConfig.RECALL_USE_OFFLINE_INDEX),
                })
            return PreflightResult(
                ok=False,
                error=msg,
                llm_endpoint=llm_endpoint,
                embedding_endpoint=emb_endpoint,
                embedding_dim=online_dim,
            )
    else:
        print("[preflight][dim] skipped")

    if logger:
        logger.log_event("startup_preflight_ok", {
            "llm_endpoint": llm_endpoint,
            "provider": LLM_PROVIDER,
            "model": LLM_MODEL,
            "embedding_endpoint": emb_endpoint,
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dim": online_dim,
            "check_dim": bool(check_dim),
        })
    print("[preflight] ALL OK")
    return PreflightResult(ok=True, llm_endpoint=llm_endpoint, embedding_endpoint=emb_endpoint, embedding_dim=online_dim)

