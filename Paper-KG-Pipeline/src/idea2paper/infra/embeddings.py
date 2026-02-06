import time
import os
import hashlib
import json
import pickle
from pathlib import Path
from typing import Optional, List

import requests

from idea2paper.config import (
    EMBEDDING_API_KEY,
    EMBEDDING_API_URL,
    EMBEDDING_MODEL,
    EMBEDDING_PROVIDER,
    OUTPUT_DIR,
)
from idea2paper.infra.run_context import get_logger

# 本地缓存目录
_CACHE_DIR = OUTPUT_DIR / "embedding_cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_CACHE_FILE = _CACHE_DIR / "local_embeddings.pkl"

# 内存缓存
_memory_cache = {}
_cache_loaded = False

def _load_cache():
    global _cache_loaded, _memory_cache
    if _cache_loaded:
        return
    if _CACHE_FILE.exists():
        try:
            with open(_CACHE_FILE, "rb") as f:
                _memory_cache = pickle.load(f)
            print(f"📦 已加载本地 Embedding 缓存 ({len(_memory_cache)} 条)")
        except Exception as e:
            print(f"⚠️  加载缓存失败: {e}")
    _cache_loaded = True

def _save_cache():
    try:
        with open(_CACHE_FILE, "wb") as f:
            pickle.dump(_memory_cache, f)
    except Exception as e:
        print(f"⚠️  保存缓存失败: {e}")

def _get_cache_key(text: str, model: str) -> str:
    """生成缓存键：模型名 + 文本哈希"""
    content = f"{model}:{text}"
    return hashlib.md5(content.encode('utf-8')).hexdigest()

def get_embedding(text: str, logger=None, timeout: int = 120) -> Optional[List[float]]:
    """Get embedding for text using SiliconFlow/Ollama API with local caching.
    
    Returns None on failure (no exception thrown).
    """
    _load_cache()
    
    # 检查缓存
    cache_key = _get_cache_key(text, EMBEDDING_MODEL)
    if cache_key in _memory_cache:
        # print(f"✨ 使用缓存 Embedding ({cache_key[:8]})")
        return _memory_cache[cache_key]

    if logger is None:
        logger = get_logger()
    start_ts = time.time()

    if not EMBEDDING_API_KEY and EMBEDDING_PROVIDER != "ollama":
        if logger:
            logger.log_embedding_call(
                request={
                    "provider": EMBEDDING_PROVIDER,
                    "url": EMBEDDING_API_URL,
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
        "Content-Type": "application/json"
    }
    if EMBEDDING_API_KEY:
        headers["Authorization"] = f"Bearer {EMBEDDING_API_KEY}"

    payload = {
        "model": EMBEDDING_MODEL,
        "input": text
    }

    try:
        print(f"   🧠 调用 Embedding: {EMBEDDING_MODEL}...", end="", flush=True)
        resp = requests.post(EMBEDDING_API_URL, headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
        latency = time.time() - start_ts
        print(f" ✅ ({latency:.2f}s)")

        data = resp.json()
        emb = data["data"][0]["embedding"]
        
        # 写入缓存
        _memory_cache[cache_key] = emb
        _save_cache()
        
        if logger:
            logger.log_embedding_call(
                request={
                    "provider": EMBEDDING_PROVIDER,
                    "url": EMBEDDING_API_URL,
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
        print(f" ❌ 失败: {str(e)[:50]}")
        if logger:
            logger.log_embedding_call(
                request={
                    "provider": EMBEDDING_PROVIDER,
                    "url": EMBEDDING_API_URL,
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
    start_ts = time.time()

    # 1. 加载缓存
    _load_cache()
    
    # 2. 检查批量缓存
    texts_to_compute = []
    indices_to_compute = []
    results = [None] * len(texts)
    
    for i, text in enumerate(texts):
        cache_key = _get_cache_key(text, EMBEDDING_MODEL)
        if cache_key in _memory_cache:
            results[i] = _memory_cache[cache_key]
        else:
            texts_to_compute.append(text)
            indices_to_compute.append(i)
            
    # 3. 如果全部命中缓存，直接返回
    if not texts_to_compute:
        return results

    if not EMBEDDING_API_KEY and EMBEDDING_PROVIDER != "ollama":
        if logger:
            logger.log_embedding_call(
                request={
                    "provider": EMBEDDING_PROVIDER,
                    "url": EMBEDDING_API_URL,
                    "model": EMBEDDING_MODEL,
                    "input_preview": _preview_texts(texts),
                    "timeout": timeout,
                    "simulated": True,
                    "batch_size": len(texts),
                },
                response={
                    "ok": False,
                    "latency_ms": int((time.time() - start_ts) * 1000),
                    "error": "SILICONFLOW_API_KEY not configured"
                }
            )
        return None

    headers = {
        "Content-Type": "application/json"
    }
    if EMBEDDING_API_KEY:
        headers["Authorization"] = f"Bearer {EMBEDDING_API_KEY}"

    payload = {
        "model": EMBEDDING_MODEL,
        "input": texts_to_compute
    }

    try:
        print(f"   🧠 批量 Embedding ({len(texts_to_compute)}/{len(texts)} new): {EMBEDDING_MODEL}...", end="", flush=True)
        resp = requests.post(EMBEDDING_API_URL, headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
        latency = time.time() - start_ts
        print(f" ✅ ({latency:.2f}s)")
        
        data = resp.json()
        new_embs = [item["embedding"] for item in data.get("data", [])]
        
        if len(new_embs) != len(texts_to_compute):
            raise ValueError(f"embedding batch size mismatch: got {len(new_embs)} expected {len(texts_to_compute)}")
            
        # 更新缓存并填充结果
        for idx, emb in zip(indices_to_compute, new_embs):
            results[idx] = emb
            cache_key = _get_cache_key(texts[idx], EMBEDDING_MODEL)
            _memory_cache[cache_key] = emb
            
        _save_cache()
        
        if logger:
            logger.log_embedding_call(
                request={
                    "provider": EMBEDDING_PROVIDER,
                    "url": EMBEDDING_API_URL,
                    "model": EMBEDDING_MODEL,
                    "input_preview": _preview_texts(texts),
                    "timeout": timeout,
                    "simulated": False,
                    "batch_size": len(texts),
                },
                response={
                    "ok": True,
                    "latency_ms": int((time.time() - start_ts) * 1000)
                }
            )
        return results
    except Exception as e:
        if logger:
            logger.log_embedding_call(
                request={
                    "provider": EMBEDDING_PROVIDER,
                    "url": EMBEDDING_API_URL,
                    "model": EMBEDDING_MODEL,
                    "input_preview": _preview_texts(texts),
                    "timeout": timeout,
                    "simulated": False,
                    "batch_size": len(texts),
                },
                response={
                    "ok": False,
                    "latency_ms": int((time.time() - start_ts) * 1000),
                    "error": str(e)
                }
            )
        return None
