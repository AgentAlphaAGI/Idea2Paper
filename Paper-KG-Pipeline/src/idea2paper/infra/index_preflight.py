import json
import os
import time
import hashlib
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Optional

import numpy as np

try:
    import fcntl  # type: ignore
except Exception:  # pragma: no cover
    fcntl = None  # Windows or unavailable


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _count_jsonl_lines(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _load_manifest(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


@contextmanager
def acquire_lock(lock_path: Path, timeout_sec: int = 600):
    """
    File lock for index build. Uses fcntl if available; otherwise a simple lockfile.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()

    if fcntl is not None:
        f = lock_path.open("a+")
        acquired = False
        try:
            while True:
                try:
                    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                    break
                except BlockingIOError:
                    if time.time() - start > timeout_sec:
                        raise TimeoutError(f"Lock timeout: {lock_path}")
                    time.sleep(1)
            yield
        finally:
            if acquired:
                try:
                    fcntl.flock(f, fcntl.LOCK_UN)
                except Exception:
                    pass
            f.close()
    else:
        fd = None
        try:
            while True:
                try:
                    fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
                    break
                except FileExistsError:
                    if time.time() - start > timeout_sec:
                        raise TimeoutError(f"Lock timeout: {lock_path}")
                    time.sleep(1)
            yield
        finally:
            if fd is not None:
                os.close(fd)
            try:
                lock_path.unlink()
            except Exception:
                pass


def validate_novelty_index(index_dir: Path, nodes_paper_path: Path, embedding_model: str) -> Dict:
    index_dir = Path(index_dir)
    meta_path = index_dir / "paper_meta.jsonl"
    emb_path = index_dir / "paper_emb.npy"
    manifest_path = index_dir / "index_manifest.json"

    result = {
        "ok": False,
        "reason": "missing",
        "details": {},
        "paths": {
            "index_dir": str(index_dir),
            "manifest": str(manifest_path),
        },
    }

    if not (meta_path.exists() and emb_path.exists() and manifest_path.exists()):
        return result

    manifest = _load_manifest(manifest_path)
    if not manifest:
        result["reason"] = "load_failed"
        return result

    if not nodes_paper_path.exists():
        result["reason"] = "missing_nodes"
        return result

    current_hash = sha256_file(nodes_paper_path)
    result["details"]["nodes_paper_hash"] = current_hash
    result["details"]["manifest_hash"] = manifest.get("nodes_paper_hash")
    result["details"]["embedding_model"] = embedding_model
    result["details"]["manifest_model"] = manifest.get("embedding_model")

    if manifest.get("embedding_model") != embedding_model:
        result["reason"] = "mismatch"
        return result
    if manifest.get("nodes_paper_hash") != current_hash:
        result["reason"] = "mismatch"
        return result

    try:
        emb_count = int(np.load(emb_path, mmap_mode="r").shape[0])
    except Exception:
        result["reason"] = "load_failed"
        return result
    meta_count = _count_jsonl_lines(meta_path)

    result["details"]["emb_count"] = emb_count
    result["details"]["meta_count"] = meta_count
    result["details"]["index_count"] = manifest.get("index_count")
    result["details"]["paper_count"] = manifest.get("paper_count")

    if meta_count != emb_count:
        result["reason"] = "incomplete"
        return result
    if manifest.get("index_count") is not None and meta_count != int(manifest.get("index_count")):
        result["reason"] = "incomplete"
        return result
    if manifest.get("paper_count") is not None and manifest.get("index_count") is not None:
        expected = int(manifest.get("index_count")) + int(manifest.get("skipped", 0))
        if int(manifest.get("paper_count")) != expected:
            result["reason"] = "incomplete"
            return result

    result["ok"] = True
    result["reason"] = "ok"
    return result


def _validate_recall_kind(kind: str, index_dir: Path, nodes_path: Path, embedding_model: str) -> Dict:
    emb_path = index_dir / f"{kind}_emb.npy"
    meta_path = index_dir / f"{kind}_meta.jsonl"
    manifest_path = index_dir / f"{kind}_manifest.json"
    result = {
        "ok": False,
        "reason": "missing",
        "details": {},
        "paths": {
            "manifest": str(manifest_path),
        },
    }
    if not (emb_path.exists() and meta_path.exists() and manifest_path.exists()):
        return result
    manifest = _load_manifest(manifest_path)
    if not manifest:
        result["reason"] = "load_failed"
        return result
    if not nodes_path.exists():
        result["reason"] = "missing_nodes"
        return result
    current_hash = sha256_file(nodes_path)
    result["details"][f"nodes_{kind}_hash"] = current_hash
    result["details"]["manifest_hash"] = manifest.get(f"nodes_{kind}_hash")
    result["details"]["embedding_model"] = embedding_model
    result["details"]["manifest_model"] = manifest.get("embedding_model")
    if manifest.get("embedding_model") != embedding_model:
        result["reason"] = "mismatch"
        return result
    if manifest.get(f"nodes_{kind}_hash") != current_hash:
        result["reason"] = "mismatch"
        return result
    try:
        emb_count = int(np.load(emb_path, mmap_mode="r").shape[0])
    except Exception:
        result["reason"] = "load_failed"
        return result
    meta_count = _count_jsonl_lines(meta_path)
    result["details"]["emb_count"] = emb_count
    result["details"]["meta_count"] = meta_count
    result["details"]["index_count"] = manifest.get("index_count")
    result["details"]["count"] = manifest.get("count")
    if meta_count != emb_count:
        result["reason"] = "incomplete"
        return result
    if manifest.get("index_count") is not None and meta_count != int(manifest.get("index_count")):
        result["reason"] = "incomplete"
        return result
    result["ok"] = True
    result["reason"] = "ok"
    return result


def validate_recall_index(index_dir: Path, nodes_paper_path: Path, nodes_idea_path: Path, embedding_model: str) -> Dict:
    index_dir = Path(index_dir)
    result = {
        "ok": False,
        "reason": "missing",
        "details": {},
        "paths": {
            "index_dir": str(index_dir),
        },
    }
    idea = _validate_recall_kind("idea", index_dir, nodes_idea_path, embedding_model)
    paper = _validate_recall_kind("paper", index_dir, nodes_paper_path, embedding_model)
    result["details"]["idea"] = idea
    result["details"]["paper"] = paper
    if idea["ok"] and paper["ok"]:
        result["ok"] = True
        result["reason"] = "ok"
    else:
        result["reason"] = "mismatch"
        if idea["reason"] == "missing" or paper["reason"] == "missing":
            result["reason"] = "missing"
        if idea["reason"] == "incomplete" or paper["reason"] == "incomplete":
            result["reason"] = "incomplete"
        if idea["reason"] == "load_failed" or paper["reason"] == "load_failed":
            result["reason"] = "load_failed"
        if idea["reason"] == "missing_nodes" or paper["reason"] == "missing_nodes":
            result["reason"] = "missing_nodes"
    return result
