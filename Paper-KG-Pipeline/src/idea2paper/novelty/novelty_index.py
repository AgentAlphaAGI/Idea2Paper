import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from idea2paper.infra.embeddings import get_embedding, EMBEDDING_MODEL


def _stable_string(value) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        parts = []
        for k in sorted(value.keys()):
            parts.append(f"{k}:{_stable_string(value[k])}")
        return " ".join(parts)
    if isinstance(value, (list, tuple)):
        return " ".join(_stable_string(v) for v in value)
    return str(value)


def build_story_text(story: Dict) -> str:
    parts = []
    parts.append(story.get("title", ""))
    parts.append(story.get("abstract", ""))
    parts.append(story.get("problem_framing", story.get("problem_definition", "")))
    method = story.get("method_skeleton", "")
    if isinstance(method, dict):
        parts.append(_stable_string(method))
    else:
        parts.append(str(method))
    claims = story.get("innovation_claims", story.get("claims", ""))
    parts.append(_stable_string(claims))
    experiments = story.get("experiments_plan", "")
    parts.append(_stable_string(experiments))
    return "\n".join([p for p in parts if p])


def build_paper_text(paper: Dict) -> str:
    parts = [
        paper.get("title", ""),
        paper.get("domain", ""),
        _stable_string(paper.get("sub_domains", "")),
        paper.get("idea", "")
    ]
    pattern_details = paper.get("pattern_details", {}) or {}
    if isinstance(pattern_details, dict):
        for k in ["base_problem", "solution_pattern", "story", "application"]:
            if k in pattern_details:
                parts.append(str(pattern_details.get(k, "")))
    else:
        parts.append(str(pattern_details))
    return "\n".join([p for p in parts if p])


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _normalize_matrix(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def _normalize_vec(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def _token_set(text: str) -> set:
    return set(text.lower().split())


def keyword_overlap(text1: str, text2: str) -> float:
    t1 = _token_set(text1)
    t2 = _token_set(text2)
    if not t1 or not t2:
        return 0.0
    return len(t1 & t2) / len(t1 | t2)


class NoveltyIndex:
    def __init__(self, papers: List[Dict], index_dir: Path, nodes_paper_path: Path, logger=None):
        self.papers = papers
        self.index_dir = Path(index_dir)
        self.nodes_paper_path = Path(nodes_paper_path)
        self.logger = logger

        self.meta_path = self.index_dir / "paper_meta.jsonl"
        self.emb_path = self.index_dir / "paper_emb.npy"
        self.manifest_path = self.index_dir / "index_manifest.json"

        self._embeddings = None
        self._paper_meta = None

    def ensure_index(self, force_rebuild: bool = False) -> Dict:
        status = {
            "rebuilt": False,
            "paper_count": len(self.papers),
            "skipped": 0,
            "embedding_available": True,
            "notes": []
        }
        if not self.index_dir.exists():
            self.index_dir.mkdir(parents=True, exist_ok=True)

        current_hash = _file_hash(self.nodes_paper_path) if self.nodes_paper_path.exists() else None

        if not force_rebuild and self.manifest_path.exists() and self.emb_path.exists() and self.meta_path.exists():
            try:
                manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
                if (
                    manifest.get("paper_count") == len(self.papers)
                    and manifest.get("nodes_paper_hash") == current_hash
                    and manifest.get("embedding_model") == EMBEDDING_MODEL
                ):
                    self._embeddings = np.load(self.emb_path)
                    self._paper_meta = [json.loads(l) for l in self.meta_path.read_text(encoding="utf-8").splitlines() if l.strip()]
                    status["notes"].append("index_reused")
                    return status
            except Exception as e:
                status["notes"].append(f"index_load_failed:{e}")

        # rebuild
        status["rebuilt"] = True
        vectors = []
        meta = []
        skipped = 0

        for paper in self.papers:
            text = build_paper_text(paper)
            emb = get_embedding(text, logger=self.logger)
            if emb is None:
                skipped += 1
                continue
            vectors.append(np.array(emb, dtype=np.float32))
            meta.append({
                "paper_id": paper.get("paper_id", ""),
                "title": paper.get("title", ""),
                "pattern_id": paper.get("pattern_id", ""),
                "domain": paper.get("domain", ""),
                "text_hash": hashlib.sha256(text.encode("utf-8")).hexdigest()
            })

        if not vectors:
            status["embedding_available"] = False
            status["notes"].append("no_embeddings_built")
            status["skipped"] = skipped
            return status

        mat = np.vstack(vectors)
        mat = _normalize_matrix(mat)

        np.save(self.emb_path, mat)
        self.meta_path.write_text("\n".join(json.dumps(m, ensure_ascii=False) for m in meta), encoding="utf-8")
        manifest = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "embedding_model": EMBEDDING_MODEL,
            "paper_count": len(self.papers),
            "index_count": len(meta),
            "skipped": skipped,
            "nodes_paper_hash": current_hash
        }
        self.manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

        self._embeddings = mat
        self._paper_meta = meta
        status["skipped"] = skipped
        return status

    def _ensure_loaded(self):
        if self._embeddings is None and self.emb_path.exists():
            self._embeddings = np.load(self.emb_path)
        if self._paper_meta is None and self.meta_path.exists():
            self._paper_meta = [json.loads(l) for l in self.meta_path.read_text(encoding="utf-8").splitlines() if l.strip()]

    def query(self, story_text: str, top_k: int) -> Tuple[List[Dict], Dict]:
        """Return candidates list and info dict."""
        self._ensure_loaded()
        info = {"embedding_available": True, "notes": []}

        if self._embeddings is None or self._paper_meta is None:
            info["embedding_available"] = False
            info["notes"].append("index_missing")
            return self._fallback_query(story_text, top_k), info

        story_emb = get_embedding(story_text, logger=self.logger)
        if story_emb is None:
            info["embedding_available"] = False
            info["notes"].append("story_embedding_failed")
            return self._fallback_query(story_text, top_k), info

        vec = _normalize_vec(np.array(story_emb, dtype=np.float32))
        scores = self._embeddings.dot(vec)
        top_idx = np.argsort(scores)[::-1][:top_k]
        candidates = []
        for idx in top_idx:
            meta = self._paper_meta[int(idx)]
            candidates.append({
                **meta,
                "cosine": float(scores[int(idx)]),
                "keyword_overlap": None
            })
        return candidates, info

    def _fallback_query(self, story_text: str, top_k: int) -> List[Dict]:
        results = []
        for paper in self.papers:
            text = build_paper_text(paper)
            overlap = keyword_overlap(story_text, text)
            results.append({
                "paper_id": paper.get("paper_id", ""),
                "title": paper.get("title", ""),
                "pattern_id": paper.get("pattern_id", ""),
                "domain": paper.get("domain", ""),
                "cosine": None,
                "keyword_overlap": overlap
            })
        results.sort(key=lambda x: x["keyword_overlap"], reverse=True)
        return results[:top_k]
