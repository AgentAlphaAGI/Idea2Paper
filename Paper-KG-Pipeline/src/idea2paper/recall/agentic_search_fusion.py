"""
Agentic Search recall fusion (weighted RRF).

Merge original KG recall ranking and Agentic Search ranking by rank positions
instead of raw scores, because the two score systems are not directly comparable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Any


@dataclass
class WeightedRRFConfig:
    """Configuration for weighted reciprocal rank fusion."""

    k: int = 60
    weight_old: float = 0.7
    weight_agentic: float = 0.3
    top_n_old: int = 10
    top_m_agentic: int = 8
    final_top_k: int = 10


def _safe_pattern_id(pattern: Dict[str, Any]) -> str | None:
    if not isinstance(pattern, dict):
        return None
    pid = pattern.get("pattern_id")
    if not pid:
        return None
    return str(pid)


def _rrf_add(scores: Dict[str, float], pattern_id: str, rank: int, weight: float, k: int) -> None:
    if rank <= 0:
        return
    scores[pattern_id] = scores.get(pattern_id, 0.0) + float(weight) * (1.0 / float(k + rank))


def fuse_old_and_agentic_search(
    old_ranked: List[Tuple[str, Dict[str, Any], float]],
    agentic_ranked: List[Dict[str, Any]],
    config: WeightedRRFConfig,
) -> List[Tuple[str, Dict[str, Any], float]]:
    """Fuse two ranked lists and return a unified top-k list.

    Args:
        old_ranked: original recall ranked list in format [(pattern_id, pattern_info, score), ...]
        agentic_ranked: agentic search ranked patterns (dicts), sorted by internal rank
        config: weighted RRF config
    """
    rrf_scores: Dict[str, float] = {}
    pattern_info_map: Dict[str, Dict[str, Any]] = {}

    # Original recall ranked list.
    for rank, item in enumerate(old_ranked[: max(0, int(config.top_n_old))], start=1):
        pattern_id, pattern_info, _raw_score = item
        pid = str(pattern_id)
        pattern_info_map[pid] = pattern_info or {}
        _rrf_add(rrf_scores, pid, rank, config.weight_old, config.k)

    # Agentic search ranked list.
    sliced_agentic = agentic_ranked[: max(0, int(config.top_m_agentic))]
    for fallback_rank, pattern in enumerate(sliced_agentic, start=1):
        pid = _safe_pattern_id(pattern)
        if not pid:
            continue
        rank_scores = pattern.get("rank_scores") if isinstance(pattern, dict) else None
        rank = None
        if isinstance(rank_scores, dict):
            try:
                rank = int(rank_scores.get("rank"))
            except Exception:
                rank = None
        if rank is None or rank <= 0:
            rank = fallback_rank

        if pid not in pattern_info_map:
            pattern_info_map[pid] = pattern

        _rrf_add(rrf_scores, pid, rank, config.weight_agentic, config.k)

    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    top_k = ranked[: max(0, int(config.final_top_k))]

    return [(pid, pattern_info_map.get(pid, {}), score) for pid, score in top_k]
