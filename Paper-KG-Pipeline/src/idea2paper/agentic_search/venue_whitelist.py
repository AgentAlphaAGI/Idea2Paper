"""
CS 顶会白名单 — 用于 Path4 Searcher 的 venue 硬过滤

规范化策略:
  1. 取 publicationVenue.name (若存在) 否则取 venue 字符串
  2. lower + 去标点 + 去多余空格
  3. 用白名单 alias 表做「包含匹配」
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

# ────────────────────────────────────────
# Alias table: each entry is (canonical_name, [alias_patterns])
# alias patterns are lowercased substrings for containment matching
# ────────────────────────────────────────

_CS_TOP_VENUES: List[Tuple[str, List[str]]] = [
    # ML / AI
    ("ICLR", ["iclr", "international conference on learning representations"]),
    ("ICML", ["icml", "international conference on machine learning"]),
    ("NeurIPS", ["neurips", "neural information processing systems", "nips"]),
    ("AAAI", ["aaai", "association for the advancement of artificial intelligence"]),

    # CV
    ("CVPR", ["cvpr", "computer vision and pattern recognition"]),
    ("ICCV", ["iccv", "international conference on computer vision"]),
    ("ECCV", ["eccv", "european conference on computer vision"]),

    # NLP
    ("ACL", ["annual meeting of the association for computational linguistics"]),
    ("EMNLP", ["emnlp", "empirical methods in natural language processing"]),
    ("NAACL", ["naacl", "north american chapter"]),

    # Data / IR
    ("KDD", ["kdd", "knowledge discovery and data mining"]),
    ("WWW", ["www", "world wide web"]),
    ("SIGIR", ["sigir", "information retrieval"]),

    # Systems / Robotics / Others
    ("CoRL", ["corl", "conference on robot learning"]),
    ("RSS", ["robotics: science and systems", "rss "]),
    ("OSDI", ["osdi", "operating systems design and implementation"]),
    ("SOSP", ["sosp", "symposium on operating systems principles"]),
    ("SIGMOD", ["sigmod"]),
    ("VLDB", ["vldb", "very large data bases"]),
    ("ICRA", ["icra", "international conference on robotics and automation"]),
    ("IJCAI", ["ijcai", "international joint conference on artificial intelligence"]),
]

_STRIP_RE = re.compile(r"[^\w\s]")
_MULTI_SPACE_RE = re.compile(r"\s+")


def _normalize_venue_text(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = _STRIP_RE.sub(" ", text)
    text = _MULTI_SPACE_RE.sub(" ", text).strip()
    return text


def _build_alias_index(
    venues: List[Tuple[str, List[str]]],
) -> List[Tuple[str, str]]:
    """Pre-normalize alias patterns for fast matching."""
    idx = []
    for canonical, aliases in venues:
        for alias in aliases:
            idx.append((canonical, _normalize_venue_text(alias)))
    return idx


_ALIAS_INDEX = _build_alias_index(_CS_TOP_VENUES)


def match_venue(
    venue_name: Optional[str],
    publication_venue: Optional[Dict],
) -> Optional[str]:
    """Check if a paper's venue matches the CS top whitelist.

    Args:
        venue_name: The ``venue`` string from Semantic Scholar.
        publication_venue: The ``publicationVenue`` dict from S2 (may have ``name``).

    Returns:
        Canonical venue name (e.g. "ICLR") if matched, else None.
    """
    raw = ""
    if publication_venue and isinstance(publication_venue, dict):
        raw = publication_venue.get("name") or ""
    if not raw and venue_name:
        raw = venue_name
    if not raw:
        return None

    normalized = _normalize_venue_text(raw)
    if not normalized:
        return None

    for canonical, alias_norm in _ALIAS_INDEX:
        if alias_norm in normalized:
            return canonical

    return None


def get_whitelist_names() -> List[str]:
    """Return a list of canonical venue names in the whitelist."""
    return [name for name, _ in _CS_TOP_VENUES]
