# 中文注释: 引用链路的输出安全校验（禁止编造引用事实）。
from __future__ import annotations

import re

_FORBIDDEN_PATTERNS = [
    r"\bdoi\b",
    r"arxiv",
    r"bibtex",
    r"\bvenue\b",
    r"\bconference\b",
    r"\bjournal\b",
    r"proceedings",
    r"https?://",
]


def contains_forbidden_citation_facts(text: str) -> bool:
    if not text:
        return False
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in _FORBIDDEN_PATTERNS)
