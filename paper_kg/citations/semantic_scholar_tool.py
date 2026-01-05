# 中文注释: 离线引用检索工具（Mock 版）。
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Dict, List, Optional


@dataclass
class PaperCandidate:
    paper_id: str
    title: str
    year: Optional[int]
    venue: str
    authors: List[str]
    external_ids: Dict[str, str]
    url: Optional[str]


@dataclass
class PaperDetails:
    paper_id: str
    title: str
    year: Optional[int]
    venue: str
    authors: List[str]
    external_ids: Dict[str, str]
    url: Optional[str]
    citation_styles: Dict[str, str]


class MockSemanticScholarTool:
    """Mock 版本：本地生成可验证字段，避免外部请求。"""

    def __init__(self, seed: str = "mock") -> None:
        self.seed = seed
        self._cache: Dict[str, PaperDetails] = {}

    def search(
        self,
        query: str,
        limit: int = 5,
        fields: str = "paperId,title,authors,venue,year,externalIds,url",
    ) -> List[PaperCandidate]:
        if not query:
            return []
        paper_id = self._make_paper_id(query)
        details = self._get_or_create_details(paper_id, query)
        candidate = PaperCandidate(
            paper_id=details.paper_id,
            title=details.title,
            year=details.year,
            venue=details.venue,
            authors=details.authors,
            external_ids=details.external_ids,
            url=details.url,
        )
        return [candidate][: max(1, limit)]

    def fetch_details(
        self,
        paper_id: str,
        fields: str = "paperId,title,authors,venue,year,externalIds,url,citationStyles",
    ) -> Optional[PaperDetails]:
        if not paper_id:
            return None
        return self._get_or_create_details(paper_id, None)

    def fetch_bibtex(self, paper_id: str) -> str:
        details = self.fetch_details(paper_id)
        if details is None:
            return ""
        bibtex = details.citation_styles.get("bibtex") if isinstance(details.citation_styles, dict) else ""
        return bibtex or ""

    def _make_paper_id(self, query: str) -> str:
        digest = hashlib.sha1(f"{self.seed}:{query}".encode("utf-8")).hexdigest()[:10]
        return f"mock-{digest}"

    def _get_or_create_details(self, paper_id: str, query: Optional[str]) -> PaperDetails:
        cached = self._cache.get(paper_id)
        if cached is not None:
            return cached

        key = query or paper_id
        digest = hashlib.sha1(f"{self.seed}:{key}".encode("utf-8")).hexdigest()
        year = 2018 + (int(digest[:2], 16) % 8)
        title_base = query or f"Mock Paper {paper_id}"
        title = f"Mock Study on {title_base}".strip()[:160]
        doi = f"10.0000/{paper_id}"
        url = f"https://example.org/{paper_id}"
        details = PaperDetails(
            paper_id=paper_id,
            title=title,
            year=year,
            venue="MockVenue",
            authors=["Mock, Alice", "Mock, Bob"],
            external_ids={"DOI": doi},
            url=url,
            citation_styles={},
        )
        self._cache[paper_id] = details
        return details
