# 中文注释: 引用检索接口与默认实现（可注入 FakeProvider）。
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol

import httpx

from paper_kg.citations.bibtex_utils import build_bibtex, make_bibkey
from paper_kg.citations.models import CitationNeed, RetrievedCitation


logger = logging.getLogger(__name__)


class ICitationProvider(Protocol):
    """引用检索接口。"""

    def search_and_resolve(self, need: CitationNeed) -> RetrievedCitation:  # noqa: D401
        """根据引用需求检索并返回引用结果。"""


@dataclass
class CitationProviderConfig:
    require_verifiable: bool
    timeout_sec: float = 20.0


class NullCitationProvider:
    """空实现：永远返回 NOT_FOUND（用于离线/测试）。"""

    def __init__(self, _config: CitationProviderConfig) -> None:
        self.config = _config

    def search_and_resolve(self, need: CitationNeed) -> RetrievedCitation:
        return RetrievedCitation(candidate_id=need.candidate_id, status="NOT_FOUND")


class SemanticScholarProvider:
    """Semantic Scholar 检索实现。"""

    def __init__(self, config: CitationProviderConfig) -> None:
        self.config = config

    def search_and_resolve(self, need: CitationNeed) -> RetrievedCitation:
        query = " ".join(need.keywords).strip() or need.candidate_id
        if not query:
            return RetrievedCitation(candidate_id=need.candidate_id, status="NOT_FOUND")

        params = {
            "query": query,
            "limit": 5,
            "fields": "paperId,title,authors,venue,year,externalIds,url",
        }
        try:
            with httpx.Client(timeout=self.config.timeout_sec) as client:
                resp = client.get("https://api.semanticscholar.org/graph/v1/paper/search", params=params)
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:  # noqa: BLE001
            logger.warning("CITATION_RETRIEVE_FAIL: provider=semantic_scholar err=%s", exc)
            return RetrievedCitation(candidate_id=need.candidate_id, status="NOT_FOUND")

        items = data.get("data", []) if isinstance(data, dict) else []
        if not items:
            return RetrievedCitation(candidate_id=need.candidate_id, status="NOT_FOUND")

        year_min, year_max = _parse_year_range(need.year_range)
        picked = _pick_best_item(items, year_min, year_max)
        return _build_retrieved_from_item(need.candidate_id, picked, self.config.require_verifiable)


class MockCitationProvider:
    """Mock 引用实现：生成可验证字段以便端到端测试。"""

    def __init__(self, config: CitationProviderConfig) -> None:
        self.config = config

    def search_and_resolve(self, need: CitationNeed) -> RetrievedCitation:
        seed = _mock_seed(need.candidate_id, need.keywords)
        title = f"Mock Reference for {need.candidate_id}"
        year = 2018 + (seed % 8)
        venue = "MockVenue"
        authors = ["Mock, Alice", "Mock, Bob"]
        paper_id = f"mock-{seed:08x}"
        doi = f"10.0000/{paper_id}"
        url = f"https://example.org/{paper_id}"

        evidence = [f"doi:{doi}", url]
        if self.config.require_verifiable and not evidence:
            return RetrievedCitation(candidate_id=need.candidate_id, status="NOT_FOUND")

        bibkey = make_bibkey(doi=doi, arxiv=None, title=title, year=year)
        bibtex = build_bibtex(
            bibkey=bibkey,
            title=title,
            authors=authors,
            venue=venue,
            year=year,
            doi=doi,
            url=url,
        )
        return RetrievedCitation(
            candidate_id=need.candidate_id,
            status="FOUND",
            paper_id=paper_id,
            bibtex=bibtex,
            bibkey=bibkey,
            doi=doi,
            arxiv=None,
            url=url,
            title=title,
            authors=authors,
            year=year,
            venue=venue,
            evidence=evidence,
        )


def get_citation_provider(name: str, require_verifiable: bool) -> ICitationProvider:
    config = CitationProviderConfig(require_verifiable=require_verifiable)
    name = (name or "").lower()
    if name in {"mock", "fake", "fixture"}:
        return MockCitationProvider(config)
    if name in {"semantic_scholar", "semanticscholar", "s2"}:
        return SemanticScholarProvider(config)
    if name in {"openalex", "crossref", "arxiv"}:
        # 中文注释：占位实现，后续可扩展；当前返回空结果以避免误用。
        return NullCitationProvider(config)
    return NullCitationProvider(config)


def _parse_year_range(year_range: str) -> tuple[Optional[int], Optional[int]]:
    if not year_range:
        return None, None
    match = re.match(r"^\\s*(\\d{4})\\s*-\\s*(\\d{4})\\s*$", year_range)
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def _pick_best_item(items: List[Dict], year_min: Optional[int], year_max: Optional[int]) -> Dict:
    if year_min is None or year_max is None:
        return items[0]
    for item in items:
        year = item.get("year")
        if isinstance(year, int) and year_min <= year <= year_max:
            return item
    return items[0]


def _build_retrieved_from_item(
    candidate_id: str, item: Dict, require_verifiable: bool
) -> RetrievedCitation:
    title = item.get("title") or ""
    authors = [a.get("name") for a in item.get("authors", []) if isinstance(a, dict)]
    venue = item.get("venue") or ""
    year = item.get("year") if isinstance(item.get("year"), int) else None
    paper_id = item.get("paperId") if isinstance(item.get("paperId"), str) else None
    external = item.get("externalIds") if isinstance(item.get("externalIds"), dict) else {}
    doi = external.get("DOI") if isinstance(external, dict) else None
    arxiv = external.get("ArXiv") if isinstance(external, dict) else None
    url = item.get("url") or None

    evidence: List[str] = []
    if doi:
        evidence.append(f"doi:{doi}")
    if arxiv:
        evidence.append(f"arxiv:{arxiv}")
    if url:
        evidence.append(url)

    if require_verifiable and not evidence:
        return RetrievedCitation(candidate_id=candidate_id, status="NOT_FOUND")

    bibkey = make_bibkey(doi=doi, arxiv=arxiv, title=title, year=year)
    bibtex = build_bibtex(
        bibkey=bibkey,
        title=title,
        authors=authors,
        venue=venue,
        year=year,
        doi=doi,
        url=url,
    )
    return RetrievedCitation(
        candidate_id=candidate_id,
        status="FOUND",
        paper_id=paper_id,
        bibtex=bibtex,
        bibkey=bibkey,
        doi=doi,
        arxiv=arxiv,
        url=url,
        title=title,
        authors=authors,
        year=year,
        venue=venue,
        evidence=evidence,
    )


def _mock_seed(candidate_id: str, keywords: List[str]) -> int:
    raw = f"{candidate_id}::{' '.join(keywords)}".encode("utf-8")
    digest = hashlib.sha1(raw).hexdigest()[:8]
    return int(digest, 16) if digest else 0
