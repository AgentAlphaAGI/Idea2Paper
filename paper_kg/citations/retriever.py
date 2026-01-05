# 中文注释: 引用检索接口与默认实现（可注入 FakeProvider）。
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, List, Protocol

from paper_kg.citations.bibtex_utils import build_bibtex, make_bibkey
from paper_kg.citations.models import CitationNeed, RetrievedCitation


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


def get_citation_provider(require_verifiable: bool) -> ICitationProvider:
    config = CitationProviderConfig(require_verifiable=require_verifiable)
    return MockCitationProvider(config)


def _mock_seed(candidate_id: str, keywords: List[str]) -> int:
    raw = f"{candidate_id}::{' '.join(keywords)}".encode("utf-8")
    digest = hashlib.sha1(raw).hexdigest()[:8]
    return int(digest, 16) if digest else 0
