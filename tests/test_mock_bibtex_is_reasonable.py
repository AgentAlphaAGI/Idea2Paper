# 中文注释: 验证 mock 引用的 BibTeX 结构合理且可用。

from paper_kg.citations.models import CitationNeed
from paper_kg.citations.retriever import CitationProviderConfig, MockCitationProvider


def test_mock_bibtex_contains_basic_fields() -> None:
    provider = MockCitationProvider(CitationProviderConfig(require_verifiable=True))
    need = CitationNeed(candidate_id="sec_intro_001", need_type="HARD", keywords=["test"], year_range="2018-2026")
    result = provider.search_and_resolve(need)
    bibtex = result.bibtex

    assert bibtex.startswith("@")
    assert "title =" in bibtex
    assert "author =" in bibtex
    assert "year =" in bibtex
    assert result.bibkey


def test_mock_bibtex_keys_unique() -> None:
    provider = MockCitationProvider(CitationProviderConfig(require_verifiable=True))
    need_a = CitationNeed(candidate_id="sec_intro_001", need_type="HARD", keywords=["alpha"], year_range="2018-2026")
    need_b = CitationNeed(candidate_id="sec_intro_002", need_type="HARD", keywords=["beta"], year_range="2018-2026")
    result_a = provider.search_and_resolve(need_a)
    result_b = provider.search_and_resolve(need_b)

    assert result_a.bibkey != result_b.bibkey
