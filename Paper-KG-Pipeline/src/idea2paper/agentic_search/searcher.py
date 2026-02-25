"""
Path4 Searcher — Semantic Scholar 检索 + 白名单硬过滤

职责:
  - 对每条 Planner query 调用 Semantic Scholar API
  - 白名单硬过滤: venue ∉ 白名单 → 直接丢弃
  - 跨 query 去重 (按 paperId)
  - 输出 PaperStub 列表供后续 Extractor 使用

v0.1 不包含:
  - 与 KG 存量的去重 (后续 Dedup 模块)
  - 缓存 / 重试 backtrack
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import requests

from idea2paper.config import PipelineConfig
from idea2paper.infra.run_context import get_logger
from idea2paper.agentic_search.planner import QuerySpec
from idea2paper.agentic_search.venue_whitelist import match_venue


# ────────────────────────────────────────
# Data models
# ────────────────────────────────────────

@dataclass
class PaperStub:
    """Minimal paper record from Searcher output."""
    paper_id: str
    title: str
    abstract: str
    venue_raw: str
    venue_norm: str  # canonical whitelist name (e.g. "ICLR")
    year: int
    url: str
    authors: List[str] = field(default_factory=list)
    citation_count: int = 0
    source_query: str = ""
    external_ids: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SearcherOutput:
    """Full output of the Searcher stage."""
    papers: List[PaperStub]
    stats: Dict

    def to_dict(self) -> dict:
        return {
            "papers": [p.to_dict() for p in self.papers],
            "stats": self.stats,
        }


# ────────────────────────────────────────
# Semantic Scholar API
# ────────────────────────────────────────

S2_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

S2_FIELDS = ",".join([
    "paperId",
    "externalIds",
    "title",
    "abstract",
    "venue",
    "publicationVenue",
    "year",
    "publicationDate",
    "authors",
    "url",
    "isOpenAccess",
    "openAccessPdf",
    "citationCount",
    "influentialCitationCount",
    "tldr",
])

_REQUEST_INTERVAL_SEC_ANON = 2.2
_REQUEST_INTERVAL_SEC_API_KEY = 1.0


def _compute_backoff_seconds(attempt: int, retry_after: Optional[str] = None) -> float:
    """Exponential backoff with optional Retry-After override."""
    if retry_after:
        try:
            return max(1.0, float(retry_after))
        except Exception:
            pass
    # 2, 4, 8, 12, 12 ...
    return min(12.0, 2.0 * (2 ** max(0, attempt - 1)))


def _search_s2(
    query: str,
    year: str = "2024-2026",
    limit: int = 50,
    timeout: int = 30,
    api_key: Optional[str] = None,
) -> List[Dict]:
    """Call Semantic Scholar paper search endpoint.

    Returns raw paper dicts from the API response, or empty list on failure.
    """
    params = {
        "query": query,
        "fields": S2_FIELDS,
        "fieldsOfStudy": "Computer Science",
        "year": year,
        "limit": min(limit, 100),
    }
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key

    max_attempts = 6 if not api_key else 4
    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.get(
                S2_SEARCH_URL,
                params=params,
                headers=headers,
                timeout=timeout,
            )

            if resp.status_code == 200:
                data = resp.json()
                return data.get("data", [])

            if resp.status_code == 429:
                wait_s = _compute_backoff_seconds(
                    attempt,
                    resp.headers.get("Retry-After"),
                )
                print(f"    ⚠️  S2 rate limit (attempt {attempt}/{max_attempts}), wait {wait_s:.1f}s...")
                if attempt < max_attempts:
                    time.sleep(wait_s)
                    continue

            if 500 <= resp.status_code < 600:
                wait_s = _compute_backoff_seconds(attempt)
                print(f"    ⚠️  S2 server error {resp.status_code} (attempt {attempt}/{max_attempts}), wait {wait_s:.1f}s...")
                if attempt < max_attempts:
                    time.sleep(wait_s)
                    continue

            # Non-retryable HTTP errors (or retries exhausted)
            print(f"    ❌ S2 API HTTP error {resp.status_code} after {attempt} attempts")
            return []
        except requests.RequestException as e:
            wait_s = _compute_backoff_seconds(attempt)
            print(f"    ⚠️  S2 request error (attempt {attempt}/{max_attempts}): {e}")
            if attempt < max_attempts:
                time.sleep(wait_s)
                continue
            print("    ❌ S2 API request failed after retries")
            return []

    return []


# ────────────────────────────────────────
# AgenticSearcher
# ────────────────────────────────────────

class AgenticSearcher:
    """Search Semantic Scholar and apply venue whitelist hard filter.

    Usage:
        searcher = AgenticSearcher()
        output = searcher.search(planner_output.final_queries)
        for paper in output.papers:
            print(paper.title, paper.venue_norm)
    """

    def __init__(self, logger=None, s2_api_key: Optional[str] = None):
        self.logger = logger or get_logger()
        self._year_range = str(PipelineConfig.PATH4_YEAR_RANGE)
        self._limit_per_query = int(PipelineConfig.PATH4_SEARCH_LIMIT_PER_QUERY)
        self._s2_api_key = s2_api_key
        self._request_interval_sec = (
            _REQUEST_INTERVAL_SEC_API_KEY if s2_api_key else _REQUEST_INTERVAL_SEC_ANON
        )

    def search(self, queries: List[QuerySpec]) -> SearcherOutput:
        """Execute search for each query and apply venue whitelist.

        Args:
            queries: List of QuerySpec from Planner.

        Returns:
            SearcherOutput with deduplicated, whitelist-filtered papers.
        """
        print(f"\n🔎 [Path4 Searcher] 执行 {len(queries)} 条 query 的联网检索...")

        all_papers: Dict[str, PaperStub] = {}  # paperId -> PaperStub (dedup)
        stats = {
            "total_raw": 0,
            "total_after_whitelist": 0,
            "total_after_dedup": 0,
            "per_query": [],
        }

        for i, qspec in enumerate(queries):
            print(f"  [{i+1}/{len(queries)}] ({qspec.intent}) \"{qspec.query}\"")

            raw_papers = _search_s2(
                query=qspec.query,
                year=self._year_range,
                limit=self._limit_per_query,
                api_key=self._s2_api_key,
            )
            stats["total_raw"] += len(raw_papers)

            whitelist_count = 0
            new_count = 0
            for paper in raw_papers:
                pid = paper.get("paperId")
                if not pid:
                    continue

                venue_canonical = match_venue(
                    paper.get("venue"),
                    paper.get("publicationVenue"),
                )
                if venue_canonical is None:
                    continue
                whitelist_count += 1

                if pid in all_papers:
                    continue
                new_count += 1

                # Extract author names
                authors = []
                for a in (paper.get("authors") or []):
                    name = a.get("name", "")
                    if name:
                        authors.append(name)

                venue_raw = ""
                pub_venue = paper.get("publicationVenue")
                if pub_venue and isinstance(pub_venue, dict):
                    venue_raw = pub_venue.get("name", "")
                if not venue_raw:
                    venue_raw = paper.get("venue", "")

                stub = PaperStub(
                    paper_id=pid,
                    title=paper.get("title", ""),
                    abstract=paper.get("abstract") or "",
                    venue_raw=venue_raw,
                    venue_norm=venue_canonical,
                    year=int(paper.get("year") or 0),
                    url=paper.get("url", ""),
                    authors=authors[:5],
                    citation_count=int(paper.get("citationCount") or 0),
                    source_query=qspec.query,
                    external_ids=paper.get("externalIds") or {},
                )
                all_papers[pid] = stub

            query_stat = {
                "query": qspec.query,
                "intent": qspec.intent,
                "raw_count": len(raw_papers),
                "whitelist_count": whitelist_count,
                "new_count": new_count,
            }
            stats["per_query"].append(query_stat)
            print(f"    → 返回 {len(raw_papers)} 篇, 白名单 {whitelist_count} 篇, 新增 {new_count} 篇")

            # Respect S2 rate limit
            if i < len(queries) - 1:
                time.sleep(self._request_interval_sec)

        papers_list = list(all_papers.values())
        stats["total_after_whitelist"] = sum(
            q["whitelist_count"] for q in stats["per_query"]
        )
        stats["total_after_dedup"] = len(papers_list)

        if self.logger:
            self.logger.log_event("path4_searcher_done", {
                "total_raw": stats["total_raw"],
                "total_after_whitelist": stats["total_after_whitelist"],
                "total_after_dedup": stats["total_after_dedup"],
                "query_count": len(queries),
            })

        print(f"\n  ✓ 检索完成: {stats['total_raw']} 篇原始 → "
              f"{stats['total_after_whitelist']} 篇白名单 → "
              f"{stats['total_after_dedup']} 篇去重后")

        venue_dist = {}
        for p in papers_list:
            venue_dist[p.venue_norm] = venue_dist.get(p.venue_norm, 0) + 1
        if venue_dist:
            print(f"  会议分布: {venue_dist}")

        return SearcherOutput(papers=papers_list, stats=stats)
