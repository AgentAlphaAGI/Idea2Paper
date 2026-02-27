"""
Path4 Searcher — OpenAlex 检索 + 白名单硬过滤

职责:
  - 对每条 Planner query 调用 OpenAlex Works API
  - 白名单硬过滤: venue ∉ 白名单 → 直接丢弃
  - 跨 query 去重 (按 paper_id)
  - 输出 PaperStub 列表供后续 Extractor 使用
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

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
# Helpers
# ────────────────────────────────────────

def _compute_backoff_seconds(attempt: int, retry_after: Optional[str] = None) -> float:
    if retry_after:
        try:
            return max(1.0, float(retry_after))
        except Exception:
            pass
    return min(12.0, 2.0 * (2 ** max(0, attempt - 1)))


# ────────────────────────────────────────
# OpenAlex Works API
# ────────────────────────────────────────

_OA_SEARCH_URL = "https://api.openalex.org/works"
_OA_CS_CONCEPT_ID = "C121332964"  # Computer Science

_OA_SELECT_FIELDS = ",".join([
    "id",
    "display_name",
    "publication_year",
    "abstract_inverted_index",
    "primary_location",
    "locations",
    "authorships",
    "cited_by_count",
    "doi",
    "type",
    "ids",
])


def _reconstruct_abstract(inverted_index: Optional[Dict]) -> str:
    """Reconstruct plain text from OpenAlex abstract_inverted_index."""
    if not inverted_index or not isinstance(inverted_index, dict):
        return ""
    pairs: List[tuple] = []
    for word, positions in inverted_index.items():
        for pos in positions:
            pairs.append((pos, word))
    pairs.sort(key=lambda x: x[0])
    return " ".join(w for _, w in pairs)


def _extract_venue_from_openalex(paper: Dict) -> Tuple[str, List[str]]:
    """Extract (venue_raw, venue_candidates) from OpenAlex work object.

    Returns the best venue_raw string and a list of candidate strings
    for whitelist matching.
    """
    candidates = []
    venue_raw = ""

    primary = paper.get("primary_location") or {}
    source = primary.get("source") or {}
    if source.get("display_name"):
        candidates.append(source["display_name"])
        venue_raw = source["display_name"]
    if primary.get("raw_source_name"):
        candidates.append(primary["raw_source_name"])
        if not venue_raw:
            venue_raw = primary["raw_source_name"]

    for loc in (paper.get("locations") or []):
        src = loc.get("source") or {}
        if src.get("display_name"):
            candidates.append(src["display_name"])
        if loc.get("raw_source_name"):
            candidates.append(loc["raw_source_name"])

    return venue_raw, candidates


def _match_venue_openalex(paper: Dict) -> Optional[str]:
    """Try to match OpenAlex venue against whitelist using all location candidates."""
    _, candidates = _extract_venue_from_openalex(paper)
    for candidate in candidates:
        result = match_venue(candidate, None)
        if result is not None:
            return result
    return None


def _search_openalex(
    query: str,
    year: str = "2024-2026",
    limit: int = 50,
    page: int = 1,
    timeout: int = 30,
) -> List[Dict]:
    """Call OpenAlex works search endpoint.

    Returns raw work dicts or empty list on failure.
    """
    year_parts = year.split("-")
    filters = [f"concepts.id:{_OA_CS_CONCEPT_ID}"]
    if len(year_parts) == 2:
        filters.append(f"publication_year:{year_parts[0]}-{year_parts[1]}")
    elif len(year_parts) == 1:
        filters.append(f"publication_year:{year_parts[0]}")

    params = {
        "search": query,
        "select": _OA_SELECT_FIELDS,
        "per-page": min(limit, 200),
        "page": max(1, int(page)),
        "filter": ",".join(filters),
    }

    headers = {
        "User-Agent": "Idea2Paper/1.0 (mailto:research@example.com)",
    }

    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.get(
                _OA_SEARCH_URL,
                params=params,
                headers=headers,
                timeout=timeout,
            )

            if resp.status_code == 200:
                data = resp.json()
                return data.get("results", [])

            if resp.status_code == 429:
                wait_s = _compute_backoff_seconds(
                    attempt,
                    resp.headers.get("Retry-After"),
                )
                print(f"    ⚠️  OpenAlex rate limit (attempt {attempt}/{max_attempts}), wait {wait_s:.1f}s...")
                if attempt < max_attempts:
                    time.sleep(wait_s)
                    continue

            if 500 <= resp.status_code < 600:
                wait_s = _compute_backoff_seconds(attempt)
                print(f"    ⚠️  OpenAlex server error {resp.status_code} (attempt {attempt}/{max_attempts}), wait {wait_s:.1f}s...")
                if attempt < max_attempts:
                    time.sleep(wait_s)
                    continue

            print(f"    ❌ OpenAlex API HTTP error {resp.status_code} after {attempt} attempts")
            return []
        except requests.RequestException as e:
            wait_s = _compute_backoff_seconds(attempt)
            print(f"    ⚠️  OpenAlex request error (attempt {attempt}/{max_attempts}): {e}")
            if attempt < max_attempts:
                time.sleep(wait_s)
                continue
            print("    ❌ OpenAlex API request failed after retries")
            return []

    return []


def _openalex_paper_id(paper: Dict) -> str:
    """Extract a stable paper ID from OpenAlex work. Prefer OpenAlex ID."""
    oa_id = paper.get("id", "")
    if oa_id:
        return oa_id.replace("https://openalex.org/", "")
    return ""


class AgenticSearcher:
    """Search OpenAlex and apply venue whitelist hard filter."""

    def __init__(self, logger=None):
        self.logger = logger or get_logger()
        self._year_range = str(PipelineConfig.AGENTIC_SEARCH_YEAR_RANGE)
        self._limit_per_query = int(PipelineConfig.AGENTIC_SEARCH_LIMIT_PER_QUERY)
        self._min_whitelist_per_query = int(PipelineConfig.AGENTIC_SEARCH_MIN_WHITELIST_PER_QUERY)
        self._max_pages_per_query = int(PipelineConfig.AGENTIC_SEARCH_MAX_PAGES_PER_QUERY)
        # OpenAlex polite pool is permissive, keep a small interval for safety.
        self._request_interval_sec = 0.5

    def search(self, queries: List[QuerySpec]) -> SearcherOutput:
        """Execute search for each query and apply venue whitelist."""
        print(f"\n🔎 [Path4 Searcher (OpenAlex)] 执行 {len(queries)} 条 query 的联网检索...")

        all_papers: Dict[str, PaperStub] = {}
        stats = {
            "provider": "openalex",
            "total_raw": 0,
            "total_after_whitelist": 0,
            "total_after_dedup": 0,
            "per_query": [],
        }

        for i, qspec in enumerate(queries):
            print(f"  [{i+1}/{len(queries)}] ({qspec.intent}) \"{qspec.query}\"")

            # Fetch multiple pages until whitelist hits reach target, or page budget exhausted.
            raw_papers: List[Dict] = []
            pages_fetched = 0
            est_whitelist_hits = 0
            for page in range(1, self._max_pages_per_query + 1):
                page_papers = _search_openalex(
                    query=qspec.query,
                    year=self._year_range,
                    limit=self._limit_per_query,
                    page=page,
                )
                pages_fetched += 1
                raw_papers.extend(page_papers)

                # Estimate whitelist hits from this page to decide whether to continue paging.
                page_hits = 0
                for paper in page_papers:
                    if _match_venue_openalex(paper) is not None:
                        page_hits += 1
                est_whitelist_hits += page_hits

                # Stop paging if enough whitelist candidates already observed.
                if est_whitelist_hits >= self._min_whitelist_per_query:
                    break
                # Stop if this page is shorter than requested page size (likely no more results).
                if len(page_papers) < self._limit_per_query:
                    break

                print(
                    f"    ↻ 翻页 page={page + 1} "
                    f"(当前白名单估计 {est_whitelist_hits}/{self._min_whitelist_per_query})"
                )
                time.sleep(self._request_interval_sec)

            whitelist_count, new_count, venue_debug = self._process_openalex_results(
                raw_papers, qspec.query, all_papers,
            )

            stats["total_raw"] += len(raw_papers)

            query_stat = {
                "query": qspec.query,
                "intent": qspec.intent,
                "raw_count": len(raw_papers),
                "whitelist_count": whitelist_count,
                "new_count": new_count,
                "pages_fetched": pages_fetched,
                "estimated_whitelist_hits": est_whitelist_hits,
                "venue_debug": venue_debug,
            }
            stats["per_query"].append(query_stat)
            print(
                f"    → 返回 {len(raw_papers)} 篇 (pages={pages_fetched}), "
                f"白名单 {whitelist_count} 篇, 新增 {new_count} 篇"
            )
            self._print_query_venue_debug(venue_debug)

            if i < len(queries) - 1:
                time.sleep(self._request_interval_sec)

        papers_list = list(all_papers.values())
        stats["total_after_whitelist"] = sum(
            q["whitelist_count"] for q in stats["per_query"]
        )
        stats["total_after_dedup"] = len(papers_list)

        if self.logger:
            self.logger.log_event("agenticSearch_searcher_done", {
                "provider": "openalex",
                "total_raw": stats["total_raw"],
                "total_after_whitelist": stats["total_after_whitelist"],
                "total_after_dedup": stats["total_after_dedup"],
                "query_count": len(queries),
            })

        print(f"\n  ✓ 检索完成 (OpenAlex): {stats['total_raw']} 篇原始 → "
              f"{stats['total_after_whitelist']} 篇白名单 → "
              f"{stats['total_after_dedup']} 篇去重后")

        venue_dist: Dict[str, int] = {}
        for p in papers_list:
            venue_dist[p.venue_norm] = venue_dist.get(p.venue_norm, 0) + 1
        if venue_dist:
            print(f"  会议分布: {venue_dist}")

        return SearcherOutput(papers=papers_list, stats=stats)

    # ── OpenAlex result processing ──

    def _process_openalex_results(
        self,
        raw_papers: List[Dict],
        source_query: str,
        all_papers: Dict[str, PaperStub],
    ) -> tuple:
        whitelist_count = 0
        new_count = 0
        matched_counts: Dict[str, int] = {}
        unmatched_counts: Dict[str, int] = {}
        unmatched_examples: List[Dict[str, str]] = []

        for paper in raw_papers:
            pid = _openalex_paper_id(paper)
            if not pid:
                continue

            venue_raw, venue_candidates = _extract_venue_from_openalex(paper)
            venue_canonical = _match_venue_openalex(paper)
            if venue_canonical is None:
                debug_venue = ""
                for cand in venue_candidates:
                    if cand and cand.strip():
                        debug_venue = cand.strip()
                        break
                if not debug_venue:
                    debug_venue = venue_raw.strip() if venue_raw else "(empty_venue)"
                unmatched_counts[debug_venue] = unmatched_counts.get(debug_venue, 0) + 1
                if len(unmatched_examples) < 5:
                    unmatched_examples.append({
                        "paper_id": pid,
                        "title": (paper.get("display_name") or "")[:120],
                        "venue_candidate": debug_venue,
                    })
                continue
            whitelist_count += 1
            matched_counts[venue_canonical] = matched_counts.get(venue_canonical, 0) + 1

            if pid in all_papers:
                continue
            new_count += 1

            # Authors
            authors = []
            for authorship in (paper.get("authorships") or [])[:5]:
                author_obj = authorship.get("author") or {}
                name = author_obj.get("display_name", "")
                if name:
                    authors.append(name)

            # Abstract
            abstract = _reconstruct_abstract(paper.get("abstract_inverted_index"))

            # URL
            doi = paper.get("doi") or ""
            primary_loc = paper.get("primary_location") or {}
            url = primary_loc.get("landing_page_url") or doi or ""

            # External IDs
            ids_dict = paper.get("ids") or {}
            external_ids = {}
            if doi:
                external_ids["DOI"] = doi
            if ids_dict.get("openalex"):
                external_ids["OpenAlex"] = ids_dict["openalex"]

            stub = PaperStub(
                paper_id=pid,
                title=paper.get("display_name", ""),
                abstract=abstract,
                venue_raw=venue_raw,
                venue_norm=venue_canonical,
                year=int(paper.get("publication_year") or 0),
                url=url,
                authors=authors,
                citation_count=int(paper.get("cited_by_count") or 0),
                source_query=source_query,
                external_ids=external_ids,
            )
            all_papers[pid] = stub

        def _top_items(d: Dict[str, int], k: int = 8) -> List[Dict[str, int]]:
            items = sorted(d.items(), key=lambda x: x[1], reverse=True)[:k]
            return [{"name": name, "count": count} for name, count in items]

        venue_debug = {
            "matched_top": _top_items(matched_counts, 8),
            "unmatched_top": _top_items(unmatched_counts, 12),
            "unmatched_examples": unmatched_examples,
        }
        return whitelist_count, new_count, venue_debug

    def _print_query_venue_debug(self, venue_debug: Dict) -> None:
        """Print compact venue matching diagnostics for one query."""
        unmatched_top = venue_debug.get("unmatched_top", [])
        if unmatched_top:
            print("      未命中 venue top:")
            for item in unmatched_top[:5]:
                name = item.get("name", "")[:90]
                count = item.get("count", 0)
                print(f"        - {name}  ({count})")

