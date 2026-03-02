"""
Path4 Searcher — OpenAlex 检索 + 白名单硬过滤 + arXiv 补充召回

职责:
  - 对每条 Planner query 调用 OpenAlex Works API（白名单硬过滤）
  - 对每条 Planner query 调用 arXiv Atom API（初筛 + 轻量预筛）
  - 跨 query、跨来源去重 (按 paper_id / arXiv ID)
  - 输出 PaperStub 列表供后续 Extractor 使用
"""

from __future__ import annotations

import re
import requests
import time
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timezone
from idea2paper.agentic_search.planner import QuerySpec
from idea2paper.agentic_search.venue_whitelist import match_venue
from idea2paper.config import PipelineConfig
from idea2paper.infra.run_context import get_logger
from typing import Dict, List, Optional, Tuple
from xml.etree import ElementTree as ET


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


# ────────────────────────────────────────
# arXiv Atom API helpers
# ────────────────────────────────────────

_ARXIV_API_URL = "https://export.arxiv.org/api/query"

# arXiv Atom XML namespaces
_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
    "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
}

# 核心 CS 分类（权重 1.0），其余 CS 分类权重 0.7
_ARXIV_CORE_CATEGORIES = {"cs.LG", "cs.AI", "cs.CL", "cs.CV"}

_ARXIV_VERSION_RE = re.compile(r"v(\d+)$", re.IGNORECASE)


def _parse_arxiv_id(entry_id: str) -> Tuple[str, int]:
    """从 arXiv entry id URL 中提取 (arxiv_id_no_version, version_num).

    例: 'https://arxiv.org/abs/2401.12345v3' → ('2401.12345', 3)
    """
    # 去掉 URL 前缀
    raw = entry_id.strip().rstrip("/")
    raw = re.sub(r"^https?://arxiv\.org/abs/", "", raw)
    m = _ARXIV_VERSION_RE.search(raw)
    if m:
        version = int(m.group(1))
        arxiv_id = raw[: m.start()]
    else:
        version = 1
        arxiv_id = raw
    return arxiv_id, version


def _parse_arxiv_date(date_str: str) -> Optional[date]:
    """Parse arXiv <published> ISO 8601 string to date object."""
    try:
        dt = datetime.fromisoformat(date_str.rstrip("Z").replace("Z", "+00:00"))
        return dt.date()
    except Exception:
        pass
    # fallback: try just YYYY-MM-DD
    try:
        return date.fromisoformat(date_str[:10])
    except Exception:
        return None


def _arxiv_category_score(
    primary_cat: str,
    allowed_cats: List[str],
    filter_enable: bool,
) -> float:
    """计算分类得分（0.4 / 0.7 / 1.0）.

    filter_enable=False 时，允许任意分类，仍按核心/扩展/其他打分。
    """
    if primary_cat in _ARXIV_CORE_CATEGORIES:
        return 1.0
    if primary_cat in allowed_cats:
        return 0.7
    return 0.4


def _arxiv_prescreen_score(
    published: date,
    version: int,
    primary_cat: str,
    min_year: int,
    allowed_cats: List[str],
    filter_enable: bool,
) -> float:
    """轻量预筛总分（无 embedding）.

    recency_score:  0.0–1.0，以 min_year-01-01 为基准，最多计 2 年
    revision_score: v1=0, v2=0.33, v3=0.67, v4+=1.0
    category_score: 1.0 / 0.7 / 0.4
    """
    base_date = date(min_year, 1, 1)
    delta_days = (published - base_date).days
    recency = max(0.0, min(1.0, delta_days / 730.0))  # 730 ≈ 2 years

    revision = min(max(0, version - 1), 3) / 3.0

    cat_score = _arxiv_category_score(primary_cat, allowed_cats, filter_enable)

    return 0.6 * recency + 0.2 * revision + 0.2 * cat_score


def _extract_core_phrase(query: str, max_words: int = 3) -> str:
    """从 Planner query 中提取最核心的短语（前 max_words 个有效词）.

    arXiv 搜索策略（实测）：
    - ti/abs 字段支持精确短语匹配（加双引号）
    - 短语长度建议 2-3 词：4 词以上标题精确匹配命中极少；1 词太宽泛
    - 连字符词（如 chain-of-thought）算 1 个 token 处理

    例: "chain-of-thought reasoning large language models"
        → "chain-of-thought reasoning large"（3词）
        → ti:"chain-of-thought reasoning large" → 约 200-700 条命中
    """
    _STOP = {"a", "an", "the", "of", "in", "on", "for", "to", "and", "or",
             "is", "are", "with", "from", "by", "at", "as", "be", "using",
             "via", "based", "across", "towards", "toward"}
    raw = query.strip().replace('"', '').replace("'", "")
    # 连字符（chain-of-thought → chain of thought）在 arXiv 里按空格处理
    raw = raw.replace("-", " ")
    tokens = [t for t in raw.split() if t.lower() not in _STOP and len(t) >= 2]
    if not tokens:
        tokens = raw.split()
    # 取前 max_words 个词
    return " ".join(tokens[:max_words])


def _build_arxiv_query(
    query: str,
    allowed_cats: List[str],
    filter_enable: bool,
    use_abs_fallback: bool = False,
) -> str:
    """构建 arXiv search_query 参数字符串.

    arXiv 搜索语法（实测）：
    - `ti:"short phrase"`  — 标题精确短语匹配（2-4 词），精准度最高
    - `abs:"short phrase"` — 摘要精确短语匹配，召回更多
    - `ti_abs:word AND`    — 多词 AND 在 arXiv 中无效（返回 0）
    - 完整长句当短语：命中极少（arXiv 里很少有与 Planner query 完全一致的标题）

    策略：
    1. 默认 ti:"前4核心词"（精准 300-1000 条）
    2. fallback  abs:"前4核心词"（更宽松 1000-5000 条）
    """
    phrase = _extract_core_phrase(query, max_words=3)
    if use_abs_fallback:
        search_part = f'abs:"{phrase}"'
    else:
        search_part = f'ti:"{phrase}"'
    if filter_enable and allowed_cats:
        cat_part = " OR ".join(f"cat:{c}" for c in allowed_cats)
        return f"({cat_part}) AND ({search_part})"
    return search_part


def _call_arxiv_api(search_query: str, max_results: int, timeout: int) -> Optional[bytes]:
    """单次 arXiv API 调用，返回原始 XML bytes 或 None（失败时）。"""
    params = {
        "search_query": search_query,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
        "max_results": max_results,
    }
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.get(_ARXIV_API_URL, params=params, timeout=timeout)
            if resp.status_code == 200:
                return resp.content
            if resp.status_code == 429:
                wait_s = _compute_backoff_seconds(attempt)
                print(f"    ⚠️  arXiv rate limit (attempt {attempt}/{max_attempts}), wait {wait_s:.1f}s...")
                if attempt < max_attempts:
                    time.sleep(wait_s)
                    continue
            print(f"    ❌ arXiv API HTTP {resp.status_code}")
            return None
        except requests.RequestException as e:
            wait_s = _compute_backoff_seconds(attempt)
            print(f"    ⚠️  arXiv request error (attempt {attempt}/{max_attempts}): {e}")
            if attempt < max_attempts:
                time.sleep(wait_s)
                continue
            print("    ❌ arXiv API request failed after retries")
            return None
    return None


def _parse_arxiv_xml(content: bytes) -> List[Dict]:
    """解析 arXiv Atom XML，返回条目列表。"""
    try:
        root = ET.fromstring(content)
    except ET.ParseError as e:
        print(f"    ❌ arXiv XML parse error: {e}")
        return []

    entries = []
    for entry in root.findall("atom:entry", _NS):
        id_elem = entry.find("atom:id", _NS)
        if id_elem is None or not id_elem.text:
            continue
        arxiv_id, version = _parse_arxiv_id(id_elem.text)

        title_elem = entry.find("atom:title", _NS)
        title = (title_elem.text or "").strip().replace("\n", " ") if title_elem is not None else ""

        summary_elem = entry.find("atom:summary", _NS)
        abstract = (summary_elem.text or "").strip().replace("\n", " ") if summary_elem is not None else ""

        pub_elem = entry.find("atom:published", _NS)
        published = _parse_arxiv_date(pub_elem.text or "") if pub_elem is not None else None

        pcat_elem = entry.find("arxiv:primary_category", _NS)
        primary_cat = pcat_elem.get("term", "") if pcat_elem is not None else ""

        authors = []
        for auth_elem in entry.findall("atom:author", _NS):
            name_elem = auth_elem.find("atom:name", _NS)
            if name_elem is not None and name_elem.text:
                authors.append(name_elem.text.strip())

        url = ""
        for link_elem in entry.findall("atom:link", _NS):
            if link_elem.get("type") == "text/html":
                url = link_elem.get("href", "")
                break
        if not url:
            url = f"https://arxiv.org/abs/{arxiv_id}"

        entries.append({
            "arxiv_id": arxiv_id,
            "version": version,
            "title": title,
            "abstract": abstract,
            "published": published,
            "primary_category": primary_cat,
            "authors": authors,
            "url": url,
        })
    return entries


def _search_arxiv_raw(
    query: str,
    max_results: int = 25,
    allowed_cats: Optional[List[str]] = None,
    filter_enable: bool = True,
    timeout: int = 30,
) -> List[Dict]:
    """调用 arXiv Atom API，返回解析后的原始条目列表（未做时效/质量过滤）.

    搜索策略（自动 fallback）：
    1. 先用 ti:"phrase" 精确标题匹配（高精准）
    2. 若返回 < 5 条，fallback 到 abs:"phrase" 摘要匹配（更宽松）

    每个元素包含: arxiv_id, version, title, abstract,
                  published (date), primary_category, authors, url
    """
    if allowed_cats is None:
        allowed_cats = list(_ARXIV_CORE_CATEGORIES)

    # ── 策略 1: ti:"phrase" ──
    q_ti = _build_arxiv_query(query, allowed_cats, filter_enable, use_abs_fallback=False)
    content = _call_arxiv_api(q_ti, max_results, timeout)
    if content is None:
        return []
    entries = _parse_arxiv_xml(content)
    strategy = "ti"

    # ── 策略 2 fallback: abs:"phrase" ──
    if len(entries) < 5:
        print(f"    ↩ arXiv ti: 返回 {len(entries)} 条（<5），fallback 到 abs:...")
        time.sleep(3.0)  # 遵守 arXiv rate limit
        q_abs = _build_arxiv_query(query, allowed_cats, filter_enable, use_abs_fallback=True)
        content2 = _call_arxiv_api(q_abs, max_results, timeout)
        if content2:
            entries2 = _parse_arxiv_xml(content2)
            if len(entries2) > len(entries):
                entries = entries2
                strategy = "abs"

    print(f"    arXiv [{strategy}] 原始 {len(entries)} 条")
    return entries


def _filter_and_score_arxiv(
    entries: List[Dict],
    min_year: int,
    min_prescreen_score: float,
    allowed_cats: List[str],
    filter_enable: bool,
    top_n: int,
    seen_arxiv_ids: set,
) -> List[Dict]:
    """硬过滤 + 软预筛，返回排好序的 top-N 条目（附 prescreen_score）."""
    scored = []
    for e in entries:
        arxiv_id = e["arxiv_id"]
        published = e["published"]
        abstract = e["abstract"]
        primary_cat = e["primary_category"]

        # ── 硬过滤 ──
        if not arxiv_id:
            continue
        if arxiv_id in seen_arxiv_ids:
            continue
        if published is None or published.year < min_year:
            continue
        if len(abstract) < 100:
            continue
        if filter_enable and allowed_cats:
            if primary_cat not in allowed_cats:
                continue

        # ── 软预筛得分 ──
        score = _arxiv_prescreen_score(
            published=published,
            version=e["version"],
            primary_cat=primary_cat,
            min_year=min_year,
            allowed_cats=allowed_cats,
            filter_enable=filter_enable,
        )
        if score < min_prescreen_score:
            continue

        scored.append({**e, "prescreen_score": score})

    # 按 prescreen_score 降序排列，取 top-N
    scored.sort(key=lambda x: x["prescreen_score"], reverse=True)
    return scored[:top_n]


class AgenticSearcher:
    """Search OpenAlex (whitelist) + arXiv (preprint) and merge results."""

    def __init__(self, logger=None):
        self.logger = logger or get_logger()
        # ── OpenAlex 配置 ──
        self._year_range = str(PipelineConfig.AGENTIC_SEARCH_YEAR_RANGE)
        self._limit_per_query = int(PipelineConfig.AGENTIC_SEARCH_LIMIT_PER_QUERY)
        self._min_whitelist_per_query = int(PipelineConfig.AGENTIC_SEARCH_MIN_WHITELIST_PER_QUERY)
        self._max_pages_per_query = int(PipelineConfig.AGENTIC_SEARCH_MAX_PAGES_PER_QUERY)
        self._request_interval_sec = 0.5  # OpenAlex polite pool

        # ── arXiv 配置 ──
        self._arxiv_enable = bool(PipelineConfig.AGENTIC_SEARCH_ARXIV_ENABLE)
        self._arxiv_min_year = int(PipelineConfig.AGENTIC_SEARCH_ARXIV_MIN_YEAR)
        self._arxiv_max_results = int(PipelineConfig.AGENTIC_SEARCH_ARXIV_MAX_RESULTS_PER_QUERY)
        self._arxiv_top_n = int(PipelineConfig.AGENTIC_SEARCH_ARXIV_TOP_N_PER_QUERY)
        self._arxiv_interval_sec = float(PipelineConfig.AGENTIC_SEARCH_ARXIV_REQUEST_INTERVAL_SEC)
        self._arxiv_min_score = float(PipelineConfig.AGENTIC_SEARCH_ARXIV_MIN_PRESCREEN_SCORE)
        self._arxiv_cat_filter = bool(PipelineConfig.AGENTIC_SEARCH_ARXIV_CATEGORY_FILTER_ENABLE)
        self._arxiv_allowed_cats = list(PipelineConfig.AGENTIC_SEARCH_ARXIV_CS_CATEGORIES)

    def search(self, queries: List[QuerySpec]) -> SearcherOutput:
        """Execute OpenAlex + arXiv search for each query and merge results."""
        arxiv_tag = " + arXiv" if self._arxiv_enable else ""
        print(f"\n🔎 [Path4 Searcher (OpenAlex{arxiv_tag})] 执行 {len(queries)} 条 query 的联网检索...")

        # all_papers keyed by paper_id (openalex ID or arxiv ID)
        all_papers: Dict[str, PaperStub] = {}
        # seen arXiv IDs for cross-source dedup
        seen_arxiv_ids: set = set()

        stats = {
            "provider": "openalex+arxiv" if self._arxiv_enable else "openalex",
            "total_raw": 0,
            "total_after_whitelist": 0,
            "total_arxiv_raw": 0,
            "total_arxiv_after_filter": 0,
            "total_after_dedup": 0,
            "per_query": [],
        }

        for i, qspec in enumerate(queries):
            print(f"  [{i+1}/{len(queries)}] ({qspec.intent}) \"{qspec.query}\"")

            # ── OpenAlex 多页召回 ──
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

                page_hits = sum(
                    1 for p in page_papers if _match_venue_openalex(p) is not None
                )
                est_whitelist_hits += page_hits

                if est_whitelist_hits >= self._min_whitelist_per_query:
                    break
                if len(page_papers) < self._limit_per_query:
                    break

                print(
                    f"    ↻ 翻页 page={page + 1} "
                    f"(当前白名单估计 {est_whitelist_hits}/{self._min_whitelist_per_query})"
                )
                time.sleep(self._request_interval_sec)

            whitelist_count, new_oa_count, venue_debug = self._process_openalex_results(
                raw_papers, qspec.query, all_papers,
            )
            # 收集 OpenAlex 中已有 arXiv ID 的映射，用于跨源去重
            for p in all_papers.values():
                if "arxiv" in (p.external_ids or {}):
                    seen_arxiv_ids.add(p.external_ids["arxiv"])

            stats["total_raw"] += len(raw_papers)

            # ── arXiv 召回（可选） ──
            arxiv_raw_count = 0
            arxiv_after_count = 0
            if self._arxiv_enable:
                time.sleep(self._arxiv_interval_sec)
                arxiv_new = self._search_arxiv(
                    query=qspec.query,
                    all_papers=all_papers,
                    seen_arxiv_ids=seen_arxiv_ids,
                )
                arxiv_raw_count = arxiv_new.get("raw_count", 0)
                arxiv_after_count = arxiv_new.get("after_filter_count", 0)
                stats["total_arxiv_raw"] += arxiv_raw_count
                stats["total_arxiv_after_filter"] += arxiv_after_count

            query_stat = {
                "query": qspec.query,
                "intent": qspec.intent,
                "oa_raw_count": len(raw_papers),
                "oa_whitelist_count": whitelist_count,
                "oa_new_count": new_oa_count,
                "oa_pages_fetched": pages_fetched,
                "oa_estimated_whitelist_hits": est_whitelist_hits,
                "arxiv_raw_count": arxiv_raw_count,
                "arxiv_after_filter_count": arxiv_after_count,
                "venue_debug": venue_debug,
                # legacy key aliases for backward compat
                "raw_count": len(raw_papers),
                "whitelist_count": whitelist_count,
                "new_count": new_oa_count,
                "pages_fetched": pages_fetched,
                "estimated_whitelist_hits": est_whitelist_hits,
            }
            stats["per_query"].append(query_stat)
            arxiv_note = f", arXiv {arxiv_raw_count}→{arxiv_after_count}" if self._arxiv_enable else ""
            print(
                f"    → OA: {len(raw_papers)} 篇 (pages={pages_fetched}), "
                f"白名单 {whitelist_count} 篇, 新增 {new_oa_count} 篇"
                f"{arxiv_note}"
            )
            self._print_query_venue_debug(venue_debug)

            if i < len(queries) - 1:
                time.sleep(self._request_interval_sec)

        papers_list = list(all_papers.values())
        stats["total_after_whitelist"] = sum(
            q["oa_whitelist_count"] for q in stats["per_query"]
        )
        stats["total_after_dedup"] = len(papers_list)

        oa_count = sum(1 for p in papers_list if p.venue_norm != "arXiv")
        arxiv_count = sum(1 for p in papers_list if p.venue_norm == "arXiv")

        if self.logger:
            self.logger.log_event("agenticSearch_searcher_done", {
                "provider": stats["provider"],
                "total_raw": stats["total_raw"],
                "total_after_whitelist": stats["total_after_whitelist"],
                "total_arxiv_raw": stats["total_arxiv_raw"],
                "total_arxiv_after_filter": stats["total_arxiv_after_filter"],
                "total_after_dedup": stats["total_after_dedup"],
                "openalex_count": oa_count,
                "arxiv_count": arxiv_count,
                "query_count": len(queries),
            })

        print(f"\n  ✓ 检索完成: OA {stats['total_raw']} 原始 → {stats['total_after_whitelist']} 白名单 | "
              f"arXiv {stats['total_arxiv_raw']} 原始 → {stats['total_arxiv_after_filter']} 筛后")
        print(f"  合并去重: {len(papers_list)} 篇 (OpenAlex={oa_count}, arXiv={arxiv_count})")

        venue_dist: Dict[str, int] = {}
        for p in papers_list:
            venue_dist[p.venue_norm] = venue_dist.get(p.venue_norm, 0) + 1
        if venue_dist:
            print(f"  来源分布: {venue_dist}")

        return SearcherOutput(papers=papers_list, stats=stats)

    # ── arXiv result processing ──

    def _search_arxiv(
        self,
        query: str,
        all_papers: Dict[str, PaperStub],
        seen_arxiv_ids: set,
    ) -> Dict:
        """搜索 arXiv 并将结果追加到 all_papers。

        返回 {'raw_count': N, 'after_filter_count': M}。
        任何异常静默降级，返回零值。
        """
        try:
            raw_entries = _search_arxiv_raw(
                query=query,
                max_results=self._arxiv_max_results,
                allowed_cats=self._arxiv_allowed_cats,
                filter_enable=self._arxiv_cat_filter,
            )
        except Exception as e:
            print(f"    ⚠️  arXiv 通路异常，静默降级: {e}")
            return {"raw_count": 0, "after_filter_count": 0}

        raw_count = len(raw_entries)

        try:
            filtered = _filter_and_score_arxiv(
                entries=raw_entries,
                min_year=self._arxiv_min_year,
                min_prescreen_score=self._arxiv_min_score,
                allowed_cats=self._arxiv_allowed_cats,
                filter_enable=self._arxiv_cat_filter,
                top_n=self._arxiv_top_n,
                seen_arxiv_ids=seen_arxiv_ids,
            )
        except Exception as e:
            print(f"    ⚠️  arXiv 初筛异常，静默降级: {e}")
            return {"raw_count": raw_count, "after_filter_count": 0}

        for entry in filtered:
            arxiv_id = entry["arxiv_id"]
            # 跨源去重：如果 OpenAlex 已有同 arXiv ID，跳过
            if arxiv_id in seen_arxiv_ids:
                continue

            stub = PaperStub(
                paper_id=arxiv_id,
                title=entry["title"],
                abstract=entry["abstract"],
                venue_raw="arXiv preprint",
                venue_norm="arXiv",
                year=entry["published"].year if entry["published"] else self._arxiv_min_year,
                url=entry["url"],
                authors=entry["authors"][:5],
                citation_count=0,
                source_query=query,
                external_ids={
                    "arxiv": arxiv_id,
                    "arxiv_version": f"v{entry['version']}",
                },
            )
            all_papers[arxiv_id] = stub
            seen_arxiv_ids.add(arxiv_id)

        return {"raw_count": raw_count, "after_filter_count": len(filtered)}

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

