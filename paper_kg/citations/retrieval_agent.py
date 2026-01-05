# 中文注释: LLM 引用检索 Agent（只做 query + 选择 paperId，事实来自工具）。
from __future__ import annotations

import json
import logging
import re
import time
from typing import Dict, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from paper_kg.citations.bibtex_utils import build_bibtex, make_bibkey
from paper_kg.citations.guards import contains_forbidden_citation_facts
from paper_kg.citations.models import CitationCandidate, CitationNeed, RetrievedCitation
from paper_kg.citations.semantic_scholar_tool import PaperCandidate, MockSemanticScholarTool
from paper_kg.prompts import load_paper_prompt


logger = logging.getLogger(__name__)
_LLM_CALL_DELAY_SEC = 5.0


class CitationRetrievalAgent:
    """LLM + 离线工具的检索代理。"""

    def __init__(
        self,
        llm: Optional[BaseChatModel],
        tool: Optional[MockSemanticScholarTool] = None,
        require_verifiable: bool = True,
        max_candidates: int = 5,
    ) -> None:
        self.llm = llm
        self.tool = tool or MockSemanticScholarTool()
        self.require_verifiable = require_verifiable
        self.max_candidates = max_candidates
        self.query_prompt: Optional[ChatPromptTemplate] = None
        self.select_prompt: Optional[ChatPromptTemplate] = None
        if llm is not None:
            query_system = load_paper_prompt("citation_retrieval_query_system.txt")
            select_system = load_paper_prompt("citation_retrieval_select_system.txt")
            self.query_prompt = ChatPromptTemplate.from_messages(
                [SystemMessage(content=query_system), ("human", "候选：{candidate}\n需求：{need}")]
            )
            self.select_prompt = ChatPromptTemplate.from_messages(
                [SystemMessage(content=select_system), ("human", "候选列表：{choices}\n需求：{need}")]
            )

    def retrieve(
        self, needs: List[CitationNeed], candidates: List[CitationCandidate]
    ) -> List[RetrievedCitation]:
        candidate_map: Dict[str, CitationCandidate] = {c.candidate_id: c for c in candidates}
        results: List[RetrievedCitation] = []
        for need in needs:
            if need.need_type == "NONE":
                continue
            cand = candidate_map.get(need.candidate_id)
            if cand is None:
                continue
            results.append(self._retrieve_one(need, cand))
        return results

    def _retrieve_one(self, need: CitationNeed, cand: CitationCandidate) -> RetrievedCitation:
        if self.llm is None or self.query_prompt is None or self.select_prompt is None:
            return RetrievedCitation(candidate_id=need.candidate_id, status="NOT_FOUND")

        query = self._build_query(need, cand)
        candidates = self.tool.search(query, limit=self.max_candidates)
        candidates = _filter_by_year_range(candidates, need.year_range)
        if not candidates:
            return RetrievedCitation(candidate_id=need.candidate_id, status="NOT_FOUND")

        selection = self._select_candidate(need, cand, candidates)
        if selection is None:
            return RetrievedCitation(candidate_id=need.candidate_id, status="NOT_FOUND")

        details = self.tool.fetch_details(selection)
        if details is None:
            return RetrievedCitation(candidate_id=need.candidate_id, status="NOT_FOUND")

        external = details.external_ids if isinstance(details.external_ids, dict) else {}
        doi = external.get("DOI") if isinstance(external, dict) else None
        arxiv = external.get("ArXiv") if isinstance(external, dict) else None
        url = details.url

        evidence: List[str] = []
        if doi:
            evidence.append(f"doi:{doi}")
        if arxiv:
            evidence.append(f"arxiv:{arxiv}")
        if url:
            evidence.append(url)

        if self.require_verifiable and not evidence:
            return RetrievedCitation(candidate_id=need.candidate_id, status="NOT_FOUND")

        bibtex = details.citation_styles.get("bibtex") if isinstance(details.citation_styles, dict) else ""
        if not bibtex:
            bibkey = make_bibkey(doi=doi, arxiv=arxiv, title=details.title, year=details.year)
            bibtex = build_bibtex(
                bibkey=bibkey,
                title=details.title,
                authors=details.authors,
                venue=details.venue,
                year=details.year,
                doi=doi,
                url=url,
            )
        else:
            bibkey = make_bibkey(doi=doi, arxiv=arxiv, title=details.title, year=details.year)

        return RetrievedCitation(
            candidate_id=need.candidate_id,
            status="FOUND",
            paper_id=details.paper_id,
            bibtex=bibtex,
            bibkey=bibkey,
            doi=doi,
            arxiv=arxiv,
            url=url,
            title=details.title,
            authors=details.authors,
            year=details.year,
            venue=details.venue,
            evidence=evidence,
        )

    def _build_query(self, need: CitationNeed, cand: CitationCandidate) -> str:
        fallback = " ".join(need.keywords).strip() or cand.claim_text or cand.context_text
        time.sleep(_LLM_CALL_DELAY_SEC)
        response = self.llm.invoke(
            self.query_prompt.format_messages(
                candidate=cand.model_dump(),
                need=need.model_dump(),
            )
        )
        text = getattr(response, "content", "") or ""
        if contains_forbidden_citation_facts(text):
            return fallback
        data = _parse_json(text)
        query = str(data.get("query", "")).strip() if isinstance(data, dict) else ""
        if not query:
            return fallback
        return query

    def _select_candidate(
        self, need: CitationNeed, cand: CitationCandidate, candidates: List[PaperCandidate]
    ) -> Optional[str]:
        choices = [
            {
                "paper_id": c.paper_id,
                "title": c.title,
                "year": c.year,
                "venue": c.venue,
                "url": c.url,
            }
            for c in candidates
        ]
        time.sleep(_LLM_CALL_DELAY_SEC)
        response = self.llm.invoke(
            self.select_prompt.format_messages(
                choices=choices,
                need=need.model_dump(),
            )
        )
        text = getattr(response, "content", "") or ""
        if contains_forbidden_citation_facts(text):
            return None
        data = _parse_json(text)
        if not isinstance(data, dict):
            return None
        status = str(data.get("status", "")).strip().upper()
        if status == "NOT_FOUND":
            return None
        selected = str(data.get("selected_paperId") or data.get("selected_paper_id") or "").strip()
        if not selected:
            return None
        if selected not in {c.paper_id for c in candidates}:
            logger.warning("CITATION_RETRIEVE_SELECT_INVALID: candidate_id=%s", cand.candidate_id)
            return None
        return selected


def _parse_json(text: str) -> object:
    try:
        return json.loads(text)
    except Exception:
        return {}


def _filter_by_year_range(candidates: List[PaperCandidate], year_range: str) -> List[PaperCandidate]:
    match = re.match(r"^\\s*(\\d{4})\\s*-\\s*(\\d{4})\\s*$", year_range or "")
    if not match:
        return candidates
    year_min, year_max = int(match.group(1)), int(match.group(2))
    filtered = [
        c for c in candidates if c.year is None or (year_min <= c.year <= year_max)
    ]
    return filtered or candidates
