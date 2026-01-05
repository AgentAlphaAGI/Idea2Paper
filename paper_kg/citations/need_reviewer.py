# 中文注释: 引用必要性判断（规则版 + 可选 LLM）。
from __future__ import annotations

import re
from typing import Dict, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from paper_kg.citations.guards import contains_forbidden_citation_facts
from paper_kg.citations.models import CitationCandidate, CitationNeed
from paper_kg.prompts import load_paper_prompt


_HARD_HINTS = [
    "已有研究",
    "研究表明",
    "文献表明",
    "首次提出",
    "显著优于",
    "显著提升",
    "公认",
    "经典方法",
    "SOTA",
    "state-of-the-art",
]
_SOFT_HINTS = [
    "相关研究",
    "已有工作",
    "领域内",
    "广泛讨论",
    "综述",
    "survey",
    "代表性工作",
]
_NONE_HINTS = [
    "本文",
    "我们",
    "本研究",
    "本工作",
]


class CitationNeedReviewer:
    """
    功能：判断引用必要性（HARD/SOFT/NONE）。
    说明：默认使用规则版；如果提供 llm，会调用 LLM，但仍只输出分类与关键词。
    """

    def __init__(self, llm: Optional[BaseChatModel] = None) -> None:
        self.llm = llm
        self.prompt: Optional[ChatPromptTemplate] = None
        if llm is not None:
            system = load_paper_prompt("citation_need_review_system.txt")
            self.prompt = ChatPromptTemplate.from_messages(
                [
                    SystemMessage(content=system),
                    ("human", "候选列表：{candidates}"),
                ]
            )

    def review(self, candidates: List[CitationCandidate], year_range: str) -> List[CitationNeed]:
        if not candidates:
            return []
        if self.llm is None or self.prompt is None:
            return deterministic_need_review(candidates, year_range)

        # 中文注释：LLM 仅用于分类和关键词，不得生成引用事实。
        response = self.llm.invoke(self.prompt.format_messages(candidates=[c.model_dump() for c in candidates]))
        text = getattr(response, "content", "") or ""
        if contains_forbidden_citation_facts(text):
            return deterministic_need_review(candidates, year_range)

        try:
            import json

            data = json.loads(text)
        except Exception:
            return deterministic_need_review(candidates, year_range)

        return _merge_llm_needs_with_fallback(candidates, data, year_range)


def deterministic_need_review(
    candidates: List[CitationCandidate], year_range: str
) -> List[CitationNeed]:
    needs: List[CitationNeed] = []
    for cand in candidates:
        text = cand.claim_text or cand.context_text
        need_type = _classify_need(text)
        keywords = _extract_keywords(text)
        if cand.keywords_suggested:
            keywords = _merge_keywords(keywords, cand.keywords_suggested)
        needs.append(
            CitationNeed(
                candidate_id=cand.candidate_id,
                need_type=need_type,
                keywords=keywords,
                year_range=year_range,
            )
        )
    return needs


def _merge_llm_needs_with_fallback(
    candidates: List[CitationCandidate],
    data: object,
    year_range: str,
) -> List[CitationNeed]:
    fallback = deterministic_need_review(candidates, year_range)
    fallback_map: Dict[str, CitationNeed] = {n.candidate_id: n for n in fallback}
    if not isinstance(data, list):
        return fallback

    llm_map: Dict[str, CitationNeed] = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        candidate_id = str(item.get("candidate_id", "")).strip()
        if not candidate_id or candidate_id not in fallback_map:
            continue
        need_type = str(item.get("need_type", "")).strip().upper()
        if need_type not in {"HARD", "SOFT", "NONE"}:
            continue
        keywords = item.get("keywords", [])
        if not isinstance(keywords, list):
            keywords = []
        cleaned = _sanitize_keywords([str(k).strip() for k in keywords])
        yr = str(item.get("year_range", "")).strip() or year_range
        llm_map[candidate_id] = CitationNeed(
            candidate_id=candidate_id,
            need_type=need_type,  # type: ignore[arg-type]
            keywords=cleaned,
            year_range=yr,
        )

    merged: List[CitationNeed] = []
    for cand in candidates:
        merged.append(llm_map.get(cand.candidate_id, fallback_map[cand.candidate_id]))
    return merged


def _classify_need(text: str) -> str:
    if not text:
        return "SOFT"
    lower = text.lower()
    if any(hint in text for hint in _HARD_HINTS) or any(hint in lower for hint in _HARD_HINTS):
        return "HARD"
    if any(hint in text for hint in _SOFT_HINTS) or any(hint in lower for hint in _SOFT_HINTS):
        return "SOFT"
    if any(hint in text for hint in _NONE_HINTS):
        return "NONE"
    return "SOFT"


def _extract_keywords(text: str, max_keywords: int = 6) -> List[str]:
    tokens = re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]{2,}", text)
    seen = set()
    keywords: List[str] = []
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        if token in {"我们", "本文", "本研究", "该", "这些", "方法", "模型", "系统"}:
            continue
        keywords.append(token)
        if len(keywords) >= max_keywords:
            break
    return keywords


def _merge_keywords(a: List[str], b: List[str]) -> List[str]:
    seen = set()
    merged: List[str] = []
    for token in a + b:
        if token in seen:
            continue
        seen.add(token)
        merged.append(token)
    return merged


def _sanitize_keywords(keywords: List[str], max_keywords: int = 6) -> List[str]:
    cleaned: List[str] = []
    for token in keywords:
        if not token:
            continue
        if contains_forbidden_citation_facts(token):
            continue
        cleaned.append(token)
        if len(cleaned) >= max_keywords:
            break
    return cleaned
