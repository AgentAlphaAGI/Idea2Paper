# 中文注释: LLM claim 降级 Agent（只做合法降级，不编造引用事实）。
from __future__ import annotations

import json
import re
from typing import Dict, List, Optional, Tuple

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from paper_kg.citations.claim_resolver import resolve_claims_in_sections
from paper_kg.citations.guards import contains_forbidden_citation_facts
from paper_kg.citations.models import CitationCandidate
from paper_kg.prompts import load_paper_prompt


_MARKER_RE = re.compile(r"<!--\s*CITE_CANDIDATE(?::[A-Za-z0-9:_-]+)?\s*-->")


class ClaimResolutionAgent:
    """LLM Claim 降级 Agent。"""

    def __init__(self, llm: Optional[BaseChatModel]) -> None:
        self.llm = llm
        self.prompt: Optional[ChatPromptTemplate] = None
        if llm is not None:
            system = load_paper_prompt("claim_resolution_system.txt")
            self.prompt = ChatPromptTemplate.from_messages(
                [
                    SystemMessage(content=system),
                    ("human", "候选列表：{items}"),
                ]
            )

    def resolve(
        self,
        sections: List[Tuple[str, str]],
        candidates: List[CitationCandidate],
        unresolved_ids: List[str],
    ) -> List[Tuple[str, str]]:
        if not unresolved_ids:
            return sections
        if self.llm is None or self.prompt is None:
            return resolve_claims_in_sections(sections, candidates, unresolved_ids)

        items = _build_resolution_items(sections, candidates, unresolved_ids)
        if not items:
            return resolve_claims_in_sections(sections, candidates, unresolved_ids)

        response = self.llm.invoke(self.prompt.format_messages(items=items))
        text = getattr(response, "content", "") or ""
        if contains_forbidden_citation_facts(text):
            return resolve_claims_in_sections(sections, candidates, unresolved_ids)

        try:
            data = json.loads(text)
        except Exception:
            return resolve_claims_in_sections(sections, candidates, unresolved_ids)

        if not isinstance(data, list):
            return resolve_claims_in_sections(sections, candidates, unresolved_ids)

        patch_map: Dict[str, str] = {}
        for item in data:
            if not isinstance(item, dict):
                continue
            candidate_id = str(item.get("candidate_id", "")).strip()
            new_paragraph = str(item.get("new_paragraph", "")).strip()
            if not candidate_id or not new_paragraph:
                continue
            if contains_forbidden_citation_facts(new_paragraph):
                continue
            if _MARKER_RE.search(new_paragraph):
                continue
            patch_map[candidate_id] = new_paragraph

        updated_sections = sections
        for candidate_id in unresolved_ids:
            replacement = patch_map.get(candidate_id)
            if replacement:
                updated_sections = _apply_patch(updated_sections, candidate_id, replacement)
            else:
                updated_sections = resolve_claims_in_sections(
                    updated_sections, candidates, [candidate_id]
                )
        return updated_sections


def _build_resolution_items(
    sections: List[Tuple[str, str]],
    candidates: List[CitationCandidate],
    unresolved_ids: List[str],
) -> List[Dict[str, str]]:
    cand_map: Dict[str, CitationCandidate] = {c.candidate_id: c for c in candidates}
    section_map: Dict[str, str] = {sid: text for sid, text in sections}
    items: List[Dict[str, str]] = []
    for cid in unresolved_ids:
        cand = cand_map.get(cid)
        if cand is None:
            continue
        markdown = section_map.get(cand.section_id, "")
        paragraph = _extract_paragraph(markdown, cid)
        if not paragraph:
            paragraph = cand.context_text or cand.claim_text
        items.append(
            {
                "candidate_id": cid,
                "section_id": cand.section_id,
                "claim_text": cand.claim_text,
                "context_text": cand.context_text,
                "paragraph": paragraph,
            }
        )
    return items


def _extract_paragraph(markdown: str, candidate_id: str) -> str:
    marker = f"<!-- CITE_CANDIDATE:{candidate_id} -->"
    pos = markdown.find(marker)
    if pos == -1:
        return ""
    start = markdown.rfind("\n\n", 0, pos)
    if start == -1:
        start = 0
    else:
        start += 2
    end = markdown.find("\n\n", pos)
    if end == -1:
        end = len(markdown)
    paragraph = markdown[start:end].strip()
    return _MARKER_RE.sub("", paragraph).strip()


def _apply_patch(
    sections: List[Tuple[str, str]],
    candidate_id: str,
    new_paragraph: str,
) -> List[Tuple[str, str]]:
    marker = f"<!-- CITE_CANDIDATE:{candidate_id} -->"
    updated: List[Tuple[str, str]] = []
    for section_id, markdown in sections:
        pos = markdown.find(marker)
        if pos == -1:
            updated.append((section_id, markdown))
            continue
        start = markdown.rfind("\n\n", 0, pos)
        if start == -1:
            start = 0
        else:
            start += 2
        end = markdown.find("\n\n", pos)
        if end == -1:
            end = len(markdown)
        clean_para = _MARKER_RE.sub("", new_paragraph).strip()
        new_text = markdown[:start] + clean_para + markdown[end:]
        updated.append((section_id, new_text))
    return updated
