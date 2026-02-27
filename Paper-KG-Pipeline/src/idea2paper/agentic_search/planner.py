"""
Path4 Planner — 将用户 idea 转为结构化的英文检索 query

策略: LLM 生成候选 → 规则裁剪（多样性 + 去重 + 长度约束）

参考 Gemini / OpenAI Deep Research 的 planning 思路:
  先从 idea 中抽取 method / task / constraint 三类 anchor,
  再组合成 5-8 条兼顾「核心召回」与「探索性召回」的 query.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

from idea2paper.config import PipelineConfig
from idea2paper.infra.llm import call_llm, parse_json_from_llm
from idea2paper.infra.run_context import get_logger


# ────────────────────────────────────────
# Data models
# ────────────────────────────────────────

@dataclass
class QuerySpec:
    """One search query with metadata."""
    query: str
    intent: str  # core_method | task_setting | related_area | contrast | evaluation
    must_have: List[str] = field(default_factory=list)
    source: str = "planner"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PlannerOutput:
    """Full output of the Planner stage."""
    anchors: Dict[str, List[str]]
    candidates: List[QuerySpec]
    final_queries: List[QuerySpec]

    def to_dict(self) -> dict:
        return {
            "anchors": self.anchors,
            "candidates": [q.to_dict() for q in self.candidates],
            "final_queries": [q.to_dict() for q in self.final_queries],
        }


# ────────────────────────────────────────
# Planner prompt
# ────────────────────────────────────────

_PLANNER_SYSTEM = """\
You are an expert research assistant that helps generate academic search queries.
Your goal is to convert a research idea into a set of precise English search queries
suitable for academic paper search engines (e.g. OpenAlex).
"""

_PLANNER_TEMPLATE = """\
Below is a research idea (and optionally a structured brief).
Your task: generate **{max_queries} candidate search queries** for finding the most relevant recent CS papers.

## Instructions

1. **Extract anchors** from the idea:
   - **method**: core model / algorithm / training paradigm / module names
     (e.g. "retrieval-augmented generation", "diffusion policy", "speculative decoding")
   - **task**: task / input-output / scenario / data modality
     (e.g. "long-context reasoning", "video understanding", "3D reconstruction")
   - **constraint**: constraints / selling points / differentiators
     (e.g. "efficient", "robust", "low-resource", "privacy-preserving")

2. **Generate queries** by combining 2-4 anchors per query:
   - Each query must be plain English text (NO boolean operators, NO quotes, NO advanced syntax)
   - Each query should be 6-14 English words
   - Avoid overly broad terms as the only keyword (e.g. "deep learning" alone)
   - At least 2 queries must have intent "core_method"
   - At least 1 query must have intent "task_setting"
   - Include 1-2 queries with intent "contrast" that contain differentiating/opposing keywords
   - Each query must list 1-3 "must_have" anchor terms that should appear in relevant papers

3. **Diversity**: ensure queries are NOT near-duplicates. Each query should target a different angle.

## Research Idea
{user_idea}

{brief_section}

## Output Format (JSON only, no markdown)
{{
  "anchors": {{
    "method": ["anchor1", "anchor2", ...],
    "task": ["anchor1", "anchor2", ...],
    "constraint": ["anchor1", "anchor2", ...]
  }},
  "candidates": [
    {{
      "q": "search query text here",
      "intent": "core_method",
      "must_have": ["term1", "term2"]
    }},
    ...
  ]
}}
"""


def _build_brief_section(idea_brief: Optional[Dict]) -> str:
    if not idea_brief:
        return ""
    parts = ["## Structured Brief (from IdeaPackager)"]
    if idea_brief.get("keywords_en"):
        parts.append(f"Keywords: {', '.join(idea_brief['keywords_en'])}")
    if idea_brief.get("problem_definition"):
        parts.append(f"Problem: {idea_brief['problem_definition']}")
    if idea_brief.get("technical_plan"):
        parts.append(f"Technical Plan: {idea_brief['technical_plan']}")
    if idea_brief.get("constraints"):
        parts.append(f"Constraints: {', '.join(idea_brief['constraints'])}")
    if idea_brief.get("motivation"):
        parts.append(f"Motivation: {idea_brief['motivation']}")
    return "\n".join(parts)


# ────────────────────────────────────────
# Rule-based pruning
# ────────────────────────────────────────

def _token_set(text: str) -> set:
    return set(text.lower().split())


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _prune_candidates(
    candidates: List[QuerySpec],
    max_final: int,
) -> List[QuerySpec]:
    """Rule-based pruning: diversity + dedup + length + intent coverage."""

    # 1. Remove queries that are too short or too long
    valid = []
    for q in candidates:
        word_count = len(q.query.split())
        if 3 <= word_count <= 20:
            valid.append(q)
    if not valid:
        valid = candidates[:max_final]

    # 2. Deduplicate by Jaccard > 0.8 (keep shorter / more anchor-dense)
    deduped: List[QuerySpec] = []
    token_sets: List[set] = []
    for q in valid:
        ts = _token_set(q.query)
        is_dup = False
        for existing_ts in token_sets:
            if _jaccard(ts, existing_ts) > 0.8:
                is_dup = True
                break
        if not is_dup:
            deduped.append(q)
            token_sets.append(ts)

    # 3. Intent-aware selection: ensure mandatory intents are covered
    mandatory_intents = {"core_method", "task_setting"}
    selected: List[QuerySpec] = []
    remaining: List[QuerySpec] = []

    for q in deduped:
        if q.intent in mandatory_intents and any(
            q.intent == mi for mi in mandatory_intents
        ):
            selected.append(q)
            mandatory_intents.discard(q.intent)
        else:
            remaining.append(q)

    # Fill up to max_final with remaining candidates
    for q in remaining:
        if len(selected) >= max_final:
            break
        selected.append(q)

    return selected[:max_final]


def _build_user_idea_query(user_idea: str) -> str:
    """Build a direct query from raw user idea text."""
    if not user_idea:
        return ""
    q = " ".join(user_idea.strip().split())

    # Remove common prompt-like prefixes to reduce search noise.
    q = re.sub(
        r"^(idea|research\s*idea|project\s*idea|my\s*idea(\s+is)?)\s*[:\-]\s*",
        "",
        q,
        flags=re.IGNORECASE,
    )
    q = re.sub(r"^(idea|research\s*idea|my\s*idea(\s+is)?)\s+", "", q, flags=re.IGNORECASE)
    q = " ".join(q.strip().split())
    if not q:
        return ""

    # Keep query reasonably short for search relevance.
    words = q.split()
    if len(words) > 24:
        q = " ".join(words[:24])
    return q


# ────────────────────────────────────────
# AgenticPlanner
# ────────────────────────────────────────

class AgenticPlanner:
    """Generate search queries from a user research idea.

    Usage:
        planner = AgenticPlanner()
        output = planner.plan("Use graph neural networks for protein folding prediction")
        for q in output.final_queries:
            print(q.query, q.intent)
    """

    def __init__(self, logger=None):
        self.logger = logger or get_logger()
        self._max_queries = int(PipelineConfig.AGENTIC_SEARCH_MAX_QUERIES)
        self._final_queries = int(PipelineConfig.AGENTIC_SEARCH_FINAL_QUERIES)
        self._temperature = float(PipelineConfig.LLM_TEMPERATURE_AGENTIC_SEARCH_PLANNER)

    def plan(
        self,
        user_idea: str,
        idea_brief: Optional[Dict] = None,
    ) -> PlannerOutput:
        """Generate and prune search queries for the given idea.

        Args:
            user_idea: Raw research idea text (can be Chinese or English).
            idea_brief: Optional structured brief from IdeaPackager.parse_raw_idea().

        Returns:
            PlannerOutput with anchors, raw candidates, and pruned final queries.
        """
        print("\n📋 [Path4 Planner] 生成检索 query...")

        brief_section = _build_brief_section(idea_brief)
        prompt = _PLANNER_SYSTEM + "\n\n" + _PLANNER_TEMPLATE.format(
            max_queries=self._max_queries,
            user_idea=user_idea,
            brief_section=brief_section,
        )

        raw_text = call_llm(
            prompt,
            temperature=self._temperature,
            max_tokens=4096,
            timeout=120,
        )

        anchors, candidates = self._parse_response(raw_text)

        if not candidates:
            print("  ⚠️  LLM 未返回有效 query，使用 fallback")
            candidates = self._fallback_queries(user_idea, idea_brief)
            anchors = {"method": [], "task": [], "constraint": []}

        # Always include one direct query from the original user idea.
        direct_query = _build_user_idea_query(user_idea)
        if direct_query:
            exists = any(q.query.strip().lower() == direct_query.lower() for q in candidates)
            if not exists:
                candidates.append(QuerySpec(
                    query=direct_query,
                    intent="related_area",
                    must_have=[],
                    source="user_idea",
                ))

        final = _prune_candidates(candidates, self._final_queries)

        # Ensure the direct user-idea query is kept in final queries.
        if direct_query:
            has_direct = any(q.query.strip().lower() == direct_query.lower() for q in final)
            if not has_direct:
                direct_q = QuerySpec(
                    query=direct_query,
                    intent="related_area",
                    must_have=[],
                    source="user_idea",
                )
                if len(final) < self._final_queries:
                    final.append(direct_q)
                elif final:
                    final[-1] = direct_q

        output = PlannerOutput(
            anchors=anchors,
            candidates=candidates,
            final_queries=final,
        )

        if self.logger:
            self.logger.log_event("agenticSearch_planner_done", {
                "candidate_count": len(candidates),
                "final_count": len(final),
                "anchors": anchors,
                "final_queries": [q.to_dict() for q in final],
            })

        print(f"  ✓ 生成 {len(candidates)} 条候选 query → 裁剪为 {len(final)} 条")
        for i, q in enumerate(final, 1):
            print(f"    [{i}] ({q.intent}) {q.query}")

        return output

    def _parse_response(self, raw_text: str) -> tuple:
        """Parse LLM JSON response into anchors + candidate list."""
        parsed = parse_json_from_llm(raw_text)
        if not parsed:
            return {}, []

        anchors = parsed.get("anchors", {})
        if not isinstance(anchors, dict):
            anchors = {}
        for key in ("method", "task", "constraint"):
            if key not in anchors or not isinstance(anchors[key], list):
                anchors[key] = []

        raw_candidates = parsed.get("candidates", [])
        if not isinstance(raw_candidates, list):
            return anchors, []

        candidates = []
        for item in raw_candidates:
            if not isinstance(item, dict):
                continue
            q_text = (item.get("q") or item.get("query") or "").strip()
            if not q_text:
                continue

            intent = (item.get("intent") or "core_method").strip().lower()
            valid_intents = {"core_method", "task_setting", "related_area", "contrast", "evaluation"}
            if intent not in valid_intents:
                intent = "core_method"

            must_have = item.get("must_have", [])
            if not isinstance(must_have, list):
                must_have = []

            candidates.append(QuerySpec(
                query=q_text,
                intent=intent,
                must_have=[str(m) for m in must_have],
            ))

        return anchors, candidates

    def backtrack_queries(
        self,
        user_idea: str,
        original_anchors: Dict[str, List[str]],
        extra_count: int = 3,
    ) -> List[QuerySpec]:
        """Generate broader queries when initial search yields too few papers.

        Strategy: drop constraint anchors, use only method + task anchors,
        and generate wider queries via LLM.
        """
        print("\n🔄 [Path4 Planner] Backtrack: 生成更宽泛的 query...")

        method_anchors = original_anchors.get("method", [])
        task_anchors = original_anchors.get("task", [])

        if method_anchors or task_anchors:
            prompt = _PLANNER_SYSTEM + "\n\n" + (
                f"The initial search for the research idea below returned too few "
                f"papers from top CS venues. Generate {extra_count} BROADER search queries "
                f"that are more likely to find related papers.\n\n"
                f"## Research Idea\n{user_idea}\n\n"
                f"## Known Method Anchors\n{', '.join(method_anchors)}\n\n"
                f"## Known Task Anchors\n{', '.join(task_anchors)}\n\n"
                f"## Instructions\n"
                f"- Use ONLY method + task anchors (drop constraints / narrow qualifiers)\n"
                f"- Each query: 4-10 English words, plain text, NO advanced syntax\n"
                f"- Prefer broader parent-concept terms over niche terms\n"
                f"- Intent should be 'related_area' or 'task_setting'\n\n"
                f"## Output Format (JSON only, no markdown)\n"
                f'{{"candidates": [{{"q": "...", "intent": "related_area", "must_have": ["..."]}}]}}'
            )

            raw_text = call_llm(
                prompt,
                temperature=0.3,
                max_tokens=2048,
                timeout=120,
            )
            _, candidates = self._parse_response(raw_text)
            if candidates:
                final = _prune_candidates(candidates, extra_count)
                print(f"  ✓ Backtrack 生成 {len(final)} 条宽泛 query")
                for i, q in enumerate(final, 1):
                    q.source = "backtrack"
                    print(f"    [BT-{i}] ({q.intent}) {q.query}")
                return final

        # Rule-based fallback if LLM fails or no anchors
        return self._fallback_queries(user_idea, None, source="backtrack")

    def _fallback_queries(
        self,
        user_idea: str,
        idea_brief: Optional[Dict],
        source: str = "fallback",
    ) -> List[QuerySpec]:
        """Simple keyword-based fallback when LLM fails."""
        queries = []

        if idea_brief and idea_brief.get("keywords_en"):
            kws = idea_brief["keywords_en"]
            if len(kws) >= 2:
                queries.append(QuerySpec(
                    query=" ".join(kws[:4]),
                    intent="core_method",
                    must_have=kws[:2],
                    source=source,
                ))
                queries.append(QuerySpec(
                    query=" ".join(kws[1:5]) if len(kws) > 2 else " ".join(kws),
                    intent="task_setting",
                    must_have=kws[1:3],
                    source=source,
                ))
            if idea_brief.get("problem_definition"):
                words = idea_brief["problem_definition"].split()[:10]
                queries.append(QuerySpec(
                    query=" ".join(words),
                    intent="task_setting",
                    must_have=[],
                    source=source,
                ))

        if not queries:
            words = user_idea.split()[:12]
            queries.append(QuerySpec(
                query=" ".join(words),
                intent="core_method",
                must_have=[],
                source=source,
            ))

        return queries
