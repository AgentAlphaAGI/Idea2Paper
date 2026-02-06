import json
from typing import Dict, List, Optional, Tuple

from idea2paper.config import PipelineConfig
from idea2paper.infra.llm import call_llm, parse_json_from_llm


class IdeaPackager:
    """Pattern-guided Idea Packaging (runtime, no KG schema changes)."""

    def __init__(self, logger=None):
        self.logger = logger

    def parse_raw_idea(self, raw_idea: str) -> Tuple[Dict, str]:
        """Parse raw idea into structured IdeaBrief + retrieval query."""
        prompt = f"""
You are a research assistant. Convert the raw idea into a structured brief.
Return JSON ONLY.

Raw Idea:
{raw_idea}

Return JSON with this schema:
{{
  "motivation": "Why it matters (real-world or academic gap)",
  "problem_definition": "Task/object/input-output/setting (include adversarial/faithful/irrelevant if mentioned)",
  "constraints": ["latency", "small model", "privacy", "data scarcity", "robustness", "..."],
  "technical_plan": "Core method outline (modules, training objective, inference flow)",
  "expected_contributions": ["contrib1", "contrib2", "contrib3"],
  "evaluation_plan": "datasets / metrics / baselines / ablations / robustness settings",
  "keywords_en": ["keyword1", "keyword2", "..."],
  "keywords_zh": ["关键词1", "关键词2", "..."],
  "assumptions": {{
    "explicit": ["facts explicitly stated by user"],
    "inferred": ["reasonable inferences (mark as inferred)"]
  }}
}}
"""
        try:
            response = call_llm(
                prompt,
                temperature=PipelineConfig.LLM_TEMPERATURE_IDEA_PACKAGING_PARSE,
                max_tokens=900,
                timeout=120,
            )
            brief = parse_json_from_llm(response) or {}
        except Exception:
            brief = {}

        brief = self._normalize_brief(brief, raw_idea)
        query = self._build_retrieval_query(brief, force_en=getattr(PipelineConfig, "IDEA_PACKAGING_FORCE_EN_QUERY", True))
        return brief, query

    def build_pattern_evidence(
        self,
        pattern_id: str,
        pattern_info: Dict,
        papers_by_id: Dict,
        max_exemplar_papers: int = 8,
        max_abstract_chars: int = 400,
    ) -> Dict:
        """Build evidence pack from pattern summary + exemplar papers."""
        evidence = {
            "pattern_id": pattern_id,
            "name": pattern_info.get("name", ""),
            "size": int(pattern_info.get("size", 0) or 0),
            "domain": pattern_info.get("domain", ""),
            "sub_domains": pattern_info.get("sub_domains", []) or [],
            "summary": {},
            "llm_enhanced_summary": {},
            "exemplar_papers": [],
        }

        if pattern_info.get("summary"):
            summary = pattern_info.get("summary", {})
            evidence["summary"] = {
                "representative_ideas": (summary.get("representative_ideas") or [])[:5],
                "solution_approaches": (summary.get("solution_approaches") or [])[:5],
                "story": (summary.get("story") or [])[:3],
                "common_problems": (summary.get("common_problems") or [])[:5],
            }

        if pattern_info.get("llm_enhanced_summary"):
            llm_sum = pattern_info.get("llm_enhanced_summary", {})
            evidence["llm_enhanced_summary"] = {
                "representative_ideas": llm_sum.get("representative_ideas", ""),
                "common_problems": llm_sum.get("common_problems", ""),
                "solution_approaches": llm_sum.get("solution_approaches", ""),
                "story": llm_sum.get("story", ""),
            }

        exemplar_ids = (pattern_info.get("exemplar_paper_ids") or [])[:max_exemplar_papers]
        for pid in exemplar_ids:
            paper = papers_by_id.get(pid) or {}
            title = paper.get("title", "")
            abstract = paper.get("abstract", "")
            if abstract and max_abstract_chars:
                abstract = abstract[:max_abstract_chars]
            if title or abstract:
                evidence["exemplar_papers"].append({
                    "paper_id": pid,
                    "title": title,
                    "abstract": abstract,
                })

        return evidence

    def package_with_pattern(
        self,
        raw_idea: str,
        brief_a: Dict,
        evidence_pack: Dict,
    ) -> Tuple[Dict, str]:
        """Pattern-guided packaging into a candidate IdeaBrief + retrieval query."""
        prompt = f"""
You are a research assistant. Align the user's idea with the given Pattern evidence.
Return JSON ONLY.

[User Raw Idea]
{raw_idea}

[Parsed Idea Brief]
{json.dumps(brief_a, ensure_ascii=False)}

[Pattern Evidence Pack]
{json.dumps(evidence_pack, ensure_ascii=False)}

Task:
1) Produce a refined IdeaBrief that is faithful to the user's intent.
2) Use the pattern evidence to enrich motivation/problem/evaluation.
3) Clearly separate explicit vs inferred assumptions.
4) Provide strong English keywords for retrieval.

Return JSON with the same schema as IdeaBrief:
{{
  "motivation": "...",
  "problem_definition": "...",
  "constraints": ["..."],
  "technical_plan": "...",
  "expected_contributions": ["..."],
  "evaluation_plan": "...",
  "keywords_en": ["..."],
  "keywords_zh": ["..."],
  "assumptions": {{
    "explicit": ["..."],
    "inferred": ["..."]
  }}
}}
"""
        try:
            response = call_llm(
                prompt,
                temperature=PipelineConfig.LLM_TEMPERATURE_IDEA_PACKAGING_PATTERN_GUIDED,
                max_tokens=1200,
                timeout=180,
            )
            brief = parse_json_from_llm(response) or {}
        except Exception:
            brief = {}

        brief = self._normalize_brief(brief, raw_idea, fallback=brief_a)
        query = self._build_retrieval_query(brief, force_en=getattr(PipelineConfig, "IDEA_PACKAGING_FORCE_EN_QUERY", True))
        return brief, query

    def judge_best_candidate(self, raw_idea: str, candidates: List[Dict]) -> Tuple[int, Dict]:
        """LLM judge to pick the best candidate brief."""
        if not candidates:
            return 0, {"reason": "no_candidates"}

        prompt = f"""
You are an evaluator. Choose the best candidate that is faithful, complete, and actionable.
Return JSON ONLY.

[Raw Idea]
{raw_idea}

[Candidates]
{json.dumps(candidates, ensure_ascii=False)}

Select best_index and provide rationale. Criteria:
- Faithfulness to user intent
- Completeness (motivation/problem/solution/contribution/evaluation)
- Clarity and testability

Return JSON:
{{
  "best_index": 0,
  "rationale": "..."
}}
"""
        try:
            response = call_llm(
                prompt,
                temperature=PipelineConfig.LLM_TEMPERATURE_IDEA_PACKAGING_JUDGE,
                max_tokens=400,
                timeout=120,
            )
            result = parse_json_from_llm(response) or {}
        except Exception:
            result = {}

        best_index = int(result.get("best_index", 0)) if candidates else 0
        if best_index < 0 or best_index >= len(candidates):
            best_index = 0
        return best_index, result

    def build_prompt_context(self, raw_idea: str, idea_brief: Dict) -> str:
        """Build prompt context text for downstream modules."""
        if not idea_brief:
            return raw_idea

        ctx = "【User Requirements Brief】\n"
        ctx += f"Motivation: {idea_brief.get('motivation', '')}\n"
        ctx += f"Problem Definition: {idea_brief.get('problem_definition', '')}\n"
        ctx += f"Constraints: {', '.join(idea_brief.get('constraints', []) or [])}\n"
        ctx += f"Technical Plan: {idea_brief.get('technical_plan', '')}\n"
        ctx += "Expected Contributions:\n"
        for i, c in enumerate(idea_brief.get("expected_contributions", []) or [], 1):
            ctx += f"  {i}. {c}\n"
        ctx += f"Evaluation Plan: {idea_brief.get('evaluation_plan', '')}\n"
        kws = idea_brief.get("keywords_en", []) or []
        if kws:
            ctx += f"Keywords (EN): {', '.join(kws)}\n"
        return f"{raw_idea}\n\n{ctx}".strip()

    def _build_retrieval_query(self, brief: Dict, force_en: bool = True) -> str:
        if not brief:
            return ""
        parts = [
            brief.get("problem_definition", ""),
            "Constraints: " + ", ".join(brief.get("constraints", []) or []),
            brief.get("technical_plan", ""),
        ]
        keywords_en = brief.get("keywords_en", []) or []
        keywords_zh = brief.get("keywords_zh", []) or []
        if force_en and keywords_en:
            parts.insert(0, "Keywords: " + ", ".join(keywords_en))
        if keywords_zh:
            parts.append("关键词: " + "，".join(keywords_zh))
        return " ".join([p for p in parts if p]).strip()

    def _normalize_brief(self, brief: Dict, raw_idea: str, fallback: Optional[Dict] = None) -> Dict:
        """Ensure all required fields exist."""
        if not isinstance(brief, dict):
            brief = {}
        base = fallback or {}
        def _get(key, default):
            return brief.get(key) if brief.get(key) not in (None, "") else base.get(key, default)

        normalized = {
            "motivation": _get("motivation", raw_idea),
            "problem_definition": _get("problem_definition", raw_idea),
            "constraints": _get("constraints", []) or [],
            "technical_plan": _get("technical_plan", ""),
            "expected_contributions": _get("expected_contributions", []) or [],
            "evaluation_plan": _get("evaluation_plan", ""),
            "keywords_en": _get("keywords_en", []) or [],
            "keywords_zh": _get("keywords_zh", []) or [],
            "assumptions": _get("assumptions", {"explicit": [], "inferred": []}) or {"explicit": [], "inferred": []},
        }
        return normalized
