"""
Path4 Extractor — PaperStub list → paper-level pattern 抽取

Prompt 对齐 extract_patterns_ICLR_en_local.py:
  ⟨Base Problem, Solution Pattern, Story⟩ + idea + domain + sub_domains
输出对齐 nodes_pattern.json 格式

支持批量处理（batch_size 可配置），批量 JSON 解析失败时逐篇重试。
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

from idea2paper.config import PipelineConfig
from idea2paper.infra.llm import call_llm, clean_json_text
from idea2paper.infra.run_context import get_logger
from idea2paper.agentic_search.searcher import PaperStub


# ────────────────────────────────────────
# Data models
# ────────────────────────────────────────

@dataclass
class PaperPattern:
    """Paper-level pattern aligned with nodes_pattern.json."""
    pattern_id: str
    source: str = "agentic_search"
    paper_id: str = ""
    title: str = ""
    venue_norm: str = ""
    year: int = 0
    url: str = ""
    name: str = ""
    size: int = 1
    domain: str = ""
    sub_domains: List[str] = field(default_factory=list)
    summary: Dict = field(default_factory=dict)
    exemplar_count: int = 1
    exemplar_paper_ids: List[str] = field(default_factory=list)
    citation_count: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    def to_nodes_pattern_format(self) -> dict:
        """Format compatible with nodes_pattern.json for downstream consumption."""
        return {
            "pattern_id": self.pattern_id,
            "source": self.source,
            "name": self.name,
            "size": self.size,
            "domain": self.domain,
            "sub_domains": self.sub_domains,
            "summary": self.summary,
            "exemplar_count": self.exemplar_count,
            "exemplar_paper_ids": self.exemplar_paper_ids,
            "paper_id": self.paper_id,
            "title": self.title,
            "venue_norm": self.venue_norm,
            "year": self.year,
            "url": self.url,
            "citation_count": self.citation_count,
        }


@dataclass
class ExtractorOutput:
    """Full output of the Extractor stage."""
    patterns: List[PaperPattern]
    stats: Dict

    def to_dict(self) -> dict:
        return {
            "patterns": [p.to_dict() for p in self.patterns],
            "stats": self.stats,
        }


# ────────────────────────────────────────
# Prompt template — aligned with extract_patterns_ICLR_en_local.py
# ────────────────────────────────────────

# Shared core: Role + Task Objective + Key Definitions + Few-shot Examples
# Uses raw string to avoid escape issues.
_PROMPT_CORE = r"""【Role】
You are a "Top-Conference Research Pattern Extractor".
Your task is NOT to summarize the paper, but to extract the reusable research narration framework that turns ordinary techniques into a convincing top-tier contribution through framing, conceptual packaging, and evidence design.

【Task Objective】
Given a paper's title, keywords, and abstract, extract the paper's Research Pattern core framework:
⟨Base Problem, Solution Pattern, Story⟩,
and identify the paper's core idea.

This extraction is used to learn how top papers package methods into high-impact research narratives. The Story field is the most important.

【Key Definitions】
1) Research Pattern = ⟨Base Problem, Solution Pattern, Story⟩
- Base Problem:
  The concrete, actionable pain point in a specific scenario that the paper targets.
  Must be specific and grounded. Avoid vague phrasing such as "improves performance" without context.
- Solution Pattern:
  The core technical route that solves the Base Problem.
  Explicitly describe key components, pipeline structure, and the mechanism of improvement.
- Story (MOST IMPORTANT):
  The conceptual packaging and narrative framing that makes the work look novel, forward-looking, and impactful.
  Focus on narrative devices such as reframing the problem, introducing a new lens, elevating engineering issues into research questions, and highlighting long-term implications.

2) idea (Paper-level, OPTIONAL)
- A concise 1–2 sentence description of the paper's key innovation or central insight.
- This should capture what is fundamentally new or different at a high level, without technical detail.
- If the core idea cannot be clearly stated, omit this field.

3) Taxonomy Fields (for retrieval)
- domain (string):
  One primary top-level field (Level-1). Must be a broad discipline label.
  Examples: "Machine Learning", "Computer Vision", "Natural Language Processing", "Systems", "Security & Privacy", "Fairness & Accountability".
- sub_domains (array of strings):
  Level-2 tags under the chosen domain. 2–5 items preferred.
  These should be specific and retrieval-friendly.
  Rules:
  - sub_domains must be consistent with the chosen domain
  - avoid repeating the domain name itself

4) application (string)
Concrete deployable scenarios implied by the paper.
"""

_PROMPT_FEWSHOT = r"""
========================
【Few-shot Example 1】

【Input】
- paper_title: Research on Self-Evolution of Intelligent Agents Based on Reflect+Memory
- keywords: Intelligent Agent, Self-Evolution, Memory Mechanism
- abstract: Existing agents fail to retain experience after task execution, repeatedly making the same mistakes in similar tasks. This work introduces a Reflect module and a long-term Memory module to store, summarize, and retrieve experience for improved task execution over time.

【Output】
{
  "paper_id": "ARR_2022_106",
  "paper_title": "Research on Self-Evolution of Intelligent Agents Based on Reflect+Memory",
  "idea": "Enable intelligent agents to continuously improve by accumulating, summarizing, and reusing task experience through a reflection-plus-memory architecture",
  "domain": "Artificial Intelligence",
  "sub_domains": ["Agentic Systems", "Memory-Augmented Models", "Reflection", "Experience Retrieval"],
  "research_patterns": [
    {
      "base_problem": "Task-executing agents do not accumulate reusable experience, causing repeated failures and stagnant capability when facing recurring task families",
      "solution_pattern": "Augment the agent architecture with a reflection module that converts trajectories into distilled lessons, store them in long-term memory, and retrieve relevant lessons during inference to guide future decisions",
      "story": "Reframe agents from one-shot executors into a self-evolving paradigm where accumulated experience becomes a scalable capability multiplier, enabling sustained improvement and long-horizon autonomy",
      "application": "Customer-support automation with iterative playbook refinement, autonomous operations incident handling, robotics skill transfer across tasks, long-horizon decision-making assistants"
    }
  ]
}

========================
【Few-shot Example 2】

【Input】
- paper_title: Quantifying and Mitigating the Impact of Label Errors on Model Disparity Metrics
- keywords: Fairness, Label Noise, Influence Function, Disparity Metrics
- abstract: Existing fairness evaluation methods assume reliable labels, but real-world data often contains label errors that disproportionately affect different groups. This work analyzes label noise impact on disparity metrics and proposes an influence-function-based method to identify and correct high-impact label errors.

【Output】
{
  "paper_id": "ICLR_2023_089",
  "paper_title": "Quantifying and Mitigating the Impact of Label Errors on Model Disparity Metrics",
  "idea": "Diagnose and mitigate fairness failures by analyzing how individual label errors influence group-level disparity metrics",
  "domain": "Fairness & Accountability",
  "sub_domains": ["Label Noise", "Influence Functions", "Disparity Metrics", "Model Auditing"],
  "research_patterns": [
    {
      "base_problem": "Fairness evaluation becomes unreliable in the presence of label noise, systematically distorting disparity metrics and disproportionately harming minority groups",
      "solution_pattern": "Extend influence functions from loss-based analysis to group disparity metrics in order to quantify the effect of individual label perturbations and prioritize high-impact label corrections",
      "story": "Reframe fairness from a model optimization problem into an auditing and reliability problem, introducing a principled framework for diagnosing and correcting data-induced fairness failures",
      "application": "Fairness auditing in high-stakes decision systems, robustness evaluation under noisy labels, automated data quality inspection pipelines"
    }
  ]
}
"""

# ── Single-paper prompt ──
# Maximum alignment with extract_patterns_ICLR_en_local.py: same structure, same field order.
_SINGLE_PROMPT_TEMPLATE = (
    _PROMPT_CORE
    + r"""
【Hard Output Constraints】
- Output STRICT JSON only. No extra text, no Markdown, no comments.
- All values must be in English, concise, academic.
- paper_id must be: __PAPER_ID__
- paper_title must be the input title; if missing use "N/A"
- idea is OPTIONAL
- domain must be a non-empty string
- sub_domains must be a non-empty array (at least 1 item)
- research_patterns must be a non-empty array (at least 1 object)
- No field inside research_patterns is allowed to be empty.

【Output Schema】
{
  "paper_id": "__PAPER_ID__",
  "paper_title": "paper title",
  "idea": "concise 1–2 sentence description of the paper's key innovation or core insight",
  "domain": "Level-1 domain",
  "sub_domains": ["Level-2 tag 1", "Level-2 tag 2"],
  "research_patterns": [
    {
      "base_problem": "concrete pain point in a specific scenario",
      "solution_pattern": "core technical route (components + workflow + mechanism)",
      "story": "conceptual packaging that makes the work look novel and high-impact",
      "application": "deployable scenarios"
    }
  ]
}
"""
    + _PROMPT_FEWSHOT
    + r"""
========================
【Now Process This Paper】
- paper_title: __PAPER_TITLE__
- keywords: __KEYWORDS__
- abstract: __ABSTRACT__

Return STRICT JSON only.
"""
)

# ── Batch prompt (3–5 papers per call) ──
_BATCH_PROMPT_TEMPLATE = (
    _PROMPT_CORE
    + r"""
【Hard Output Constraints】
- Output a STRICT JSON **array** only. No extra text, no Markdown, no comments.
- The array must contain exactly __PAPER_COUNT__ objects, one per paper, in the same input order.
- All values must be in English, concise, academic.
- Each object's paper_id must match the corresponding input paper_id exactly.
- paper_title must be the input title; if missing use "N/A"
- idea is OPTIONAL
- domain must be a non-empty string
- sub_domains must be a non-empty array (at least 1 item)
- research_patterns must be a non-empty array (at least 1 object)
- No field inside research_patterns is allowed to be empty.

【Output Schema (each element)】
{
  "paper_id": "same as input",
  "paper_title": "paper title",
  "idea": "concise 1–2 sentence description of the paper's key innovation or core insight",
  "domain": "Level-1 domain",
  "sub_domains": ["Level-2 tag 1", "Level-2 tag 2"],
  "research_patterns": [
    {
      "base_problem": "concrete pain point in a specific scenario",
      "solution_pattern": "core technical route (components + workflow + mechanism)",
      "story": "conceptual packaging that makes the work look novel and high-impact",
      "application": "deployable scenarios"
    }
  ]
}
"""
    + _PROMPT_FEWSHOT
    + r"""
========================
【Now Process These Papers】
__PAPERS_SECTION__

Return STRICT JSON array only.
"""
)


# ────────────────────────────────────────
# Prompt builders
# ────────────────────────────────────────

def _build_single_prompt(paper_id: str, title: str, keywords: str, abstract: str) -> str:
    return (
        _SINGLE_PROMPT_TEMPLATE
        .replace("__PAPER_ID__", paper_id)
        .replace("__PAPER_TITLE__", title)
        .replace("__KEYWORDS__", keywords or "N/A")
        .replace("__ABSTRACT__", abstract)
    )


def _build_batch_prompt(papers: List[Dict[str, str]]) -> str:
    lines = []
    for i, p in enumerate(papers, 1):
        lines.append(
            f"[{i}] paper_id: {p['paper_id']}\n"
            f"    paper_title: {p['title']}\n"
            f"    keywords: {p.get('keywords') or 'N/A'}\n"
            f"    abstract: {p['abstract']}\n"
        )
    papers_section = "\n".join(lines)
    return (
        _BATCH_PROMPT_TEMPLATE
        .replace("__PAPER_COUNT__", str(len(papers)))
        .replace("__PAPERS_SECTION__", papers_section)
    )


# ────────────────────────────────────────
# JSON parsing helpers
# ────────────────────────────────────────

def _parse_json_response(text: str):
    """Parse LLM response into a dict or list, tolerating markdown fences."""
    cleaned = clean_json_text(text)

    # Direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Try JSON array
    arr_start = cleaned.find("[")
    arr_end = cleaned.rfind("]")
    if arr_start != -1 and arr_end > arr_start:
        try:
            return json.loads(cleaned[arr_start : arr_end + 1])
        except json.JSONDecodeError:
            pass

    # Try JSON object
    obj_start = cleaned.find("{")
    obj_end = cleaned.rfind("}")
    if obj_start != -1 and obj_end > obj_start:
        try:
            return json.loads(cleaned[obj_start : obj_end + 1])
        except json.JSONDecodeError:
            pass

    # Last resort: fix trailing commas
    repaired = re.sub(r",(\s*[}\]])", r"\1", cleaned)
    try:
        return json.loads(repaired)
    except Exception:
        pass

    return None


# ────────────────────────────────────────
# Result → PaperPattern conversion
# ────────────────────────────────────────

def _make_short_name(idea: str, title: str, max_words: int = 8) -> str:
    """Derive a ≤max_words narrative name from idea or title."""
    source = idea if idea else title
    words = source.split()
    if len(words) > max_words:
        return " ".join(words[:max_words])
    return source


def _llm_result_to_pattern(paper: PaperStub, extracted: dict) -> PaperPattern:
    """Map one LLM extraction dict + PaperStub → PaperPattern."""
    patterns_list = extracted.get("research_patterns", [])
    rp = patterns_list[0] if patterns_list else {}

    idea = extracted.get("idea", "")
    base_problem = rp.get("base_problem", "")
    solution_pattern = rp.get("solution_pattern", "")
    story = rp.get("story", "")
    application = rp.get("application", "")

    # nodes_pattern.json uses list-based summary
    summary = {
        "representative_ideas": [idea] if idea else [],
        "common_problems": [base_problem] if base_problem else [],
        "solution_approaches": [solution_pattern] if solution_pattern else [],
        "story": [story] if story else [],
        "application": [application] if application else [],
    }

    name = _make_short_name(idea, paper.title)
    domain = extracted.get("domain", "")
    sub_domains = extracted.get("sub_domains", [])
    if isinstance(sub_domains, str):
        sub_domains = [sub_domains]

    pid_short = paper.paper_id[:12] if len(paper.paper_id) > 12 else paper.paper_id

    return PaperPattern(
        pattern_id=f"p4_{pid_short}",
        source="agentic_search",
        paper_id=paper.paper_id,
        title=paper.title,
        venue_norm=paper.venue_norm,
        year=paper.year,
        url=paper.url,
        name=name,
        size=1,
        domain=domain,
        sub_domains=sub_domains,
        summary=summary,
        exemplar_count=1,
        exemplar_paper_ids=[paper.paper_id],
        citation_count=getattr(paper, "citation_count", 0) or 0,
    )


# ────────────────────────────────────────
# AgenticExtractor
# ────────────────────────────────────────

class AgenticExtractor:
    """Extract paper-level patterns from PaperStub list.

    Prompt aligned with extract_patterns_ICLR_en_local.py.
    Supports batch processing with single-paper fallback on JSON parse failure.

    Usage:
        extractor = AgenticExtractor()
        output = extractor.extract(searcher_output.papers)
        for pattern in output.patterns:
            print(pattern.name, pattern.domain)
    """

    def __init__(self, logger=None):
        self.logger = logger or get_logger()
        self._batch_size = int(PipelineConfig.AGENTIC_SEARCH_EXTRACTOR_BATCH_SIZE)
        self._temperature = float(PipelineConfig.LLM_TEMPERATURE_AGENTIC_SEARCH_EXTRACTOR)
        self._max_retries = 3

    # ── public API ──

    def extract(self, papers: List[PaperStub]) -> ExtractorOutput:
        """Extract patterns from all papers.

        Args:
            papers: PaperStub list from Searcher.

        Returns:
            ExtractorOutput with extracted patterns and stats.
        """
        print(f"\n🔬 [Path4 Extractor] 从 {len(papers)} 篇论文抽取 pattern...")

        valid_papers: List[PaperStub] = []
        skipped_no_abstract = 0
        for p in papers:
            if p.abstract and p.abstract.strip():
                valid_papers.append(p)
            else:
                skipped_no_abstract += 1
                print(f"  ⏭️  跳过 (无 abstract): {p.title[:60]}")

        patterns: List[PaperPattern] = []
        skipped_llm_fail = 0

        total_batches = max(1, (len(valid_papers) + self._batch_size - 1) // self._batch_size)
        for batch_start in range(0, len(valid_papers), self._batch_size):
            batch = valid_papers[batch_start : batch_start + self._batch_size]
            batch_idx = batch_start // self._batch_size + 1
            print(f"  📦 Batch {batch_idx}/{total_batches} ({len(batch)} 篇)...")

            batch_patterns = self._extract_batch(batch)

            if batch_patterns is None:
                # Batch failed entirely → fall back to single-paper extraction
                print(f"    ⚠️  批量抽取失败, 逐篇重试...")
                for paper in batch:
                    pattern = self._extract_single(paper)
                    if pattern:
                        patterns.append(pattern)
                        print(f"    ✓ {paper.title[:50]}")
                    else:
                        skipped_llm_fail += 1
                        print(f"    ❌ {paper.title[:50]}")
            else:
                patterns.extend(batch_patterns)
                failed_in_batch = len(batch) - len(batch_patterns)
                if failed_in_batch > 0:
                    skipped_llm_fail += failed_in_batch
                for pp in batch_patterns:
                    print(f"    ✓ [{pp.venue_norm}] {pp.title[:50]}")

        stats = {
            "total_input": len(papers),
            "valid_with_abstract": len(valid_papers),
            "skipped_no_abstract": skipped_no_abstract,
            "skipped_llm_fail": skipped_llm_fail,
            "patterns_extracted": len(patterns),
        }

        if self.logger:
            self.logger.log_event("agenticSearch_extractor_done", stats)

        print(f"\n  ✓ 抽取完成: {len(patterns)}/{len(papers)} 篇成功")
        if skipped_no_abstract:
            print(f"    跳过 (无 abstract): {skipped_no_abstract}")
        if skipped_llm_fail:
            print(f"    跳过 (LLM 失败): {skipped_llm_fail}")

        return ExtractorOutput(patterns=patterns, stats=stats)

    # ── batch extraction ──

    def _extract_batch(self, batch: List[PaperStub]) -> Optional[List[PaperPattern]]:
        """Extract patterns for a batch. Returns None on unrecoverable failure."""
        if len(batch) == 1:
            result = self._extract_single(batch[0])
            return [result] if result else None

        papers_data = [
            {
                "paper_id": p.paper_id,
                "title": p.title,
                "keywords": "N/A",
                "abstract": p.abstract,
            }
            for p in batch
        ]
        prompt = _build_batch_prompt(papers_data)

        for attempt in range(self._max_retries):
            raw_text = call_llm(
                prompt,
                temperature=self._temperature,
                max_tokens=8192,
                timeout=180,
            )

            if not raw_text or raw_text.startswith("[模拟LLM输出]"):
                if attempt < self._max_retries - 1:
                    time.sleep(2.0 * (attempt + 1))
                    continue
                return None

            parsed = _parse_json_response(raw_text)
            if parsed is None:
                if attempt < self._max_retries - 1:
                    time.sleep(2.0 * (attempt + 1))
                    continue
                return None

            results = parsed if isinstance(parsed, list) else [parsed]

            paper_by_id = {p.paper_id: p for p in batch}
            patterns: List[PaperPattern] = []

            for idx, item in enumerate(results):
                if not isinstance(item, dict):
                    continue
                pid = item.get("paper_id", "")
                paper = paper_by_id.get(pid)
                if paper is None and idx < len(batch):
                    paper = batch[idx]
                if paper is None:
                    continue
                try:
                    patterns.append(_llm_result_to_pattern(paper, item))
                except Exception as e:
                    print(f"    ⚠️  转换 pattern 失败 ({(paper.title or '')[:40]}): {e}")

            if patterns:
                return patterns

            if attempt < self._max_retries - 1:
                time.sleep(2.0 * (attempt + 1))
                continue

        return None

    # ── single-paper fallback ──

    def _extract_single(self, paper: PaperStub) -> Optional[PaperPattern]:
        """Extract pattern from one paper. Used as fallback when batch fails."""
        prompt = _build_single_prompt(
            paper_id=paper.paper_id,
            title=paper.title,
            keywords="N/A",
            abstract=paper.abstract,
        )

        for attempt in range(self._max_retries):
            raw_text = call_llm(
                prompt,
                temperature=self._temperature,
                max_tokens=4096,
                timeout=120,
            )

            if not raw_text or raw_text.startswith("[模拟LLM输出]"):
                if attempt < self._max_retries - 1:
                    time.sleep(2.0 * (attempt + 1))
                    continue
                return None

            parsed = _parse_json_response(raw_text)
            if parsed is None:
                if attempt < self._max_retries - 1:
                    time.sleep(2.0 * (attempt + 1))
                    continue
                return None

            if isinstance(parsed, list):
                parsed = parsed[0] if parsed else None
            if not isinstance(parsed, dict):
                if attempt < self._max_retries - 1:
                    continue
                return None

            try:
                return _llm_result_to_pattern(paper, parsed)
            except Exception as e:
                print(f"    ⚠️  转换失败 ({paper.title[:40]}): {e}")
                if attempt < self._max_retries - 1:
                    continue
                return None

        return None
