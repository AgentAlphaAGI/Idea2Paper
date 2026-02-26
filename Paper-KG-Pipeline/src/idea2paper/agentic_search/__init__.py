"""
Path4: Agentic Search — 联网检索召回通路

独立于 KG 三路召回的第四路：
  Planner   → 生成检索 query
  Searcher  → OpenAlex 检索 + 白名单硬过滤
  Extractor → 从 abstract 抽取 paper-level pattern
  Ranker    → pattern 内部独立排序 (relevance + recency + impact)
  Pipeline  → 端到端编排 (Planner → Searcher → Backtrack → Extractor → Ranker)
"""

from idea2paper.agentic_search.planner import AgenticPlanner
from idea2paper.agentic_search.searcher import AgenticSearcher
from idea2paper.agentic_search.extractor import AgenticExtractor
from idea2paper.agentic_search.ranker import Path4Ranker
from idea2paper.agentic_search.pipeline import AgenticResearchPipeline

__all__ = [
    "AgenticPlanner",
    "AgenticSearcher",
    "AgenticExtractor",
    "Path4Ranker",
    "AgenticResearchPipeline",
]
