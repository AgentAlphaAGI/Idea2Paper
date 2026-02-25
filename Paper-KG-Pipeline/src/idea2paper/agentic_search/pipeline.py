"""
Path4 Agentic Research Pipeline — Planner → Searcher → Backtrack → Extractor

端到端编排：
  1. Planner: 从 user_idea 生成 5-8 条检索 query
  2. Searcher: 对每条 query 调 Semantic Scholar + 白名单硬过滤
  3. Backtrack: 若白名单过滤后论文 < 阈值, Planner 再生成宽泛 query 重搜
  4. Extractor: 从论文 abstract 抽取 paper-level pattern (对齐 nodes_pattern.json)

参考: Gemini / OpenAI Deep Research 的 planning → search → gap → re-search 循环
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

# Allow direct script execution:
#   python Paper-KG-Pipeline/src/idea2paper/agentic_search/pipeline.py "my idea"
# by adding Paper-KG-Pipeline/src to sys.path when needed.
if __package__ is None or __package__ == "":
    _CUR = Path(__file__).resolve()
    _SRC_ROOT = _CUR.parents[2]
    if str(_SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(_SRC_ROOT))

from idea2paper.config import PipelineConfig, OUTPUT_DIR
from idea2paper.infra.run_context import get_logger
from idea2paper.agentic_search.planner import AgenticPlanner, PlannerOutput, QuerySpec
from idea2paper.agentic_search.searcher import AgenticSearcher, SearcherOutput, PaperStub
from idea2paper.agentic_search.extractor import AgenticExtractor, ExtractorOutput, PaperPattern


@dataclass
class AgenticResearchResult:
    """Full result of the agentic research pipeline."""
    user_idea: str
    planner_output: PlannerOutput
    searcher_output: SearcherOutput
    backtrack_used: bool = False
    backtrack_queries: List[QuerySpec] = field(default_factory=list)
    backtrack_searcher_output: Optional[SearcherOutput] = None
    extractor_output: Optional[ExtractorOutput] = None
    total_time_sec: float = 0.0

    @property
    def all_papers(self) -> List[PaperStub]:
        """All unique papers from initial + backtrack searches."""
        seen = set()
        papers = []
        for p in self.searcher_output.papers:
            if p.paper_id not in seen:
                seen.add(p.paper_id)
                papers.append(p)
        if self.backtrack_searcher_output:
            for p in self.backtrack_searcher_output.papers:
                if p.paper_id not in seen:
                    seen.add(p.paper_id)
                    papers.append(p)
        return papers

    @property
    def all_patterns(self) -> List[PaperPattern]:
        """All extracted patterns."""
        if self.extractor_output:
            return self.extractor_output.patterns
        return []

    def to_dict(self) -> dict:
        d = {
            "user_idea": self.user_idea,
            "planner": self.planner_output.to_dict(),
            "searcher": self.searcher_output.to_dict(),
            "backtrack_used": self.backtrack_used,
            "total_papers": len(self.all_papers),
            "total_patterns": len(self.all_patterns),
            "total_time_sec": round(self.total_time_sec, 2),
        }
        if self.backtrack_used:
            d["backtrack_queries"] = [q.to_dict() for q in self.backtrack_queries]
            if self.backtrack_searcher_output:
                d["backtrack_searcher"] = self.backtrack_searcher_output.to_dict()
        if self.extractor_output:
            d["extractor"] = self.extractor_output.to_dict()
        return d

    def summary(self) -> str:
        """Human-readable summary."""
        papers = self.all_papers
        venue_dist: Dict[str, int] = {}
        for p in papers:
            venue_dist[p.venue_norm] = venue_dist.get(p.venue_norm, 0) + 1

        lines = [
            f"📊 Agentic Research 结果摘要",
            f"   Idea: {self.user_idea[:100]}{'...' if len(self.user_idea) > 100 else ''}",
            f"   Queries: {len(self.planner_output.final_queries)} 条"
            + (f" + {len(self.backtrack_queries)} 条 backtrack" if self.backtrack_used else ""),
            f"   论文总数: {len(papers)} 篇 (白名单内, 去重后)",
            f"   Pattern 数: {len(self.all_patterns)} 条",
            f"   会议分布: {venue_dist}",
            f"   耗时: {self.total_time_sec:.1f}s",
        ]
        return "\n".join(lines)


class AgenticResearchPipeline:
    """Orchestrate Planner → Searcher → (Backtrack) → Extractor for Path4.

    Usage:
        pipeline = AgenticResearchPipeline()
        result = pipeline.run("Use GNN for multi-hop reasoning over KGs")
        print(result.summary())
        for pattern in result.all_patterns:
            print(pattern.name, pattern.domain)
    """

    def __init__(
        self,
        logger=None,
        s2_api_key: Optional[str] = None,
    ):
        self.logger = logger or get_logger()
        self._planner = AgenticPlanner(logger=self.logger)
        self._searcher = AgenticSearcher(logger=self.logger, s2_api_key=s2_api_key)
        self._extractor = AgenticExtractor(logger=self.logger)
        self._backtrack_enable = bool(PipelineConfig.PATH4_BACKTRACK_ENABLE)
        self._backtrack_threshold = int(PipelineConfig.PATH4_BACKTRACK_THRESHOLD)
        self._backtrack_extra = int(PipelineConfig.PATH4_BACKTRACK_EXTRA_QUERIES)

    def run(
        self,
        user_idea: str,
        idea_brief: Optional[Dict] = None,
    ) -> AgenticResearchResult:
        """Run the full agentic research pipeline.

        Args:
            user_idea: Raw research idea (Chinese or English).
            idea_brief: Optional structured brief from IdeaPackager.

        Returns:
            AgenticResearchResult with papers and extracted patterns.
        """
        t0 = time.time()

        print("=" * 80)
        print("🚀 Path4 Agentic Research Pipeline")
        print("=" * 80)
        print(f"\n【User Idea】\n{user_idea}\n")

        # ── Phase 1: Plan ──
        planner_output = self._planner.plan(user_idea, idea_brief=idea_brief)

        # ── Phase 2: Search ──
        searcher_output = self._searcher.search(planner_output.final_queries)

        # ── Phase 3: Backtrack (if needed) ──
        backtrack_used = False
        bt_queries: List[QuerySpec] = []
        bt_searcher_output: Optional[SearcherOutput] = None

        paper_count = len(searcher_output.papers)
        if (
            self._backtrack_enable
            and paper_count < self._backtrack_threshold
            and planner_output.anchors
        ):
            print(f"\n⚠️  白名单过滤后仅 {paper_count} 篇 (阈值={self._backtrack_threshold}), 触发 Backtrack...")
            bt_queries = self._planner.backtrack_queries(
                user_idea,
                original_anchors=planner_output.anchors,
                extra_count=self._backtrack_extra,
            )
            if bt_queries:
                bt_searcher_output = self._searcher.search(bt_queries)
                backtrack_used = True

        # ── Phase 4: Extract patterns ──
        # Collect all unique papers (initial + backtrack)
        seen_ids: set = set()
        all_papers: List[PaperStub] = []
        for p in searcher_output.papers:
            if p.paper_id not in seen_ids:
                seen_ids.add(p.paper_id)
                all_papers.append(p)
        if bt_searcher_output:
            for p in bt_searcher_output.papers:
                if p.paper_id not in seen_ids:
                    seen_ids.add(p.paper_id)
                    all_papers.append(p)

        extractor_output: Optional[ExtractorOutput] = None
        if all_papers:
            extractor_output = self._extractor.extract(all_papers)
        else:
            print("\n  ⏭️  无论文可抽取, 跳过 Extractor")

        total_time = time.time() - t0

        result = AgenticResearchResult(
            user_idea=user_idea,
            planner_output=planner_output,
            searcher_output=searcher_output,
            backtrack_used=backtrack_used,
            backtrack_queries=bt_queries,
            backtrack_searcher_output=bt_searcher_output,
            extractor_output=extractor_output,
            total_time_sec=total_time,
        )

        # ── Summary ──
        print("\n" + "=" * 80)
        print(result.summary())
        print("=" * 80)

        if self.logger:
            self.logger.log_event("path4_pipeline_done", {
                "total_papers": len(result.all_papers),
                "total_patterns": len(result.all_patterns),
                "backtrack_used": backtrack_used,
                "total_time_sec": round(total_time, 2),
            })

        return result

    @staticmethod
    def save_patterns(result: AgenticResearchResult, out_path: Optional[str] = None) -> Optional[str]:
        """Save extracted patterns to JSON (nodes_pattern.json compatible).

        Args:
            result: Pipeline result.
            out_path: Output file path. Defaults to output/path4_patterns.json.

        Returns:
            Path of the saved file, or None if no patterns to save.
        """
        patterns = result.all_patterns
        if not patterns:
            return None

        if out_path is None:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            out_path = str(OUTPUT_DIR / "path4_patterns.json")

        data = [p.to_nodes_pattern_format() for p in patterns]
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"\n📝 Pattern 已保存: {out_path} ({len(data)} 条)")
        return out_path


# ────────────────────────────────────────
# CLI entry point
# ────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(
        description="Path4 Agentic Search — Planner + Searcher 调试入口",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 最简用法
  python pipeline.py "Use GNN for multi-hop reasoning over knowledge graphs"

  # 传 idea_brief (JSON 文件)
  python pipeline.py "my idea text" --brief path/to/brief.json

  # 指定 Semantic Scholar API key，结果保存到文件
  python pipeline.py "my idea" --s2-key sk-xxx --out /tmp/path4_result.json
        """,
    )
    parser.add_argument("idea", help="Research idea text (可中文可英文)")
    parser.add_argument(
        "--brief",
        metavar="JSON_FILE",
        default=None,
        help="idea_brief JSON 文件路径 (可选)，格式见设计文档",
    )
    parser.add_argument(
        "--s2-key",
        metavar="KEY",
        default=None,
        help="Semantic Scholar API key (可选，未填则使用匿名限速 1 req/s)",
    )
    parser.add_argument(
        "--out",
        metavar="OUTPUT_FILE",
        default=None,
        help="将完整结果写入 JSON 文件 (可选)",
    )
    args = parser.parse_args()

    # Load optional idea_brief
    idea_brief = None
    if args.brief:
        try:
            with open(args.brief, "r", encoding="utf-8") as f:
                idea_brief = json.load(f)
            print(f"[CLI] 已加载 idea_brief: {args.brief}")
        except Exception as e:
            print(f"[CLI] ⚠️  加载 --brief 文件失败: {e}", file=sys.stderr)

    # Run pipeline
    pipeline = AgenticResearchPipeline(s2_api_key=args.s2_key)
    result = pipeline.run(args.idea, idea_brief=idea_brief)

    # Print per-paper detail
    print(f"\n{'─'*80}")
    print(f"📄 论文列表 (共 {len(result.all_papers)} 篇):")
    for i, p in enumerate(result.all_papers, 1):
        print(f"  [{i:02d}] [{p.venue_norm}] [{p.year}] {p.title}")
        if p.abstract:
            snippet = p.abstract[:120].replace("\n", " ")
            print(f"       abstract: {snippet}...")
        print(f"       url: {p.url or '(no url)'}")

    # Print extracted patterns
    if result.all_patterns:
        print(f"\n{'─'*80}")
        print(f"🧩 抽取的 Pattern (共 {len(result.all_patterns)} 条):")
        for i, pat in enumerate(result.all_patterns, 1):
            print(f"  [{i:02d}] [{pat.venue_norm}] {pat.name}")
            print(f"       domain: {pat.domain}  sub_domains: {pat.sub_domains}")
            problems = pat.summary.get("common_problems", [])
            if problems:
                print(f"       problem: {problems[0][:100]}...")
            stories = pat.summary.get("story", [])
            if stories:
                print(f"       story:   {stories[0][:100]}...")

    # Save patterns to nodes_pattern.json-compatible file
    AgenticResearchPipeline.save_patterns(result)

    # Save full result to file
    if args.out:
        out_data = result.to_dict()
        out_data["papers_detail"] = [p.to_dict() for p in result.all_papers]
        out_data["patterns_detail"] = [p.to_nodes_pattern_format() for p in result.all_patterns]
        try:
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(out_data, f, ensure_ascii=False, indent=2)
            print(f"\n[CLI] 结果已写入: {args.out}")
        except Exception as e:
            print(f"[CLI] ⚠️  写文件失败: {e}", file=sys.stderr)
