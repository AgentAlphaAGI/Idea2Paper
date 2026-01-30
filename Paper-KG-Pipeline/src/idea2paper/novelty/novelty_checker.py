import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from idea2paper.config import (
    NOVELTY_TOPK,
    NOVELTY_HIGH_TH,
    NOVELTY_MEDIUM_TH,
    NOVELTY_INDEX_DIR,
    NOVELTY_REPORT_IN_OUTPUT,
    OUTPUT_DIR,
    RESULTS_ROOT,
)
from idea2paper.novelty.novelty_index import NoveltyIndex, build_story_text, keyword_overlap, build_paper_text


class NoveltyChecker:
    def __init__(self, papers: List[Dict], nodes_paper_path: Path, logger=None):
        self.papers = papers
        self.nodes_paper_path = Path(nodes_paper_path)
        self.logger = logger
        self.index = NoveltyIndex(
            papers=self.papers,
            index_dir=NOVELTY_INDEX_DIR,
            nodes_paper_path=self.nodes_paper_path,
            logger=logger
        )

    def _risk_level(self, max_sim: float, embedding_available: bool) -> str:
        if not embedding_available:
            return "unknown"
        if max_sim >= NOVELTY_HIGH_TH:
            return "high"
        if max_sim >= NOVELTY_MEDIUM_TH:
            return "medium"
        return "low"

    def check(self, story: Dict, run_id: str, user_idea: str) -> Dict:
        story_text = build_story_text(story)
        index_status = self.index.ensure_index()
        candidates, info = self.index.query(story_text, NOVELTY_TOPK)

        # fill keyword_overlap if cosine used
        if candidates and candidates[0].get("keyword_overlap") is None:
            for c in candidates:
                # compute overlap with full paper text for top candidates
                paper = next((p for p in self.papers if p.get("paper_id") == c["paper_id"]), None)
                if paper:
                    c["keyword_overlap"] = keyword_overlap(story_text, build_paper_text(paper))
                else:
                    c["keyword_overlap"] = 0.0

        max_sim = 0.0
        if candidates:
            if info["embedding_available"]:
                max_sim = max(c.get("cosine", 0.0) or 0.0 for c in candidates)
            else:
                max_sim = max(c.get("keyword_overlap", 0.0) or 0.0 for c in candidates)

        risk_level = self._risk_level(max_sim, info["embedding_available"])

        report = {
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "user_idea": user_idea,
            "embedding_available": info["embedding_available"],
            "embedding_model": "Qwen/Qwen3-Embedding-8B" if info["embedding_available"] else None,
            "top_k": NOVELTY_TOPK,
            "thresholds": {
                "high": NOVELTY_HIGH_TH,
                "medium": NOVELTY_MEDIUM_TH
            },
            "risk_level": risk_level,
            "max_similarity": max_sim,
            "candidates": candidates[:10],
            "notes": list(set(index_status.get("notes", []) + info.get("notes", [])))
        }

        # write report to results/run_...
        results_dir = Path(RESULTS_ROOT) / run_id
        results_dir.mkdir(parents=True, exist_ok=True)
        report_path = results_dir / "novelty_report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        report["report_path"] = str(report_path)

        # optional output dir
        if NOVELTY_REPORT_IN_OUTPUT:
            out_path = OUTPUT_DIR / "novelty_report.json"
            out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            report["output_path"] = str(out_path)

        return report
