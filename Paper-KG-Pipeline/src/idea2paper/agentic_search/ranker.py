"""
Path4 Ranker — pattern 内部独立排序

对齐方案: docs/PATH4_RANKER_DESIGN.md

打分维度:
  - relevance : cosine_sim(embed(user_idea), embed(summary_text))
  - recency   : year min-max 归一化
  - impact    : log(1 + citation_count) min-max 归一化

agenticSearch_score = α·relevance + β·recency_norm + γ·impact_norm

输出按 score 降序排列的 pattern 列表，每个 pattern 附带 rank_scores 字段。
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

if __package__ is None or __package__ == "":
    _CUR = Path(__file__).resolve()
    _SRC_ROOT = _CUR.parents[2]
    if str(_SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(_SRC_ROOT))

from idea2paper.infra.embeddings import get_embedding, get_embeddings_batch
from idea2paper.infra.run_context import get_logger


# ────────────────────────────────────────
# Defaults
# ────────────────────────────────────────
DEFAULT_ALPHA = 0.5   # relevance weight
DEFAULT_BETA = 0.3    # recency weight
DEFAULT_GAMMA = 0.2   # impact (citation) weight


# ────────────────────────────────────────
# Helpers
# ────────────────────────────────────────

def _minmax_norm(values: List[float]) -> List[float]:
    """Min-max normalize a list of floats to [0, 1]."""
    if not values:
        return []
    lo, hi = min(values), max(values)
    span = hi - lo
    if span < 1e-9:
        return [0.0] * len(values)
    return [(v - lo) / span for v in values]


def _cosine_sim(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return dot / (na * nb)


def _build_summary_text(pattern: dict, max_chars: int = 800) -> str:
    """Concatenate key summary fields into a single text for embedding."""
    summary = pattern.get("summary", {})
    parts = []
    for key in ("representative_ideas", "common_problems", "solution_approaches"):
        items = summary.get(key, [])
        if isinstance(items, list):
            parts.extend(items)
        elif isinstance(items, str):
            parts.append(items)
    text = " ".join(parts)
    if len(text) > max_chars:
        text = text[:max_chars]
    return text


# ────────────────────────────────────────
# Path4Ranker
# ────────────────────────────────────────

class Path4Ranker:
    """Rank Path4 patterns by relevance, recency, and impact.

    Usage:
        ranker = Path4Ranker()
        ranked = ranker.rank(patterns, user_idea)
        # ranked is the input list sorted by score, each with 'rank_scores' added
    """

    def __init__(
        self,
        alpha: float = DEFAULT_ALPHA,
        beta: float = DEFAULT_BETA,
        gamma: float = DEFAULT_GAMMA,
        logger=None,
    ):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.logger = logger or get_logger()

    def rank(
        self,
        patterns: List[dict],
        user_idea: str,
    ) -> List[dict]:
        """Score and rank patterns. Returns a new list sorted by score (desc).

        Each pattern dict gets a 'rank_scores' field added in-place:
          {
            "relevance": float,
            "recency_norm": float,
            "impact_norm": float,
            "agenticSearch_score": float,
            "rank": int          # 1-based
          }
        """
        if not patterns:
            return []

        n = len(patterns)
        print(f"\n📊 [Path4 Ranker] 对 {n} 条 pattern 进行内部排序...")

        # ── 1. Relevance via embedding ──
        relevances = self._compute_relevances(patterns, user_idea)

        # ── 2. Recency (year) ──
        years = [float(p.get("year", 0) or 0) for p in patterns]
        recency_norms = _minmax_norm(years)

        # ── 3. Impact (citation_count) ──
        impacts_raw = [math.log(1.0 + float(p.get("citation_count", 0) or 0)) for p in patterns]
        impact_norms = _minmax_norm(impacts_raw)

        # ── 4. Weighted score ──
        scores: List[float] = []
        for i in range(n):
            s = (self.alpha * relevances[i]
                 + self.beta * recency_norms[i]
                 + self.gamma * impact_norms[i])
            scores.append(round(s, 6))

        # ── 5. Sort by score desc (stable) ──
        indexed = sorted(range(n), key=lambda i: scores[i], reverse=True)

        ranked: List[dict] = []
        for rank_pos, idx in enumerate(indexed):
            p = patterns[idx]
            p["rank_scores"] = {
                "relevance": round(relevances[idx], 4),
                "recency_norm": round(recency_norms[idx], 4),
                "impact_norm": round(impact_norms[idx], 4),
                "agenticSearch_score": scores[idx],
                # Backward-compatible alias.
                "path4_score": scores[idx],
                "rank": rank_pos + 1,
            }
            ranked.append(p)

        # Print top-5
        print(f"\n  排序完成 (α={self.alpha}, β={self.beta}, γ={self.gamma})")
        for p in ranked[:5]:
            rs = p["rank_scores"]
            score = rs.get("agenticSearch_score", rs.get("path4_score", 0.0))
            print(f"  #{rs['rank']:2d}  score={score:.4f}  "
                  f"rel={rs['relevance']:.3f} rec={rs['recency_norm']:.3f} "
                  f"imp={rs['impact_norm']:.3f}  {p.get('title', '')[:60]}")
        if n > 5:
            print(f"  ... ({n - 5} more)")

        if self.logger:
            self.logger.log_event("agenticSearch_ranker_done", {
                "pattern_count": n,
                "alpha": self.alpha, "beta": self.beta, "gamma": self.gamma,
                "embedding_success": any(r > 0 for r in relevances),
            })

        return ranked

    # ──────────── internal ────────────

    def _compute_relevances(
        self, patterns: List[dict], user_idea: str
    ) -> List[float]:
        """Compute cosine similarity between user_idea and each pattern summary."""
        n = len(patterns)

        # Embed user_idea
        print("  ⏳ 计算 embedding (user_idea + pattern summaries)...")
        idea_emb = get_embedding(user_idea, logger=self.logger)
        if idea_emb is None:
            print("  ⚠️  embedding 调用失败, relevance 全部退化为 0")
            return [0.0] * n

        # Batch embed pattern summaries
        summary_texts = [_build_summary_text(p) for p in patterns]
        summary_embs = get_embeddings_batch(summary_texts, logger=self.logger)

        if summary_embs is None:
            # Fallback: embed one by one
            print("  ⚠️  batch embedding 失败, 逐条 fallback...")
            summary_embs = []
            for text in summary_texts:
                emb = get_embedding(text, logger=self.logger)
                summary_embs.append(emb)

        relevances = []
        for emb in summary_embs:
            if emb is not None:
                relevances.append(max(0.0, _cosine_sim(idea_emb, emb)))
            else:
                relevances.append(0.0)

        success_count = sum(1 for r in relevances if r > 0)
        print(f"  ✓ embedding 完成: {success_count}/{n} 条成功")
        return relevances


# ────────────────────────────────────────
# CLI entry point
# ────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Agentic Search Ranker — 对 agenticSearch_patterns.json 做内部排序",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python ranker.py "Use LLM agents with tool retrieval for complex task planning"
  python ranker.py "my idea" --input agenticSearch_patterns.json --output agenticSearch_patterns.json
        """,
    )
    parser.add_argument("idea", help="User research idea (用于计算 relevance)")
    parser.add_argument(
        "--input", "-i",
        default=None,
        help="输入 JSON 文件路径 (默认: output/agenticSearch_patterns.json)",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="输出 JSON 文件路径 (默认: 覆盖输入文件)",
    )
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--beta", type=float, default=DEFAULT_BETA)
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    args = parser.parse_args()

    # Resolve paths
    from idea2paper.config import OUTPUT_DIR
    input_path = args.input or str(OUTPUT_DIR / "agenticSearch_patterns.json")
    output_path = args.output or input_path

    # Load patterns
    print(f"📂 加载: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        patterns = json.load(f)
    print(f"   共 {len(patterns)} 条 pattern")

    if not patterns:
        print("⚠️  无 pattern, 退出")
        return

    # Rank
    ranker = Path4Ranker(alpha=args.alpha, beta=args.beta, gamma=args.gamma)
    ranked = ranker.rank(patterns, args.idea)

    # Save
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(ranked, f, ensure_ascii=False, indent=2)
    print(f"\n📝 排序结果已保存: {output_path} ({len(ranked)} 条)")


if __name__ == "__main__":
    main()
