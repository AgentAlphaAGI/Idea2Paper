"""
Build a canonical subdomain taxonomy for Path2 recall.

Usage:
  python Paper-KG-Pipeline/scripts/tools/build_subdomain_taxonomy.py
  python Paper-KG-Pipeline/scripts/tools/build_subdomain_taxonomy.py --patterns Paper-KG-Pipeline/output/nodes_pattern.json --output /tmp/subdomain_taxonomy.json
  python Paper-KG-Pipeline/scripts/tools/build_subdomain_taxonomy.py --target-k-min 40 --target-k-max 80
"""

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = SCRIPTS_DIR.parent
REPO_ROOT = PROJECT_ROOT.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from idea2paper.infra.dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env", override=False)
except Exception:
    pass

from idea2paper.config import OUTPUT_DIR, PipelineConfig, EMBEDDING_PROVIDER, EMBEDDING_API_URL, EMBEDDING_MODEL
from idea2paper.infra.embeddings import get_embeddings_batch


GENERIC_KEYWORDS = [
    "neural",
    "deep learning",
    "representation learning",
    "optimization",
    "benchmark",
    "machine learning",
    "artificial intelligence",
]


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_matrix(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def _truncate(text: str, max_chars: int) -> str:
    if not text:
        return ""
    return text[:max_chars]


def _tokenize(text: str) -> frozenset:
    if not text:
        return frozenset()
    # simple tokenization: lowercase + split on non-alphanum
    tokens = []
    current = []
    for ch in text.lower():
        if ch.isalnum():
            current.append(ch)
        else:
            if current:
                tokens.append("".join(current))
                current = []
    if current:
        tokens.append("".join(current))
    return frozenset(tokens)


def _jaccard(t1: frozenset, t2: frozenset) -> float:
    if not t1 or not t2:
        return 0.0
    inter = t1 & t2
    union = t1 | t2
    return len(inter) / len(union)


class _DSU:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


def _cluster_by_threshold(sim: np.ndarray, threshold: float) -> List[List[int]]:
    n = sim.shape[0]
    dsu = _DSU(n)
    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= threshold:
                dsu.union(i, j)
    groups = defaultdict(list)
    for i in range(n):
        groups[dsu.find(i)].append(i)
    return list(groups.values())


def _cluster_count(sim: np.ndarray, threshold: float) -> int:
    return len(_cluster_by_threshold(sim, threshold))


def _choose_threshold(sim: np.ndarray, k_min: int, k_max: int, t_min: float, t_max: float) -> float:
    if sim.shape[0] == 0:
        return t_min
    thresholds = np.linspace(t_max, t_min, int(round((t_max - t_min) / 0.01)) + 1)
    best = None
    k_mid = (k_min + k_max) / 2.0
    for th in thresholds:
        count = _cluster_count(sim, float(th))
        in_range = k_min <= count <= k_max
        if in_range:
            score = abs(count - k_mid)
        else:
            if count < k_min:
                score = k_min - count
            else:
                score = count - k_max
        if best is None or score < best[0] or (score == best[0] and th > best[1]):
            best = (score, float(th), count)
    return best[1] if best else t_min


def _safe_float(value, default=0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def build_taxonomy(
    patterns_path: Path,
    papers_path: Path | None,
    output_path: Path,
    target_k_min: int,
    target_k_max: int,
    sim_min: float,
    sim_max: float,
    min_papers: int,
    merge_min_sim: float,
    stoplist_ratio_max_median: float,
    stoplist_domain_ratio: float,
    max_exemplar_papers: int,
    max_cooccur: int,
    max_domains: int,
):
    patterns = _load_json(patterns_path)
    papers_by_id = {}
    if papers_path and papers_path.exists():
        papers = _load_json(papers_path)
        papers_by_id = {p.get("paper_id"): p for p in papers if p.get("paper_id")}

    raw_stats = {}
    co_counts = defaultdict(Counter)
    domain_counts = defaultdict(Counter)
    paper_ids_by_sd = defaultdict(set)
    all_domains = set()

    for pattern in patterns:
        sub_domains = pattern.get("sub_domains", []) or []
        if not sub_domains:
            continue
        pattern_id = pattern.get("pattern_id")
        domain = pattern.get("domain") or pattern.get("domain_id")
        if domain:
            all_domains.add(domain)
        size = pattern.get("size")
        if size is None:
            size = pattern.get("cluster_size")
        if size is None:
            size = len(pattern.get("exemplar_paper_ids", []) or [])
        size = _safe_float(size, 0.0)
        exemplar_ids = pattern.get("exemplar_paper_ids", []) or []

        for sd in sub_domains:
            if not sd:
                continue
            stats = raw_stats.setdefault(sd, {"pattern_ids": set(), "domain_ids": set(), "paper_count_est": 0.0})
            if pattern_id:
                stats["pattern_ids"].add(pattern_id)
            if domain:
                stats["domain_ids"].add(domain)
                domain_counts[sd][domain] += 1
            stats["paper_count_est"] += size
            for other in sub_domains:
                if other and other != sd:
                    co_counts[sd][other] += 1
            for pid in exemplar_ids:
                if pid:
                    paper_ids_by_sd[sd].add(pid)

    raw_subdomains = sorted(raw_stats.keys())

    def build_card(sd: str) -> str:
        parts = [f"Subdomain: {sd}"]
        co = [x for x, _ in co_counts[sd].most_common(max_cooccur)]
        if co:
            parts.append("Co-occurring: " + ", ".join(co))
        doms = [x for x, _ in domain_counts[sd].most_common(max_domains)]
        if doms:
            parts.append("Domains: " + ", ".join(doms))
        if papers_by_id:
            paper_ids = sorted(paper_ids_by_sd.get(sd, []))[:max_exemplar_papers]
            snippets = []
            for pid in paper_ids:
                paper = papers_by_id.get(pid) or {}
                title = paper.get("title", "")
                abstract = paper.get("abstract", "") or ""
                snippet = title
                if abstract:
                    snippet = f"{title} - {abstract}"
                snippets.append(_truncate(snippet, 200))
            if snippets:
                parts.append("Exemplars: " + " | ".join(snippets))
        return "\n".join(parts)

    cards = [build_card(sd) for sd in raw_subdomains]

    embeddings = None
    if raw_subdomains:
        embeddings = get_embeddings_batch(cards, timeout=120)
    if embeddings is not None:
        emb = np.array(embeddings, dtype=np.float32)
        emb = _normalize_matrix(emb)
        sim = emb @ emb.T
    else:
        # fallback to simple string similarity
        tokens = [_tokenize(card) for card in cards]
        n = len(tokens)
        sim = np.eye(n, dtype=np.float32)
        for i in range(n):
            for j in range(i + 1, n):
                s = _jaccard(tokens[i], tokens[j])
                sim[i, j] = s
                sim[j, i] = s
        emb = None

    threshold = _choose_threshold(sim, target_k_min, target_k_max, sim_min, sim_max)
    clusters = _cluster_by_threshold(sim, threshold)

    # canonical naming + raw -> canonical mapping
    mapping = {}
    canonical_groups = {}
    canonical_vectors = {}
    canonical_stats = {}

    for group in clusters:
        group_sds = [raw_subdomains[i] for i in group]
        if emb is not None:
            centroid = np.mean(emb[group], axis=0)
            centroid = centroid / (np.linalg.norm(centroid) or 1.0)
            best_idx = max(group, key=lambda i: float(np.dot(emb[i], centroid)))
            canonical = raw_subdomains[best_idx]
            canonical_vectors[canonical] = centroid
        else:
            canonical = max(group_sds, key=lambda sd: raw_stats[sd]["paper_count_est"])
        canonical_groups[canonical] = group_sds
        for sd in group_sds:
            mapping[sd] = canonical

    for canonical, group_sds in canonical_groups.items():
        paper_count = sum(raw_stats[sd]["paper_count_est"] for sd in group_sds)
        domain_ids = set()
        for sd in group_sds:
            domain_ids.update(raw_stats[sd]["domain_ids"])
        canonical_stats[canonical] = {
            "paper_count_est": paper_count,
            "domain_count": len(domain_ids),
        }

    # Stoplist
    counts = [v["paper_count_est"] for k, v in canonical_stats.items() if v["paper_count_est"] > 0]
    median = float(np.median(counts)) if counts else 0.0
    total_domains = max(1, len(all_domains))
    stoplist = set()
    for canonical, stats in canonical_stats.items():
        count = stats["paper_count_est"]
        domain_ratio = stats["domain_count"] / total_domains
        name_lower = canonical.lower()
        if median > 0 and count > median * stoplist_ratio_max_median:
            stoplist.add(canonical)
            continue
        if domain_ratio >= stoplist_domain_ratio and count >= median:
            stoplist.add(canonical)
            continue
        if any(key in name_lower for key in GENERIC_KEYWORDS):
            stoplist.add(canonical)

    # Merge small buckets (non-stoplist)
    if emb is not None:
        # compute canonical centroids from raw embeddings
        canon_list = list(canonical_groups.keys())
        canon_vecs = {}
        for canon in canon_list:
            idxs = [raw_subdomains.index(sd) for sd in canonical_groups[canon]]
            centroid = np.mean(emb[idxs], axis=0)
            centroid = centroid / (np.linalg.norm(centroid) or 1.0)
            canon_vecs[canon] = centroid
    else:
        canon_vecs = {canon: _tokenize(canon) for canon in canonical_groups}

    for canonical, stats in list(canonical_stats.items()):
        if canonical in stoplist:
            continue
        if stats["paper_count_est"] >= min_papers:
            continue
        # find nearest larger canonical (non-stoplist)
        best_target = None
        best_sim = -1.0
        for target, tstats in canonical_stats.items():
            if target == canonical or target in stoplist:
                continue
            if tstats["paper_count_est"] < min_papers:
                continue
            if emb is not None:
                sim_val = float(np.dot(canon_vecs[canonical], canon_vecs[target]))
            else:
                sim_val = _jaccard(canon_vecs[canonical], canon_vecs[target])
            if sim_val > best_sim:
                best_sim = sim_val
                best_target = target
        if best_target and best_sim >= merge_min_sim:
            for sd, canon in list(mapping.items()):
                if canon == canonical:
                    mapping[sd] = best_target
            # mark this canonical as merged out
            canonical_groups.pop(canonical, None)

    # rebuild canonical groups after merge
    canonical_groups = defaultdict(list)
    for sd, canon in mapping.items():
        canonical_groups[canon].append(sd)

    # rebuild stats
    canonical_stats = {}
    for canonical, group_sds in canonical_groups.items():
        paper_count = sum(raw_stats[sd]["paper_count_est"] for sd in group_sds)
        domain_ids = set()
        for sd in group_sds:
            domain_ids.update(raw_stats[sd]["domain_ids"])
        canonical_stats[canonical] = {
            "paper_count_est": paper_count,
            "domain_count": len(domain_ids),
        }

    # stats summary (exclude stoplist)
    active_counts = [v["paper_count_est"] for k, v in canonical_stats.items() if k not in stoplist and v["paper_count_est"] > 0]
    if active_counts:
        summary = {
            "min": int(min(active_counts)),
            "median": float(np.median(active_counts)),
            "max": int(max(active_counts)),
        }
    else:
        summary = {"min": 0, "median": 0, "max": 0}

    output = {
        "manifest": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "embedding_provider": EMBEDDING_PROVIDER,
            "embedding_api_url": EMBEDDING_API_URL,
            "embedding_model": EMBEDDING_MODEL,
            "nodes_pattern_sha256": _file_hash(patterns_path),
            "algorithm": {
                "target_k_min": int(target_k_min),
                "target_k_max": int(target_k_max),
                "sim_threshold": float(threshold),
                "min_papers": int(min_papers),
                "merge_min_sim": float(merge_min_sim),
                "stoplist_ratio_max_median": float(stoplist_ratio_max_median),
            },
        },
        "canonical_subdomains": sorted(canonical_groups.keys()),
        "mapping": dict(sorted(mapping.items(), key=lambda x: x[0])),
        "stoplist": sorted(stoplist),
        "stats": {
            "raw_count": len(raw_subdomains),
            "canonical_count": len(canonical_groups),
            "stoplist_count": len(stoplist),
            "paper_count_summary": summary,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(f"✅ taxonomy written: {output_path}")
    print(f"   raw={output['stats']['raw_count']} canonical={output['stats']['canonical_count']} stoplist={output['stats']['stoplist_count']}")
    print(f"   paper_count_summary={output['stats']['paper_count_summary']}")


def main():
    parser = argparse.ArgumentParser(description="Build canonical subdomain taxonomy (offline).")
    parser.add_argument("--patterns", type=str, default=str(OUTPUT_DIR / "nodes_pattern.json"))
    parser.add_argument("--papers", type=str, default=str(OUTPUT_DIR / "nodes_paper.json"))
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--target-k-min", type=int, default=40)
    parser.add_argument("--target-k-max", type=int, default=80)
    parser.add_argument("--sim-min", type=float, default=0.70)
    parser.add_argument("--sim-max", type=float, default=0.95)
    parser.add_argument("--min-papers", type=int, default=30)
    parser.add_argument("--merge-min-sim", type=float, default=0.75)
    parser.add_argument("--stoplist-ratio-max-median", type=float, default=10.0)
    parser.add_argument("--stoplist-domain-ratio", type=float, default=0.30)
    parser.add_argument("--max-exemplar-papers", type=int, default=5)
    parser.add_argument("--max-cooccur", type=int, default=5)
    parser.add_argument("--max-domains", type=int, default=3)
    args = parser.parse_args()

    patterns_path = Path(args.patterns)
    papers_path = Path(args.papers) if args.papers else None
    if not patterns_path.exists():
        raise FileNotFoundError(f"patterns not found: {patterns_path}")

    if args.output:
        output_path = Path(args.output)
    else:
        recall_dir = Path(PipelineConfig.RECALL_INDEX_DIR)
        output_path = recall_dir / "subdomain_taxonomy.json"

    build_taxonomy(
        patterns_path=patterns_path,
        papers_path=papers_path,
        output_path=output_path,
        target_k_min=args.target_k_min,
        target_k_max=args.target_k_max,
        sim_min=args.sim_min,
        sim_max=args.sim_max,
        min_papers=args.min_papers,
        merge_min_sim=args.merge_min_sim,
        stoplist_ratio_max_median=args.stoplist_ratio_max_median,
        stoplist_domain_ratio=args.stoplist_domain_ratio,
        max_exemplar_papers=args.max_exemplar_papers,
        max_cooccur=args.max_cooccur,
        max_domains=args.max_domains,
    )


if __name__ == "__main__":
    main()
