#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_clusters.py

Purpose
- Flatten paper JSONL -> pattern records
- Embed pattern text (Story-centric by default)
- UMAP + HDBSCAN clustering
- Compute cluster coherence metrics
- Fit Zipf (rank-size) stats
- LLM-based concise cluster naming (instead of top-words)
- Auto-tier clusters (A/B/C) and write tier_A/B/C.jsonl
- Generate report.md with Zipf + noise share + Top-10 table

Input JSONL format (each line):
{
  "paper_id": "...",
  "paper_title": "...",
  "idea": "...",
  "domain": "...",
  "sub_domains": [...],
  "research_patterns": [
    {"base_problem": "...", "solution_pattern": "...", "story": "...", "application": "..."}
  ]
}

Outputs (in --outdir):
- patterns_flat.jsonl               (flattened pattern records)
- embeddings.npy                    (float32 matrix)
- assignments.jsonl                 (pattern -> cluster labels)
- clusters.jsonl                    (cluster-level summary, incl. coherence + llm name)
- cluster_library.jsonl             (RAG-ready cluster objects w/ exemplars)
- tier_A.jsonl / tier_B.jsonl / tier_C.jsonl
- report.md

usage:
python analyze_clusters.py \
  --input your_extracted_papers.jsonl \
  --outdir output \
  --sbert_model sentence-transformers/all-MiniLM-L6-v2 \
  --llm_name \
  --llm_model gpt-4.1-mini

"""

from __future__ import annotations

import os
import re
import json
import math
import time
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Optional deps
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    import umap
except Exception:
    umap = None

try:
    import hdbscan
except Exception:
    hdbscan = None


# ----------------------------
# IO utils
# ----------------------------
def read_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


# ----------------------------
# Math utils
# ----------------------------
def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(n, eps, None)


def cosine_sim_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    # expects rows normalized
    return A @ B.T


def safe_mean(xs: List[float]) -> float:
    return float(np.mean(xs)) if xs else float("nan")


# ----------------------------
# Flatten papers -> patterns
# ----------------------------
def ensure_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def flatten_papers_to_patterns(raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    pid_counter = 0
    for paper in raw:
        paper_id = paper.get("paper_id") or paper.get("id") or ""
        paper_title = paper.get("paper_title") or paper.get("title") or ""
        idea = (paper.get("idea") or "").strip()
        domain = (paper.get("domain") or "待明确领域").strip()
        sub_domains = ensure_list(paper.get("sub_domains"))
        sub_domains = [str(s).strip() for s in sub_domains if str(s).strip()]

        patterns = paper.get("research_patterns") or []
        for j, rp in enumerate(patterns):
            base_problem = (rp.get("base_problem") or "").strip()
            solution_pattern = (rp.get("solution_pattern") or "").strip()
            story = (rp.get("story") or "").strip()
            application = (rp.get("application") or "").strip()

            items.append({
                "paper_id": paper_id,
                "paper_title": paper_title,
                "pattern_id": f"p{j}",
                "global_pattern_id": f"g{pid_counter}",
                "idea": idea,
                "domain": domain,
                "sub_domains": sub_domains,
                "base_problem": base_problem,
                "solution_pattern": solution_pattern,
                "story": story,
                "application": application,
            })
            pid_counter += 1
    return items


def build_text(item: Dict[str, Any], template: str) -> str:
    # Keep missing keys as empty strings
    def g(k: str) -> str:
        v = item.get(k, "")
        if isinstance(v, list):
            return ", ".join([str(x) for x in v])
        return str(v)

    return template.format(
        story=g("story"),
        base_problem=g("base_problem"),
        solution_pattern=g("solution_pattern"),
        idea=g("idea"),
        domain=g("domain"),
        sub_domains=g("sub_domains"),
        application=g("application"),
        paper_title=g("paper_title"),
        paper_id=g("paper_id"),
        pattern_id=g("pattern_id"),
        global_pattern_id=g("global_pattern_id"),
    ).strip()


# ----------------------------
# Embedding
# ----------------------------
def embed_texts_sbert(texts: List[str], model_name: str, batch_size: int = 64) -> np.ndarray:
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers is not installed. pip install sentence-transformers")
    model = SentenceTransformer(model_name)
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return emb.astype(np.float32)


# ----------------------------
# Clustering
# ----------------------------
def run_umap_hdbscan(
    X: np.ndarray,
    umap_neighbors: int,
    umap_components: int,
    umap_min_dist: float,
    hdb_min_cluster_size: int,
    hdb_min_samples: int,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    if umap is None:
        raise RuntimeError("umap-learn is not installed. pip install umap-learn")
    if hdbscan is None:
        raise RuntimeError("hdbscan is not installed. pip install hdbscan")

    reducer = umap.UMAP(
        n_neighbors=umap_neighbors,
        n_components=umap_components,
        min_dist=umap_min_dist,
        metric="cosine",
        random_state=random_state,
    )
    Z = reducer.fit_transform(X)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=hdb_min_cluster_size,
        min_samples=hdb_min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(Z)
    probs = getattr(clusterer, "probabilities_", np.ones(len(labels), dtype=np.float32))
    return labels.astype(int), probs.astype(np.float32)


# ----------------------------
# Coherence
# ----------------------------
@dataclass
class CoherenceStats:
    centroid_mean: float
    centroid_p25: float
    centroid_p50: float
    centroid_p75: float
    pairwise_sample_mean: float
    pairwise_sample_p50: float


def compute_cluster_coherence(
    Xn: np.ndarray,
    idxs: np.ndarray,
    pairwise_sample_n: int = 120,
    rng: Optional[np.random.Generator] = None,
) -> CoherenceStats:
    """
    Xn: normalized embeddings (N, d)
    idxs: indices of members in this cluster
    """
    if rng is None:
        rng = np.random.default_rng(42)

    V = Xn[idxs]
    if V.shape[0] == 0:
        return CoherenceStats(float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan"))

    centroid = V.mean(axis=0, keepdims=True)
    centroid = centroid / np.clip(np.linalg.norm(centroid), 1e-12, None)

    sims_to_centroid = (V @ centroid.T).reshape(-1)
    centroid_mean = float(np.mean(sims_to_centroid))
    centroid_p25 = float(np.quantile(sims_to_centroid, 0.25))
    centroid_p50 = float(np.quantile(sims_to_centroid, 0.50))
    centroid_p75 = float(np.quantile(sims_to_centroid, 0.75))

    # Pairwise (sampled) coherence
    m = V.shape[0]
    if m < 2:
        pw_mean = float("nan")
        pw_p50 = float("nan")
    else:
        k = min(pairwise_sample_n, m)
        sample = rng.choice(m, size=k, replace=False)
        Vs = V[sample]
        S = Vs @ Vs.T  # (k,k), cosine since normalized
        triu = S[np.triu_indices(k, k=1)]
        pw_mean = float(np.mean(triu)) if triu.size else float("nan")
        pw_p50 = float(np.quantile(triu, 0.50)) if triu.size else float("nan")

    return CoherenceStats(
        centroid_mean=centroid_mean,
        centroid_p25=centroid_p25,
        centroid_p50=centroid_p50,
        centroid_p75=centroid_p75,
        pairwise_sample_mean=pw_mean,
        pairwise_sample_p50=pw_p50,
    )


# ----------------------------
# Zipf fit
# ----------------------------
@dataclass
class ZipfStats:
    alpha: float
    r2: float
    topk_share: Dict[str, float]


def fit_zipf(cluster_sizes_desc: List[int], topk_list: List[int]) -> ZipfStats:
    """
    Fit log(size) = a + b*log(rank), alpha = -b (positive)
    """
    sizes = np.array(cluster_sizes_desc, dtype=np.float64)
    ranks = np.arange(1, len(sizes) + 1, dtype=np.float64)

    x = np.log(ranks)
    y = np.log(np.clip(sizes, 1e-12, None))

    # Linear regression
    b, a = np.polyfit(x, y, 1)
    yhat = a + b * x
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    alpha = float(-b)

    total = float(np.sum(sizes))
    topk_share = {}
    for k in topk_list:
        topk_share[str(k)] = float(np.sum(sizes[:k]) / total) if total > 0 else float("nan")

    return ZipfStats(alpha=alpha, r2=r2, topk_share=topk_share)


# ----------------------------
# Cluster naming via LLM (concise)
# ----------------------------
def _truncate(s: str, max_chars: int) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    return s[:max_chars]


def llm_cluster_name(
    exemplars: List[Dict[str, Any]],
    model: str,
    api_base: Optional[str] = None,
    temperature: float = 0.2,
    max_retries: int = 3,
    sleep_s: float = 0.8,
) -> str:
    """
    Uses OpenAI-compatible Chat Completions (no response_format).
    Requires OPENAI_API_KEY in environment.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Set it to enable LLM cluster naming.")

    # Lazy import to avoid hard dependency
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("openai package not installed. pip install openai") from e

    client_kwargs = {}
    if api_base:
        client_kwargs["base_url"] = api_base
    client = OpenAI(api_key=api_key, **client_kwargs)

    # Build a compact prompt: story-first
    lines = []
    for i, ex in enumerate(exemplars[:8]):
        story = _truncate(ex.get("story", ""), 220)
        bp = _truncate(ex.get("base_problem", ""), 180)
        sol = _truncate(ex.get("solution_pattern", ""), 180)
        dom = ex.get("domain", "")
        subs = ex.get("sub_domains", [])
        subs_s = ", ".join(subs) if isinstance(subs, list) else str(subs)
        lines.append(f"- Ex{i+1} Domain: {dom} | Sub: {subs_s}\n  BaseProblem: {bp}\n  Story: {story}\n  Solution: {sol}")

    prompt = (
    "You are labeling a cluster of research narrative/pattern exemplars extracted from top-tier machine learning papers.\n"
    "Task: produce ONE concise English cluster name (3–6 words) that captures the shared research narrative or pattern.\n"
    "Constraints:\n"
    "1) The name MUST be in English.\n"
    "2) Avoid vague or generic words such as 'method', 'framework', 'model', 'approach', 'improvement', or 'optimization'.\n"
    "3) Prefer a distinctive *research story* angle, such as problem reframing, assumption removal, auditability, robustness, reliability, scalability, or efficiency–generality trade-offs.\n"
    "4) The name should sound like a top-conference research theme or paradigm, not a paper title.\n"
    "5) Output ONLY the name, with no punctuation, quotes, or extra text.\n\n"
    "Exemplars:\n"
    + "\n".join(lines)
)


    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": "You are a precise research taxonomy assistant."},
                    {"role": "user", "content": prompt},
                ],
            )
            name = resp.choices[0].message.content.strip()
            #name = re.sub(r"[\"'“”‘’`]", "", name)
            #name = re.sub(r"\s+", "", name)
            # Hard clamp
            #if len(name) > 18:
            #    name = name[:18]
            if not name:
                raise ValueError("Empty name from LLM.")
            return name
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(sleep_s * (attempt + 1))

    raise RuntimeError("LLM naming failed unexpectedly.")


# ----------------------------
# Tiering + report
# ----------------------------
def assign_tiers(
    clusters: List[Dict[str, Any]],
    size_A: int,
    size_B: int,
    coh_A: float,
    coh_B: float,
    coh_field: str = "coherence_centroid_mean",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    A, B, C = [], [], []
    for c in clusters:
        if c.get("cluster_id") == -1:
            continue
        sz = int(c.get("size", 0))
        coh = c.get(coh_field)
        coh_val = float(coh) if coh is not None and not (isinstance(coh, float) and math.isnan(coh)) else float("nan")

        if (sz >= size_A) and (not math.isnan(coh_val)) and (coh_val >= coh_A):
            c["tier"] = "A"
            A.append(c)
        elif (sz >= size_B) and (not math.isnan(coh_val)) and (coh_val >= coh_B):
            c["tier"] = "B"
            B.append(c)
        else:
            c["tier"] = "C"
            C.append(c)
    return A, B, C


def md_table(rows: List[List[str]], headers: List[str]) -> str:
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def build_report_md(
    total_patterns: int,
    n_clusters_ex_noise: int,
    noise_count: int,
    zipf: ZipfStats,
    top10: List[Dict[str, Any]],
    tier_counts: Dict[str, int],
) -> str:
    noise_share = noise_count / total_patterns if total_patterns > 0 else float("nan")

    top_rows = []
    for c in top10:
        cid = str(c["cluster_id"])
        name = c.get("cluster_name", "")
        sz = int(c.get("size", 0))
        share = (sz / total_patterns) if total_patterns > 0 else float("nan")
        coh = c.get("coherence_centroid_mean", float("nan"))
        tier = c.get("tier", "")
        top_rows.append([
            cid,
            name,
            str(sz),
            f"{share:.3f}",
            (f"{coh:.3f}" if not (isinstance(coh, float) and math.isnan(coh)) else "nan"),
            tier,
        ])

    return f"""# Cluster Analysis Report

## Summary
- Patterns: **{total_patterns}**
- Clusters (excluding noise): **{n_clusters_ex_noise}**
- Noise/outliers (-1): **{noise_count}**  (share: **{noise_share:.3%}**)

## Zipf (rank-size)
- alpha (rank-size slope): **{zipf.alpha:.3f}**
- r2 (log-log fit): **{zipf.r2:.3f}**
- topk_share: {json.dumps(zipf.topk_share, ensure_ascii=False)}

## Tiers
- Tier A: **{tier_counts.get("A", 0)}**
- Tier B: **{tier_counts.get("B", 0)}**
- Tier C: **{tier_counts.get("C", 0)}**

## Top-10 Clusters
{md_table(top_rows, ["cluster_id", "cluster_name", "size", "share", "coh", "tier"])}

"""
# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input JSONL of papers (paper-level objects).")
    ap.add_argument("--outdir", default="output", help="Output directory.")
    ap.add_argument(
        "--template",
        default="Story: {story}\nBase Problem: {base_problem}\nSolution: {solution_pattern}\nIdea: {idea}",
        help="Text template used for embedding.",
    )

    # Embedding
    ap.add_argument("--embed_backend", choices=["sbert"], default="sbert")
    ap.add_argument("--sbert_model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--embed_batch_size", type=int, default=64)

    # UMAP/HDBSCAN
    ap.add_argument("--umap_neighbors", type=int, default=15)
    ap.add_argument("--umap_components", type=int, default=5)
    ap.add_argument("--umap_min_dist", type=float, default=0.0)
    ap.add_argument("--hdb_min_cluster_size", type=int, default=15)
    ap.add_argument("--hdb_min_samples", type=int, default=5)

    # Coherence
    ap.add_argument("--pairwise_sample_n", type=int, default=120)

    # Zipf
    ap.add_argument("--zipf_topk", default="1,3,5,10,20")

    # LLM naming
    ap.add_argument("--llm_name", action="store_true", help="Use LLM to generate concise cluster_name.")
    ap.add_argument("--llm_model", default="gpt-4.1-mini")
    ap.add_argument("--llm_api_base", default=None)
    ap.add_argument("--llm_temperature", type=float, default=0.2)

    # Tiering thresholds
    ap.add_argument("--tier_size_A", type=int, default=30)
    ap.add_argument("--tier_size_B", type=int, default=10)
    ap.add_argument("--tier_coh_A", type=float, default=0.40)
    ap.add_argument("--tier_coh_B", type=float, default=0.30)

    args = ap.parse_args()

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    raw = read_jsonl(args.input)
    patterns = flatten_papers_to_patterns(raw)
    print(f"Patterns: {len(patterns)}")

    # Save flattened patterns
    flat_path = os.path.join(outdir, "patterns_flat.jsonl")
    write_jsonl(flat_path, patterns)

    # Build embed texts
    texts = [build_text(p, args.template) for p in patterns]

    # Embed
    if args.embed_backend == "sbert":
        X = embed_texts_sbert(texts, args.sbert_model, args.embed_batch_size)
    else:
        raise RuntimeError("Unsupported embed_backend")

    # Ensure normalized (SBERT normalize_embeddings=True already, but keep safe)
    Xn = l2_normalize(X)

    np.save(os.path.join(outdir, "embeddings.npy"), Xn)

    # Cluster
    labels, probs = run_umap_hdbscan(
        Xn,
        umap_neighbors=args.umap_neighbors,
        umap_components=args.umap_components,
        umap_min_dist=args.umap_min_dist,
        hdb_min_cluster_size=args.hdb_min_cluster_size,
        hdb_min_samples=args.hdb_min_samples,
    )

    # Assignments
    assignments = []
    for p, lab, pr in zip(patterns, labels, probs):
        assignments.append({
            "paper_id": p.get("paper_id"),
            "paper_title": p.get("paper_title"),
            "global_pattern_id": p.get("global_pattern_id"),
            "pattern_id": p.get("pattern_id"),
            "domain": p.get("domain"),
            "sub_domains": p.get("sub_domains"),
            "cluster_id": int(lab),
            "cluster_prob": float(pr),
        })
    write_jsonl(os.path.join(outdir, "assignments.jsonl"), assignments)

    # Build cluster index
    cluster_to_idxs: Dict[int, List[int]] = {}
    for i, lab in enumerate(labels):
        cluster_to_idxs.setdefault(int(lab), []).append(i)

    # Cluster summaries (excluding noise for counts)
    noise_count = len(cluster_to_idxs.get(-1, []))
    cluster_ids = sorted([cid for cid in cluster_to_idxs.keys() if cid != -1])
    print(f"Clusters (excluding noise): {len(cluster_ids)}")
    print(f"Noise/outliers (-1): {noise_count}")

    # Compute per-cluster coherence + facets + exemplars
    rng = np.random.default_rng(42)
    cluster_summaries = []
    cluster_library = []

    for cid in cluster_ids + ([-1] if -1 in cluster_to_idxs else []):
        idxs = np.array(cluster_to_idxs[cid], dtype=int)
        size = int(idxs.size)

        # Coherence only meaningful for non-noise clusters; for -1 keep NaN
        if cid != -1 and size > 0:
            coh = compute_cluster_coherence(Xn, idxs, pairwise_sample_n=args.pairwise_sample_n, rng=rng)
        else:
            coh = CoherenceStats(float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan"))

        # Domain/sub_domain distribution
        doms = [patterns[i].get("domain", "UNKNOWN") for i in idxs]
        subs = []
        for i in idxs:
            sd = patterns[i].get("sub_domains", [])
            if isinstance(sd, list):
                subs.extend(sd)
            elif sd:
                subs.append(str(sd))

        def top_counts(xs: List[str], k: int = 5) -> List[Tuple[str, int]]:
            from collections import Counter
            c = Counter([x for x in xs if x])
            return c.most_common(k)

        dom_top = top_counts(doms, 5)
        sub_top = top_counts(subs, 8)

        # Choose exemplars by highest membership prob (fallback random)
        # For -1, pick random few
        if cid != -1:
            idxs_list = idxs.tolist()
            idxs_list.sort(key=lambda i: probs[i], reverse=True)
            exemplar_idxs = idxs_list[:10]
        else:
            exemplar_idxs = idxs.tolist()[:10]

        exemplars = []
        for i in exemplar_idxs:
            exemplars.append({
                "paper_id": patterns[i].get("paper_id"),
                "paper_title": patterns[i].get("paper_title"),
                "global_pattern_id": patterns[i].get("global_pattern_id"),
                "domain": patterns[i].get("domain"),
                "sub_domains": patterns[i].get("sub_domains"),
                "idea": patterns[i].get("idea"),
                "base_problem": patterns[i].get("base_problem"),
                "solution_pattern": patterns[i].get("solution_pattern"),
                "story": patterns[i].get("story"),
                "application": patterns[i].get("application"),
            })

        # LLM name (only for non-noise clusters)
        cluster_name = ""
        if cid != -1 and args.llm_name:
            cluster_name = llm_cluster_name(
                exemplars=exemplars,
                model=args.llm_model,
                api_base=args.llm_api_base,
                temperature=args.llm_temperature,
            )
        else:
            # Placeholder if LLM naming disabled; keep deterministic but minimal
            cluster_name = f"Cluster{cid}"

        summary = {
            "cluster_id": int(cid),
            "cluster_name": cluster_name,
            "size": size,

            "coherence_centroid_mean": coh.centroid_mean,
            "coherence_centroid_p25": coh.centroid_p25,
            "coherence_centroid_p50": coh.centroid_p50,
            "coherence_centroid_p75": coh.centroid_p75,
            "coherence_pairwise_sample_mean": coh.pairwise_sample_mean,
            "coherence_pairwise_sample_p50": coh.pairwise_sample_p50,

            "domain_top": [{"domain": d, "count": n} for d, n in dom_top],
            "sub_domain_top": [{"sub_domain": s, "count": n} for s, n in sub_top],
        }
        cluster_summaries.append(summary)

        # Cluster library object (RAG-ready, excluding noise)
        if cid != -1:
            cluster_library.append({
                "cluster_id": int(cid),
                "cluster_name": cluster_name,
                "size": size,
                "retrieval_facets": {
                    "domain": dom_top[0][0] if dom_top else "待明确领域",
                    "sub_domains": [x["sub_domain"] for x in [{"sub_domain": s, "count": n} for s, n in sub_top[:5]]],
                },
                "coherence": {
                    "centroid_mean": coh.centroid_mean,
                    "centroid_p50": coh.centroid_p50,
                    "pairwise_sample_mean": coh.pairwise_sample_mean,
                    "pairwise_sample_p50": coh.pairwise_sample_p50,
                },
                "exemplars": exemplars[:6],
            })

    # Save clusters + library
    write_jsonl(os.path.join(outdir, "clusters.jsonl"), cluster_summaries)
    write_jsonl(os.path.join(outdir, "cluster_library.jsonl"), cluster_library)

    # Zipf stats (exclude noise)
    sizes_desc = sorted([c["size"] for c in cluster_summaries if c["cluster_id"] != -1], reverse=True)
    topk_list = [int(x.strip()) for x in args.zipf_topk.split(",") if x.strip()]
    zipf = fit_zipf(sizes_desc, topk_list)

    print("Zipf:")
    print(f"  alpha (rank-size slope): {zipf.alpha}")
    print(f"  r2 (log-log fit): {zipf.r2}")
    print(f"  topk_share: {zipf.topk_share}")

    # Tiering (exclude noise)
    non_noise_clusters = [c for c in cluster_summaries if c["cluster_id"] != -1]
    # Sort by size desc for reporting
    non_noise_clusters.sort(key=lambda x: x["size"], reverse=True)

    A, B, C = assign_tiers(
        clusters=non_noise_clusters,
        size_A=args.tier_size_A,
        size_B=args.tier_size_B,
        coh_A=args.tier_coh_A,
        coh_B=args.tier_coh_B,
        coh_field="coherence_centroid_mean",
    )

    write_jsonl(os.path.join(outdir, "tier_A.jsonl"), A)
    write_jsonl(os.path.join(outdir, "tier_B.jsonl"), B)
    write_jsonl(os.path.join(outdir, "tier_C.jsonl"), C)

    tier_counts = {"A": len(A), "B": len(B), "C": len(C)}

    # Top-10 table (by size)
    top10 = non_noise_clusters[:10]

    report_md = build_report_md(
        total_patterns=len(patterns),
        n_clusters_ex_noise=len(cluster_ids),
        noise_count=noise_count,
        zipf=zipf,
        top10=top10,
        tier_counts=tier_counts,
    )
    write_text(os.path.join(outdir, "report.md"), report_md)

    print(f"Outputs written to: {outdir}/")


if __name__ == "__main__":
    main()
