"""
基于 Skeleton + Tricks 的混合聚类（AgglomerativeClustering + K-means）生成 Patterns。

与 `generate_patterns.py` 并行使用：
- `generate_patterns.py` 使用纯层次聚类；
- 本脚本在层次聚类基础上叠加 K-means 优化，进一步提升簇内紧凑度。

用法：
    cd scripts
    python generate_patterns_hybrid.py

输出文件仍写入上级目录的 `output/`，与原脚本保持一致，由用户自行选择采用哪一版结果。
"""

import os
import json
from collections import Counter

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

# 复用现有 generate_patterns 中的核心逻辑
from generate_patterns import (
    CLUSTER_PARAMS,
    load_all_papers,
    build_pattern_embeddings,
    analyze_cluster,
    generate_pattern_summary,
    assemble_pattern,
    generate_user_guide,
    generate_statistics,
    cluster_patterns,
)


def cluster_patterns_hybrid(embeddings: np.ndarray) -> np.ndarray:
    """先用层次聚类自适应确定簇数，再用 K-means 在此基础上细化优化。

    返回：
        labels_final: 每个样本的最终簇标签。
    """

    print("\n" + "-" * 80)
    print("阶段 1：层次聚类（确定簇数和初始结构）")
    print("-" * 80)

    # 复用原有的层次聚类实现，获得初始标签
    labels_agg = cluster_patterns(embeddings)

    # 计算簇数（保持与原脚本一致的处理方式）
    unique_labels = sorted(set(labels_agg))
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    print(f"\n✅ 层次聚类自动确定簇数 k = {n_clusters}")

    print("\n" + "-" * 80)
    print("阶段 2：K-means 细化（在层次结果上优化簇内紧凑度）")
    print("-" * 80)

    # 在进入 K-means 前对嵌入做 L2 归一化，以保证与余弦距离的一致性
    embeddings_norm = normalize(embeddings, norm="l2")

    # 基于层次聚类结果构造初始中心（分层初始化）
    initial_centers = []
    for cluster_id in range(n_clusters):
        cluster_indices = [i for i, lab in enumerate(labels_agg) if lab == cluster_id]
        if not cluster_indices:
            continue
        cluster_emb = embeddings_norm[cluster_indices]
        center = cluster_emb.mean(axis=0)
        initial_centers.append(center)

    if len(initial_centers) != n_clusters:
        # 理论上不应发生，仅作安全兜底
        print("⚠️ 初始中心数量与簇数不一致，退回使用 K-means++ 初始化。")
        kmeans = KMeans(
            n_clusters=n_clusters,
            init="k-means++",
            n_init=10,
            max_iter=300,
            random_state=42,
        )
    else:
        initial_centers = np.vstack(initial_centers)
        initial_centers = normalize(initial_centers, norm="l2")
        kmeans = KMeans(
            n_clusters=n_clusters,
            init=initial_centers,
            n_init=1,  # 初始化已由层次聚类提供，无需多次随机重启
            max_iter=300,
            random_state=42,
        )

    labels_final = kmeans.fit_predict(embeddings_norm)

    # 打印优化后的聚类概况
    print(f"\n✅ K-means 优化完成")
    print(f"   簇内平方和（inertia）: {kmeans.inertia_:.2f}")

    cluster_sizes = Counter(labels_final)
    for cid, size in sorted(cluster_sizes.items(), key=lambda x: -x[1]):
        print(f"   Cluster {cid}: {size} 篇")

    return labels_final


def main() -> None:
    """主流程：基于混合聚类生成 Patterns。"""

    print("=" * 80)
    print("基于 Skeleton + Tricks 的混合聚类（层次聚类 + K-means）生成 Patterns")
    print("=" * 80)

    # 1. 加载论文
    print("\n【Step 1】加载论文数据")
    papers = load_all_papers()
    print(f"✅ 共加载 {len(papers)} 篇论文")

    # 2. 构建 pattern embeddings
    print("\n【Step 2】构建 pattern embeddings")
    embeddings, pattern_data = build_pattern_embeddings(papers)
    print(f"✅ 完成 {len(embeddings)} 个 pattern 的 embedding")

    # 3. 混合聚类（层次聚类 + K-means）
    print("\n【Step 3】混合聚类（Agglomerative + K-means）")
    labels = cluster_patterns_hybrid(embeddings)

    # 4. 分析每个 cluster 并生成 pattern
    print("\n【Step 4】生成 patterns")
    unique_labels = sorted(set(labels))
    n_clusters = len(unique_labels)
    patterns = []

    for cluster_id in range(n_clusters):
        cluster_indices = [i for i, lab in enumerate(labels) if lab == cluster_id]

        if len(cluster_indices) < CLUSTER_PARAMS["min_cluster_size"]:
            print(f"  ⚠️  Cluster {cluster_id}: {len(cluster_indices)} 篇 (过小，跳过)")
            continue

        cluster_papers = [pattern_data[i] for i in cluster_indices]

        # 分析 cluster
        cluster_analysis = analyze_cluster(cluster_papers, cluster_id)

        # 生成 summary
        summary = generate_pattern_summary(cluster_analysis)
        print(f"    Summary: {summary[:80]}...")

        # 组装 pattern
        pattern = assemble_pattern(cluster_analysis, summary)
        patterns.append(pattern)

    print(f"\n✅ 共生成 {len(patterns)} 个 patterns")

    # 5. 生成输出文件（与原脚本保持同一目录结构，由用户自行选择采用哪一版）
    print("\n【Step 5】生成输出文件")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(os.path.dirname(script_dir), "output")
    os.makedirs(output_dir, exist_ok=True)

    # 5.1 结构化 JSON
    structured_path = os.path.join(output_dir, "patterns_structured.json")
    with open(structured_path, "w", encoding="utf-8") as f:
        json.dump(patterns, f, ensure_ascii=False, indent=2)
    print(f"  ✅ {os.path.basename(structured_path)}")

    # 5.2 用户指导文档
    guide_text = generate_user_guide(patterns)
    guide_path = os.path.join(output_dir, "patterns_guide.txt")
    with open(guide_path, "w", encoding="utf-8") as f:
        f.write(guide_text)
    print(f"  ✅ {os.path.basename(guide_path)}")

    # 5.3 统计报告
    statistics = generate_statistics(patterns)
    stats_path = os.path.join(output_dir, "patterns_statistics.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(statistics, f, ensure_ascii=False, indent=2)
    print(f"  ✅ {os.path.basename(stats_path)}")

    print("\n" + "=" * 80)
    print("🎉 混合聚类版本完成！")
    print("=" * 80)
    print(f"\n生成了 {len(patterns)} 个 patterns，覆盖 {statistics['total_papers']} 篇论文")
    print(f"平均每个 pattern 包含 {statistics['average_cluster_size']:.1f} 篇论文")
    print("\n提示：当前输出会覆盖 output 目录中的同名文件，请根据需要选择使用原版或混合版结果。")


if __name__ == "__main__":
    main()
