"""
简化的召回系统Demo - 单个测试用例

使用方法:
  python scripts/simple_recall_demo.py "你的Idea描述"

示例:
  python scripts/simple_recall_demo.py "使用Transformer进行文本分类"
"""

import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

# ===================== 路径配置 =====================
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / "output"

NODES_IDEA = OUTPUT_DIR / "nodes_idea.json"
NODES_PATTERN = OUTPUT_DIR / "nodes_pattern.json"
NODES_DOMAIN = OUTPUT_DIR / "nodes_domain.json"
NODES_PAPER = OUTPUT_DIR / "nodes_paper.json"
GRAPH_FILE = OUTPUT_DIR / "knowledge_graph_v2.gpickle"

# ===================== 配置参数 =====================
TOP_K_IDEAS = 10
TOP_K_DOMAINS = 5
TOP_K_PAPERS = 20
FINAL_TOP_K = 10

PATH1_WEIGHT = 0.4
PATH2_WEIGHT = 0.2
PATH3_WEIGHT = 0.4


# ===================== 工具函数 =====================
def compute_similarity(text1, text2):
    """基于字符的 Jaccard 相似度 (适配中文)"""
    if not text1 or not text2:
        return 0.0

    # 转换为字符集合
    tokens1 = set(text1.lower().replace(" ", ""))
    tokens2 = set(text2.lower().replace(" ", ""))

    if not tokens1 or not tokens2:
        return 0.0

    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)
    return intersection / union


# ===================== 主函数 =====================
def main():
    # 获取用户输入
    if len(sys.argv) > 1:
        user_idea = " ".join(sys.argv[1:])
    else:
        user_idea = "使用蒸馏技术完成Transformer跨领域文本分类任务，并在多个数据集上验证效果"

    print("=" * 80)
    print("🎯 三路召回系统 Demo")
    print("=" * 80)
    print(f"\n【用户Idea】\n{user_idea}\n")

    # 加载数据
    print("📂 加载数据...")
    with open(NODES_IDEA, 'r', encoding='utf-8') as f:
        ideas = json.load(f)
    with open(NODES_PATTERN, 'r', encoding='utf-8') as f:
        patterns = json.load(f)
    with open(NODES_DOMAIN, 'r', encoding='utf-8') as f:
        domains = json.load(f)
    with open(NODES_PAPER, 'r', encoding='utf-8') as f:
        papers = json.load(f)
    with open(GRAPH_FILE, 'rb') as f:
        G = pickle.load(f)

    # 构建索引
    idea_map = {i['idea_id']: i for i in ideas}
    pattern_map = {p['pattern_id']: p for p in patterns}
    domain_map = {d['domain_id']: d for d in domains}
    paper_map = {p['paper_id']: p for p in papers}

    print(f"  ✓ Idea: {len(ideas)}, Pattern: {len(patterns)}, Domain: {len(domains)}, Paper: {len(papers)}")
    print(f"  ✓ 图谱: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边\n")

    # ===================== 路径1: 相似Idea召回 =====================
    print("🔍 [路径1] 相似Idea召回...")

    similarities = []
    for idea in ideas:
        sim = compute_similarity(user_idea, idea['description'])
        if sim > 0:
            similarities.append((idea['idea_id'], sim))

    similarities.sort(key=lambda x: x[1], reverse=True)
    top_ideas = similarities[:TOP_K_IDEAS]

    print(f"  找到 {len(similarities)} 个相似Idea，选择 Top-{TOP_K_IDEAS}")

    path1_scores = defaultdict(float)
    for idea_id, similarity in top_ideas:
        idea = idea_map[idea_id]
        # 打印匹配到的相似 Idea 辅助调试
        if similarity > 0.2:
            print(f"    - 匹配 Idea: {idea['description'][:40]}... (sim={similarity:.3f})")

        # 路径 1 直接从 Idea 节点的 pattern_ids 召回
        pattern_ids = idea.get('pattern_ids', [])
        for pid in pattern_ids:
            path1_scores[pid] += similarity

    print(f"  ✓ 召回 {len(path1_scores)} 个Pattern\n")

    # ===================== 路径2: 领域相关召回 =====================
    print("🌍 [路径2] 领域相关性召回...")

    # 通过最相似Idea的Domain
    top_idea = idea_map[top_ideas[0][0]] if top_ideas else None
    domain_scores = []

    if top_idea and G.has_node(top_idea['idea_id']):
        for successor in G.successors(top_idea['idea_id']):
            edge_data = G[top_idea['idea_id']][successor]
            if edge_data.get('relation') == 'belongs_to':
                domain_id = successor
                weight = edge_data.get('weight', 0.5)
                domain_scores.append((domain_id, weight))

    domain_scores.sort(key=lambda x: x[1], reverse=True)
    top_domains = domain_scores[:TOP_K_DOMAINS]

    print(f"  找到 {len(domain_scores)} 个相关Domain，选择 Top-{TOP_K_DOMAINS}")

    path2_scores = defaultdict(float)
    for domain_id, domain_weight in top_domains:
        for predecessor in G.predecessors(domain_id):
            edge_data = G[predecessor][domain_id]
            if edge_data.get('relation') == 'works_well_in':
                pattern_id = predecessor
                effectiveness = edge_data.get('effectiveness', 0.0)
                confidence = edge_data.get('confidence', 0.0)
                path2_scores[pattern_id] += domain_weight * max(effectiveness, 0.1) * confidence

    print(f"  ✓ 召回 {len(path2_scores)} 个Pattern\n")

    # ===================== 路径3: 相似Paper召回 =====================
    print("📄 [路径3] 相似Paper召回...")

    similarities = []
    for paper in papers:
        # 尝试多个可能的 Idea 描述字段
        paper_idea = paper.get('idea', {}).get('core_idea', '') or paper.get('abstract', '')[:100]
        if not paper_idea:
            continue

        sim = compute_similarity(user_idea, paper_idea)
        if sim > 0.1 and G.has_node(paper['paper_id']):
            reviews = paper.get('reviews', [])
            if reviews:
                import numpy as np
                scores = [r.get('rating', 5) for r in reviews]
                avg_score = np.mean(scores)
                quality = (avg_score - 1) / 9
            else:
                quality = 0.5

            combined = sim * quality
            similarities.append((paper['paper_id'], sim, quality, combined))

    similarities.sort(key=lambda x: x[3], reverse=True)
    top_papers = similarities[:TOP_K_PAPERS]

    print(f"  找到 {len(similarities)} 个相似Paper，选择 Top-{TOP_K_PAPERS}")

    path3_scores = defaultdict(float)
    for paper_id, similarity, quality, combined_weight in top_papers:
        if not G.has_node(paper_id):
            continue
        for successor in G.successors(paper_id):
            edge_data = G[paper_id][successor]
            if edge_data.get('relation') == 'uses_pattern':
                pattern_id = successor
                pattern_quality = edge_data.get('quality', 0.5)
                path3_scores[pattern_id] += combined_weight * pattern_quality

    print(f"  ✓ 召回 {len(path3_scores)} 个Pattern\n")

    # ===================== 融合结果 =====================
    print("🔗 融合三路召回结果...\n")

    all_patterns = set(path1_scores.keys()) | set(path2_scores.keys()) | set(path3_scores.keys())

    final_scores = {}
    for pattern_id in all_patterns:
        score1 = path1_scores.get(pattern_id, 0.0) * PATH1_WEIGHT
        score2 = path2_scores.get(pattern_id, 0.0) * PATH2_WEIGHT
        score3 = path3_scores.get(pattern_id, 0.0) * PATH3_WEIGHT
        final_scores[pattern_id] = score1 + score2 + score3

    ranked = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    top_k = ranked[:FINAL_TOP_K]

    # ===================== 输出结果 =====================
    print("=" * 80)
    print(f"📊 召回结果 Top-{FINAL_TOP_K}")
    print("=" * 80)

    for rank, (pattern_id, final_score) in enumerate(top_k, 1):
        pattern_info = pattern_map.get(pattern_id, {})

        score1 = path1_scores.get(pattern_id, 0.0) * PATH1_WEIGHT
        score2 = path2_scores.get(pattern_id, 0.0) * PATH2_WEIGHT
        score3 = path3_scores.get(pattern_id, 0.0) * PATH3_WEIGHT

        print(f"\n【Rank {rank}】 {pattern_id}")
        print(f"  名称: {pattern_info.get('name', 'N/A')}")
        print(f"  最终得分: {final_score:.4f}")

        if final_score > 0:
            print(f"  - 路径1 (相似Idea):   {score1:.4f} (占比 {score1/final_score*100:.1f}%)")
            print(f"  - 路径2 (领域相关):   {score2:.4f} (占比 {score2/final_score*100:.1f}%)")
            print(f"  - 路径3 (相似Paper):  {score3:.4f} (占比 {score3/final_score*100:.1f}%)")

        print(f"  聚类大小: {pattern_info.get('cluster_size', 0)} 篇论文")

        summary = pattern_info.get('summary', 'N/A')
        print(f"  摘要: {summary[:120]}...")

    print("\n" + "=" * 80)
    print("✅ 召回完成!")
    print("=" * 80)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()

