"""
简化的召回系统Demo - 单个测试用例 (V3版本)

使用方法:
  python scripts/simple_recall_demo.py "你的Idea描述"

示例:
  python scripts/simple_recall_demo.py "使用Transformer进行文本分类"

V3版本更新:
  - 适配V3节点结构 (Paper.idea为字符串，非嵌套字典)
  - 路径1直接使用Idea.pattern_ids，无需通过Paper中转
  - Paper通过review_stats获取质量分数，支持兼容旧结构
"""

import json
import os
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

# 提前加载 .env（确保配置读取前生效）
SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = SCRIPTS_DIR.parent
REPO_ROOT = PROJECT_ROOT.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

try:
    from idea2paper.infra.dotenv import load_dotenv
    _DOTENV_STATUS = load_dotenv(REPO_ROOT / ".env", override=False)
except Exception:
    _DOTENV_STATUS = None

import numpy as np
import requests

from pipeline.run_context import get_logger
# ===================== 路径配置 =====================
OUTPUT_DIR = PROJECT_ROOT / "output"

NODES_IDEA = OUTPUT_DIR / "nodes_idea.json"
NODES_PATTERN = OUTPUT_DIR / "nodes_pattern.json"
NODES_DOMAIN = OUTPUT_DIR / "nodes_domain.json"
NODES_PAPER = OUTPUT_DIR / "nodes_paper.json"
GRAPH_FILE = OUTPUT_DIR / "knowledge_graph_v2.gpickle"

# ===================== 配置参数 =====================
TOP_K_IDEAS = 10
TOP_K_PATTERNS_PATH1 = 10  # 路径1最终保留Top-K个Pattern（重要通道）

TOP_K_DOMAINS = 5
TOP_K_PATTERNS_PATH2 = 5   # 路径2最终保留Top-K个Pattern（辅助通道）

TOP_K_PAPERS = 20
TOP_K_PATTERNS_PATH3 = 10  # 路径3最终保留Top-K个Pattern（重要通道）

FINAL_TOP_K = 10

PATH1_WEIGHT = 0.4  # 相似Idea - 重要
PATH2_WEIGHT = 0.2  # 领域相关 - 辅助
PATH3_WEIGHT = 0.4  # 相似Paper - 重要

USE_EMBEDDING = True  # 使用embedding计算相似度（推荐）

# 两阶段召回优化（粗排+精排）
TWO_STAGE_RECALL = True      # 启用两阶段召回（大幅提速）
COARSE_RECALL_SIZE = 100     # 粗召回数量（Jaccard快速筛选）


# ===================== 工具函数 =====================
def compute_similarity(text1, text2):
    """计算两个文本的相似度"""
    if USE_EMBEDDING:
        return compute_embedding_similarity(text1, text2)
    else:
        return compute_jaccard_similarity(text1, text2)

def compute_jaccard_similarity(text1, text2):
    """词袋Jaccard相似度（快速但不准确）"""
    tokens1 = set(text1.lower().split())
    tokens2 = set(text2.lower().split())

    if not tokens1 or not tokens2:
        return 0.0

    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)
    return intersection / union

def compute_embedding_similarity(text1, text2):
    """基于embedding的余弦相似度（更准确）"""
    emb1 = get_embedding(text1)
    emb2 = get_embedding(text2)

    if emb1 is None or emb2 is None:
        return compute_jaccard_similarity(text1, text2)

    emb1 = np.array(emb1)
    emb2 = np.array(emb2)

    cosine_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return float(cosine_sim)

from idea2paper.infra.embeddings import get_embedding as infra_get_embedding

_embedding_cache = {}

def get_embedding(text, max_retries=3):
    """调用统一的Embedding API接口"""
    # 缓存检查
    if text in _embedding_cache:
        return _embedding_cache[text]

    embedding = infra_get_embedding(text, timeout=60)
    
    if embedding:
        _embedding_cache[text] = embedding
    
    return embedding


def get_paper_quality(paper):
    """计算Paper的综合质量分数

    基于review的评分，归一化到[0, 1]
    如果没有review数据，返回默认值0.5
    """
    # 优先使用新结构中的 review_stats.avg_score
    review_stats = paper.get('review_stats', {})

    if review_stats and review_stats.get('avg_score'):
        # 已经是 0-1 的分数
        return float(review_stats['avg_score'])

    # 备选方案：兼容旧结构（review 列表）
    reviews = paper.get('reviews', [])

    if not reviews:
        return 0.5  # 默认中等质量

    # 提取所有评分
    scores = []
    for review in reviews:
        score_str = review.get('overall_score', '')
        # 尝试解析评分（可能是 "7", "7/10", "7.0" 等格式）
        try:
            if '/' in score_str:
                score_str = score_str.split('/')[0]
            score = float(score_str.strip())
            scores.append(score)
        except (ValueError, AttributeError):
            continue

    if not scores:
        return 0.5

    # 计算平均分并归一化
    import numpy as np
    avg_score = np.mean(scores)
    # 假设评分范围是 1-10，归一化到 [0, 1]
    normalized_score = (avg_score - 1) / 9

    return min(max(normalized_score, 0.0), 1.0)


# ===================== 主函数 =====================
def main():
    # 获取用户输入
    if len(sys.argv) > 1:
        user_idea = " ".join(sys.argv[1:])
    else:
        # user_idea = "使用蒸馏技术完成Transformer跨领域文本分类任务，并在多个数据集上验证效果"
        user_idea = "Research on the Self-Evolution of Intelligent Agents Based on Reflection and Memory"

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

    # 两阶段召回优化
    if TWO_STAGE_RECALL and USE_EMBEDDING:
        print(f"  [粗排] 使用Jaccard快速筛选Top-{COARSE_RECALL_SIZE}...")
        coarse_similarities = []
        for idea in ideas:
            sim = compute_jaccard_similarity(user_idea, idea['description'])
            if sim > 0:
                coarse_similarities.append((idea['idea_id'], sim))

        coarse_similarities.sort(key=lambda x: x[1], reverse=True)
        candidates = coarse_similarities[:COARSE_RECALL_SIZE]

        print(f"  [精排] 使用Embedding重排Top-{TOP_K_IDEAS}...")
        fine_similarities = []
        for idea_id, _ in candidates:
            idea = idea_map[idea_id]
            sim = compute_embedding_similarity(user_idea, idea['description'])
            if sim > 0:
                fine_similarities.append((idea_id, sim))

        fine_similarities.sort(key=lambda x: x[1], reverse=True)
        top_ideas = fine_similarities[:TOP_K_IDEAS]

        print(f"  ✓ 粗排{len(coarse_similarities)}个 → 精排{len(candidates)}个 → 最终{len(top_ideas)}个")
    else:
        # 单阶段召回（原逻辑）
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
        # 打印匹配到的相似 Idea 辅助调试(增加到300字符)
        if similarity > 0.2:
            print(f"    - 匹配 Idea [{idea_id}]: {idea['description'][:300]}... (sim={similarity:.3f})")

        # 路径 1 直接从 Idea 节点的 pattern_ids 召回
        pattern_ids = idea.get('pattern_ids', [])
        for pid in pattern_ids:
            path1_scores[pid] += similarity

    # 排序并只保留Top-K个Pattern
    sorted_path1 = sorted(path1_scores.items(), key=lambda x: x[1], reverse=True)
    path1_scores = dict(sorted_path1[:TOP_K_PATTERNS_PATH1])

    print(f"  ✓ 召回 {len(sorted_path1)} 个Pattern，保留Top-{TOP_K_PATTERNS_PATH1}\n")

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
        # 打印Domain详细信息
        domain = domain_map.get(domain_id)
        if domain:
            domain_name = domain.get('name', 'N/A')
            paper_count = domain.get('paper_count', 0)
            sub_domains = domain.get('sub_domains', [])
            sub_domain_str = ', '.join(sub_domains[:5])  # 只显示前5个sub_domain
            if len(sub_domains) > 5:
                sub_domain_str += f"... (共{len(sub_domains)}个)"

            print(f"  - {domain_id} (名称={domain_name}, 相关度={domain_weight:.3f}, 论文数={paper_count})")
            if sub_domain_str:
                print(f"    子领域: {sub_domain_str}")

        for predecessor in G.predecessors(domain_id):
            edge_data = G[predecessor][domain_id]
            if edge_data.get('relation') == 'works_well_in':
                pattern_id = predecessor
                effectiveness = edge_data.get('effectiveness', 0.0)
                confidence = edge_data.get('confidence', 0.0)
                path2_scores[pattern_id] += domain_weight * max(effectiveness, 0.1) * confidence

    # 排序并只保留Top-K个Pattern
    sorted_path2 = sorted(path2_scores.items(), key=lambda x: x[1], reverse=True)
    path2_scores = dict(sorted_path2[:TOP_K_PATTERNS_PATH2])

    print(f"  ✓ 召回 {len(sorted_path2)} 个Pattern，保留Top-{TOP_K_PATTERNS_PATH2}\n")

    # ===================== 路径3: 相似Paper召回 =====================
    print("📄 [路径3] 相似Paper召回...")

    # 两阶段召回优化
    if TWO_STAGE_RECALL and USE_EMBEDDING:
        print(f"  [粗排] 使用Jaccard快速筛选Top-{COARSE_RECALL_SIZE}...")
        coarse_similarities = []
        for paper in tqdm(papers, desc="Processing papers"):
            paper_title = paper.get('title', '')
            if not paper_title:
                continue

            sim = compute_jaccard_similarity(user_idea, paper_title)
            if sim > 0.05:  # 降低阈值
                coarse_similarities.append((paper['paper_id'], sim))

        coarse_similarities.sort(key=lambda x: x[1], reverse=True)
        candidates = coarse_similarities[:COARSE_RECALL_SIZE]

        print(f"  [精排] 使用Embedding重排Top-{TOP_K_PAPERS}...")
        fine_similarities = []
        for paper_id, _ in candidates:
            paper = paper_map.get(paper_id)
            if not paper:
                continue
            paper_title = paper.get('title', '')

            sim = compute_embedding_similarity(user_idea, paper_title)
            if sim > 0.1 and G.has_node(paper_id):
                quality = get_paper_quality(paper)
                combined = sim * quality
                fine_similarities.append((paper_id, sim, quality, combined))

        fine_similarities.sort(key=lambda x: x[3], reverse=True)
        top_papers = fine_similarities[:TOP_K_PAPERS]

        print(f"  ✓ 粗排{len(coarse_similarities)}个 → 精排{len(candidates)}个 → 最终{len(top_papers)}个")
    else:
        # 单阶段召回（原逻辑）
        similarities = []
        for paper in tqdm(papers, desc="Processing papers"):
            paper_title = paper.get('title', '')
            if not paper_title:
                continue

            sim = compute_similarity(user_idea, paper_title)
            if sim > 0.1 and G.has_node(paper['paper_id']):
                quality = get_paper_quality(paper)
                combined = sim * quality
                similarities.append((paper['paper_id'], sim, quality, combined))

        similarities.sort(key=lambda x: x[3], reverse=True)
        top_papers = similarities[:TOP_K_PAPERS]

        print(f"  找到 {len(similarities)} 个相似Paper，选择 Top-{TOP_K_PAPERS}")

    path3_scores = defaultdict(float)
    for paper_id, similarity, quality, combined_weight in top_papers:
        paper = paper_map.get(paper_id, {})
        # 判断质量来源：优先检查review_stats，然后是reviews，否则是默认值
        if paper.get('review_stats'):
            quality_source = f"review({paper['review_stats'].get('review_count', 0)}条)"
        elif paper.get('reviews'):
            quality_source = "review"
        else:
            quality_source = "默认"
        title = paper.get('title', 'N/A')
        print(f"  - {paper_id} (相似度={similarity:.3f}, 质量={quality:.3f} [{quality_source}])")
        print(f"    标题: {title}")

        if not G.has_node(paper_id):
            continue
        for successor in G.successors(paper_id):
            edge_data = G[paper_id][successor]
            if edge_data.get('relation') == 'uses_pattern':
                pattern_id = successor
                pattern_quality = edge_data.get('quality', 0.5)
                path3_scores[pattern_id] += combined_weight * pattern_quality

    # 排序并只保留Top-K个Pattern
    sorted_path3 = sorted(path3_scores.items(), key=lambda x: x[1], reverse=True)
    path3_scores = dict(sorted_path3[:TOP_K_PATTERNS_PATH3])

    print(f"  ✓ 召回 {len(sorted_path3)} 个Pattern，保留Top-{TOP_K_PATTERNS_PATH3}\n")

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

        print(f"  聚类大小: {pattern_info.get('size', 0)} 篇论文")

        # V3版本: 优先显示LLM增强的总结，否则显示原始示例
        if pattern_info.get('llm_enhanced_summary'):
            llm_summary = pattern_info['llm_enhanced_summary'].get('representative_ideas', '')
            print(f"  归纳总结: {llm_summary[:120]}...")
        else:
            summary = pattern_info.get('summary', {})
            ideas = summary.get('representative_ideas', [])
            if ideas:
                print(f"  示例Idea: {ideas[0][:120] if ideas else 'N/A'}...")

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
