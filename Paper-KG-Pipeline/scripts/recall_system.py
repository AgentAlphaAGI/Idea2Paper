"""
三路召回系统 Demo - Idea2Pattern

基于知识图谱的三路召回策略：
  路径1: Idea → Idea → Pattern (相似Idea召回)
  路径2: Idea → Domain → Pattern (领域相关性召回)
  路径3: Idea → Paper → Pattern (相似Paper召回)
"""

import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ===================== 配置 =====================
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / "output"

# 输入文件
NODES_IDEA = OUTPUT_DIR / "nodes_idea.json"
NODES_PATTERN = OUTPUT_DIR / "nodes_pattern.json"
NODES_DOMAIN = OUTPUT_DIR / "nodes_domain.json"
NODES_PAPER = OUTPUT_DIR / "nodes_paper.json"
EDGES_FILE = OUTPUT_DIR / "edges.json"
GRAPH_FILE = OUTPUT_DIR / "knowledge_graph_v2.gpickle"


# ===================== 召回参数配置 =====================
class RecallConfig:
    """召回系统配置"""
    # 每路召回的Top-K
    PATH1_TOP_K_IDEAS = 10       # 路径1: 召回前K个最相似的Idea
    PATH1_TOP_K_PATTERNS = 5     # 路径1: 每个Idea最多保留K个Pattern

    PATH2_TOP_K_DOMAINS = 5      # 路径2: 召回前K个最相关的Domain
    PATH2_TOP_K_PATTERNS = 10    # 路径2: 每个Domain最多保留K个Pattern

    PATH3_TOP_K_PAPERS = 20      # 路径3: 召回前K个最相似的Paper
    PATH3_TOP_K_PATTERNS = 8     # 路径3: 每个Paper最多保留K个Pattern

    # 各路召回的权重
    PATH1_WEIGHT = 0.4  # 路径1权重
    PATH2_WEIGHT = 0.3  # 路径2权重
    PATH3_WEIGHT = 0.3  # 路径3权重

    # 最终召回的Top-K
    FINAL_TOP_K = 10


# ===================== 召回系统 =====================
class RecallSystem:
    """三路召回系统"""

    def __init__(self):
        print("🚀 初始化召回系统...")

        # 加载数据
        self.ideas = self._load_json(NODES_IDEA)
        self.patterns = self._load_json(NODES_PATTERN)
        self.domains = self._load_json(NODES_DOMAIN)
        self.papers = self._load_json(NODES_PAPER)

        # 加载图谱
        with open(GRAPH_FILE, 'rb') as f:
            self.G = pickle.load(f)

        # 构建索引
        self.idea_id_to_idea = {i['idea_id']: i for i in self.ideas}
        self.pattern_id_to_pattern = {p['pattern_id']: p for p in self.patterns}
        self.domain_id_to_domain = {d['domain_id']: d for d in self.domains}
        self.paper_id_to_paper = {p['paper_id']: p for p in self.papers}

        print(f"  ✓ 加载 {len(self.ideas)} 个Idea")
        print(f"  ✓ 加载 {len(self.patterns)} 个Pattern")
        print(f"  ✓ 加载 {len(self.domains)} 个Domain")
        print(f"  ✓ 加载 {len(self.papers)} 个Paper")
        print(f"  ✓ 图谱节点: {self.G.number_of_nodes()}, 边: {self.G.number_of_edges()}")
        print()

    def _load_json(self, filepath: Path) -> List[Dict]:
        """加载JSON文件"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _compute_text_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度（Jaccard）"""
        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())

        if not tokens1 or not tokens2:
            return 0.0

        intersection = tokens1 & tokens2
        union = tokens1 | tokens2

        return len(intersection) / len(union)

    # ===================== 路径1: Idea → Idea → Pattern =====================

    def _recall_path1_similar_ideas(self, user_idea: str) -> Dict[str, float]:
        """路径1: 通过相似Idea召回Pattern

        流程:
          1. 计算用户Idea与图谱中所有Idea的相似度
          2. 选择Top-K最相似的Idea
          3. 收集这些Idea关联的Pattern
          4. 按相似度加权计算Pattern得分

        返回: {pattern_id: score}
        """
        print("\n🔍 [路径1] 相似Idea召回...")

        # Step 1: 计算与所有Idea的相似度
        similarities = []
        for idea in self.ideas:
            sim = self._compute_text_similarity(user_idea, idea['description'])
            if sim > 0:
                similarities.append((idea['idea_id'], sim))

        # Step 2: 排序并选择Top-K
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_ideas = similarities[:RecallConfig.PATH1_TOP_K_IDEAS]

        print(f"  找到 {len(similarities)} 个相似Idea，选择Top-{RecallConfig.PATH1_TOP_K_IDEAS}")

        # Step 3: 收集Pattern并计算得分
        pattern_scores = defaultdict(float)

        for idea_id, similarity in top_ideas:
            idea = self.idea_id_to_idea[idea_id]
            pattern_ids = idea.get('pattern_ids', [])

            print(f"  - {idea_id} (相似度={similarity:.3f}): {len(pattern_ids)} 个Pattern")

            # 从图谱中找到这个Idea的所有Pattern（通过Paper中转）
            for paper_id in idea.get('source_paper_ids', []):
                if not self.G.has_node(paper_id):
                    continue

                # 找到Paper使用的Pattern
                for successor in self.G.successors(paper_id):
                    edge_data = self.G[paper_id][successor]
                    if edge_data.get('relation') == 'uses_pattern':
                        pattern_id = successor
                        quality = edge_data.get('quality', 0.5)

                        # 得分 = 相似度 × Paper质量
                        score = similarity * quality
                        pattern_scores[pattern_id] += score

        print(f"  ✓ 召回 {len(pattern_scores)} 个Pattern")
        return dict(pattern_scores)

    # ===================== 路径2: Idea → Domain → Pattern =====================

    def _recall_path2_domain_patterns(self, user_idea: str) -> Dict[str, float]:
        """路径2: 通过领域相关性召回Pattern

        流程:
          1. 找到与用户Idea最相关的Domain（基于关键词匹配）
          2. 在这些Domain中找到表现好的Pattern
          3. 按Domain相关性和Pattern效果加权计算得分

        返回: {pattern_id: score}
        """
        print("\n🌍 [路径2] 领域相关性召回...")

        # Step 1: 找到相关Domain（基于关键词匹配）
        domain_scores = []
        user_tokens = set(user_idea.lower().split())

        for domain in self.domains:
            domain_name = domain['name']
            domain_tokens = set(domain_name.lower().split())

            # 简单的关键词匹配
            match_score = len(user_tokens & domain_tokens) / max(len(user_tokens), 1)

            if match_score > 0:
                domain_scores.append((domain['domain_id'], match_score))

        # 如果没有匹配的Domain，使用最相似Idea的Domain
        if not domain_scores:
            print("  未找到直接匹配的Domain，使用相似Idea的Domain...")
            similarities = []
            for idea in self.ideas:
                sim = self._compute_text_similarity(user_idea, idea['description'])
                if sim > 0:
                    similarities.append((idea, sim))

            similarities.sort(key=lambda x: x[1], reverse=True)
            top_idea = similarities[0][0] if similarities else None

            if top_idea:
                # 通过图谱找到Idea的Domain
                for successor in self.G.successors(top_idea['idea_id']):
                    edge_data = self.G[top_idea['idea_id']][successor]
                    if edge_data.get('relation') == 'belongs_to':
                        domain_id = successor
                        weight = edge_data.get('weight', 0.5)
                        domain_scores.append((domain_id, weight))

        # Step 2: 排序并选择Top-K Domain
        domain_scores.sort(key=lambda x: x[1], reverse=True)
        top_domains = domain_scores[:RecallConfig.PATH2_TOP_K_DOMAINS]

        print(f"  找到 {len(domain_scores)} 个相关Domain，选择Top-{RecallConfig.PATH2_TOP_K_DOMAINS}")

        # Step 3: 从这些Domain中找Pattern
        pattern_scores = defaultdict(float)

        for domain_id, domain_weight in top_domains:
            domain = self.domain_id_to_domain.get(domain_id)
            if not domain:
                continue

            print(f"  - {domain_id} ({domain['name']}, 相关度={domain_weight:.3f})")

            # 找到在该Domain中表现好的Pattern
            for predecessor in self.G.predecessors(domain_id):
                edge_data = self.G[predecessor][domain_id]
                if edge_data.get('relation') == 'works_well_in':
                    pattern_id = predecessor
                    effectiveness = edge_data.get('effectiveness', 0.0)
                    confidence = edge_data.get('confidence', 0.0)

                    # 得分 = Domain相关度 × 效果 × 置信度
                    score = domain_weight * max(effectiveness, 0.1) * confidence
                    pattern_scores[pattern_id] += score

        print(f"  ✓ 召回 {len(pattern_scores)} 个Pattern")
        return dict(pattern_scores)

    # ===================== 路径3: Idea → Paper → Pattern =====================

    def _recall_path3_similar_papers(self, user_idea: str) -> Dict[str, float]:
        """路径3: 通过相似Paper召回Pattern

        流程:
          1. 找到与用户Idea最相似的Paper（基于core_idea）
          2. 收集这些Paper使用的Pattern
          3. 按Paper相似度和质量加权计算得分

        返回: {pattern_id: score}
        """
        print("\n📄 [路径3] 相似Paper召回...")

        # Step 1: 计算与所有Paper的相似度
        similarities = []

        for paper in self.papers:
            paper_idea = paper.get('idea', {}).get('core_idea', '')
            if not paper_idea:
                continue

            sim = self._compute_text_similarity(user_idea, paper_idea)
            if sim > 0.1:  # 过滤低相似度
                # 从图谱中获取Paper质量
                if self.G.has_node(paper['paper_id']):
                    # 计算质量（基于Review评分）
                    reviews = paper.get('reviews', [])
                    if reviews:
                        scores = [r.get('rating', 5) for r in reviews]
                        avg_score = np.mean(scores)
                        quality = (avg_score - 1) / 9  # 归一化到[0,1]
                    else:
                        quality = 0.5

                    combined_weight = sim * quality
                    similarities.append((paper['paper_id'], sim, quality, combined_weight))

        # Step 2: 排序并选择Top-K
        similarities.sort(key=lambda x: x[3], reverse=True)
        top_papers = similarities[:RecallConfig.PATH3_TOP_K_PAPERS]

        print(f"  找到 {len(similarities)} 个相似Paper，选择Top-{RecallConfig.PATH3_TOP_K_PAPERS}")

        # Step 3: 收集Pattern
        pattern_scores = defaultdict(float)

        for paper_id, similarity, quality, combined_weight in top_papers:
            print(f"  - {paper_id} (相似度={similarity:.3f}, 质量={quality:.3f})")

            # 从图谱中找到Paper使用的Pattern
            if not self.G.has_node(paper_id):
                continue

            for successor in self.G.successors(paper_id):
                edge_data = self.G[paper_id][successor]
                if edge_data.get('relation') == 'uses_pattern':
                    pattern_id = successor
                    pattern_quality = edge_data.get('quality', 0.5)

                    # 得分 = Paper相似度 × Paper质量 × Pattern质量
                    score = combined_weight * pattern_quality
                    pattern_scores[pattern_id] += score

        print(f"  ✓ 召回 {len(pattern_scores)} 个Pattern")
        return dict(pattern_scores)

    # ===================== 多路融合 =====================

    def recall(self, user_idea: str, verbose: bool = True) -> List[Tuple[str, Dict, float]]:
        """三路召回融合

        Args:
            user_idea: 用户输入的Idea描述
            verbose: 是否打印详细信息

        Returns:
            [(pattern_id, pattern_info, score), ...] 按得分排序
        """
        print("=" * 80)
        print("🎯 开始三路召回")
        print("=" * 80)
        print(f"\n【用户Idea】\n{user_idea}\n")

        # 路径1: 相似Idea召回
        path1_scores = self._recall_path1_similar_ideas(user_idea)

        # 路径2: 领域相关性召回
        path2_scores = self._recall_path2_domain_patterns(user_idea)

        # 路径3: 相似Paper召回
        path3_scores = self._recall_path3_similar_papers(user_idea)

        # 融合三路得分
        print("\n🔗 融合三路召回结果...")
        all_patterns = set(path1_scores.keys()) | set(path2_scores.keys()) | set(path3_scores.keys())

        final_scores = {}
        for pattern_id in all_patterns:
            score1 = path1_scores.get(pattern_id, 0.0) * RecallConfig.PATH1_WEIGHT
            score2 = path2_scores.get(pattern_id, 0.0) * RecallConfig.PATH2_WEIGHT
            score3 = path3_scores.get(pattern_id, 0.0) * RecallConfig.PATH3_WEIGHT

            final_scores[pattern_id] = score1 + score2 + score3

        # 排序并返回Top-K
        ranked = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        top_k = ranked[:RecallConfig.FINAL_TOP_K]

        # 构建返回结果
        results = []
        for pattern_id, score in top_k:
            pattern_info = self.pattern_id_to_pattern.get(pattern_id, {})
            results.append((pattern_id, pattern_info, score))

        # 打印结果
        if verbose:
            self._print_results(results, path1_scores, path2_scores, path3_scores)

        return results

    def _print_results(self, results: List[Tuple[str, Dict, float]],
                      path1_scores: Dict, path2_scores: Dict, path3_scores: Dict):
        """打印召回结果"""
        print("\n" + "=" * 80)
        print(f"📊 召回结果 Top-{RecallConfig.FINAL_TOP_K}")
        print("=" * 80)

        for rank, (pattern_id, pattern_info, final_score) in enumerate(results, 1):
            print(f"\n【Rank {rank}】 {pattern_id}")
            print(f"  名称: {pattern_info.get('name', 'N/A')}")
            print(f"  最终得分: {final_score:.4f}")

            # 显示各路得分
            score1 = path1_scores.get(pattern_id, 0.0) * RecallConfig.PATH1_WEIGHT
            score2 = path2_scores.get(pattern_id, 0.0) * RecallConfig.PATH2_WEIGHT
            score3 = path3_scores.get(pattern_id, 0.0) * RecallConfig.PATH3_WEIGHT

            print(f"  - 路径1 (相似Idea):   {score1:.4f} (占比 {score1/final_score*100:.1f}%)")
            print(f"  - 路径2 (领域相关):   {score2:.4f} (占比 {score2/final_score*100:.1f}%)")
            print(f"  - 路径3 (相似Paper):  {score3:.4f} (占比 {score3/final_score*100:.1f}%)")

            print(f"  聚类大小: {pattern_info.get('cluster_size', 0)} 篇论文")
            print(f"  摘要: {pattern_info.get('summary', 'N/A')[:100]}...")

        print("\n" + "=" * 80)


# ===================== Demo 测试用例 =====================
def demo():
    """运行Demo"""

    # 初始化召回系统
    system = RecallSystem()

    # 测试用例
    test_ideas = [
        "使用Transformer模型进行文本分类任务，在多个数据集上验证效果",
        "提出一种新的注意力机制改进神经机器翻译的对齐质量",
        "通过对抗训练提升模型在对话系统中的鲁棒性",
        "利用知识图谱增强预训练语言模型的语义理解能力",
    ]

    for i, user_idea in enumerate(test_ideas, 1):
        print("\n\n")
        print("🎬" * 40)
        print(f"测试用例 {i}/{len(test_ideas)}")
        print("🎬" * 40)

        results = system.recall(user_idea, verbose=True)

        # 等待用户查看结果
        if i < len(test_ideas):
            input("\n按Enter继续下一个测试用例...")


if __name__ == '__main__':
    demo()

