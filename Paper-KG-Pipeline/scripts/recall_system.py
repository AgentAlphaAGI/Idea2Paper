"""
三路召回系统 Demo - Idea2Pattern (V3版本)

基于知识图谱的三路召回策略：
  路径1: Idea → Idea → Pattern (相似Idea召回)
  路径2: Idea → Domain → Pattern (领域相关性召回)
  路径3: Idea → Paper → Pattern (相似Paper召回)

V3版本更新:
  - 适配V3节点结构 (Paper.idea为字符串，非嵌套字典)
  - 路径1直接使用Idea.pattern_ids，无需通过Paper中转
  - Paper暂无review数据，质量默认0.5
"""

import json
import os
import pickle
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import requests

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
    PATH1_TOP_K_IDEAS = 20       # 路径1: 召回前K个最相似的Idea
    PATH1_FINAL_TOP_K = 10       # 路径1: 最终只保留Top-K个Pattern（重要通道）

    PATH2_TOP_K_DOMAINS = 5      # 路径2: 召回前K个最相关的Domain
    PATH2_FINAL_TOP_K = 5        # 路径2: 最终只保留Top-K个Pattern（辅助通道）

    PATH3_TOP_K_PAPERS = 20      # 路径3: 召回前K个最相似的Paper
    PATH3_FINAL_TOP_K = 10       # 路径3: 最终只保留Top-K个Pattern（重要通道）

    # 各路召回的权重
    PATH1_WEIGHT = 0.4  # 路径1权重（相似Idea - 重要）
    PATH2_WEIGHT = 0.2  # 路径2权重（领域相关 - 辅助）
    PATH3_WEIGHT = 0.4  # 路径3权重（相似Paper - 重要）

    # 最终召回的Top-K
    FINAL_TOP_K = 10

    # 相似度计算方式
    USE_EMBEDDING = True  # 使用embedding计算相似度（推荐），False则使用Jaccard

    # 两阶段召回优化（粗排+精排）
    TWO_STAGE_RECALL = True      # 启用两阶段召回（大幅提速）
    COARSE_RECALL_SIZE = 100     # 粗召回数量（Jaccard快速筛选）
    FINE_RECALL_SIZE = 20        # 精排数量（Embedding精确排序）


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
        """计算两个文本的相似度

        支持两种模式:
        1. USE_EMBEDDING=True: 使用Qwen3-Embedding-4B计算语义相似度（推荐）
        2. USE_EMBEDDING=False: 使用词袋Jaccard相似度（快速但不准确）
        """
        if not text1 or not text2:
            return 0.0

        if RecallConfig.USE_EMBEDDING:
            return self._compute_embedding_similarity(text1, text2)
        else:
            return self._compute_jaccard_similarity(text1, text2)

    def _compute_jaccard_similarity(self, text1: str, text2: str) -> float:
        """词袋Jaccard相似度（快速但不准确）"""
        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())

        if not tokens1 or not tokens2:
            return 0.0

        intersection = tokens1 & tokens2
        union = tokens1 | tokens2

        return len(intersection) / len(union)

    def _compute_embedding_similarity(self, text1: str, text2: str) -> float:
        """基于embedding的余弦相似度（更准确）"""
        # 获取两个文本的embedding
        emb1 = self._get_embedding(text1)
        emb2 = self._get_embedding(text2)

        if emb1 is None or emb2 is None:
            # 降级到Jaccard相似度
            return self._compute_jaccard_similarity(text1, text2)

        # 计算余弦相似度
        emb1 = np.array(emb1)
        emb2 = np.array(emb2)

        cosine_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return float(cosine_sim)

    def _get_embedding(self, text: str, max_retries: int = 3) -> List[float]:
        """调用SiliconFlow API获取文本embedding"""
        api_key = os.environ.get('SILICONFLOW_API_KEY', '')

        if not api_key:
            if not hasattr(self, '_embedding_warning_shown'):
                print("  ⚠️  未设置SILICONFLOW_API_KEY，降级到Jaccard相似度")
                self._embedding_warning_shown = True
            return None

        url = "https://api.siliconflow.cn/v1/embeddings"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "Qwen/Qwen3-Embedding-4B",
            "input": text[:2000]  # 限制长度避免超限
        }

        for attempt in range(max_retries):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=10)
                response.raise_for_status()
                result = response.json()
                return result['data'][0]['embedding']
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(0.5)
                else:
                    if not hasattr(self, '_embedding_error_shown'):
                        print(f"  ⚠️  Embedding API调用失败: {e}，降级到Jaccard相似度")
                        self._embedding_error_shown = True
                    return None

        return None

    def _get_paper_quality(self, paper: Dict) -> float:
        """计算Paper的综合质量分数

        基于review的评分，归一化到[0, 1]
        如果没有review数据，返回默认值0.5
        """
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

    # ===================== 路径1: Idea → Idea → Pattern =====================

    def _recall_path1_similar_ideas(self, user_idea: str) -> Tuple[Dict[str, float], List[Tuple[str, float]]]:
        """路径1: 通过相似Idea召回Pattern (V3版本 + 两阶段优化)

        流程:
          1. 【粗排】使用Jaccard快速筛选Top-N候选（N=100）
          2. 【精排】对候选使用Embedding重新排序，选择Top-K（K=10）
          3. 直接获取这些Idea的pattern_ids
          4. 按相似度加权计算Pattern得分

        返回: (pattern_scores, top_ideas)
            - pattern_scores: {pattern_id: score}
            - top_ideas: [(idea_id, similarity), ...] 用于路径2的Domain查找
        """
        print("\n🔍 [路径1] 相似Idea召回...")

        # Step 1: 粗排 - 使用Jaccard快速筛选
        if RecallConfig.TWO_STAGE_RECALL and RecallConfig.USE_EMBEDDING:
            print(f"  [粗排] 使用Jaccard快速筛选Top-{RecallConfig.COARSE_RECALL_SIZE}...")
            coarse_similarities = []
            for idea in self.ideas:
                sim = self._compute_jaccard_similarity(user_idea, idea['description'])
                if sim > 0:
                    coarse_similarities.append((idea['idea_id'], sim))

            coarse_similarities.sort(key=lambda x: x[1], reverse=True)
            candidates = coarse_similarities[:RecallConfig.COARSE_RECALL_SIZE]

            print(f"  [精排] 使用Embedding重排Top-{RecallConfig.FINE_RECALL_SIZE}...")
            # Step 2: 精排 - 对候选使用Embedding重新计算
            fine_similarities = []
            for idea_id, _ in candidates:
                idea = self.idea_id_to_idea[idea_id]
                sim = self._compute_embedding_similarity(user_idea, idea['description'])
                if sim > 0:
                    fine_similarities.append((idea_id, sim))

            fine_similarities.sort(key=lambda x: x[1], reverse=True)
            top_ideas = fine_similarities[:RecallConfig.PATH1_TOP_K_IDEAS]

            print(f"  ✓ 粗排{len(coarse_similarities)}个 → 精排{len(candidates)}个 → 最终{len(top_ideas)}个")
        else:
            # 单阶段召回（原逻辑）
            similarities = []
            for idea in self.ideas:
                sim = self._compute_text_similarity(user_idea, idea['description'])
                if sim > 0:
                    similarities.append((idea['idea_id'], sim))

            similarities.sort(key=lambda x: x[1], reverse=True)
            top_ideas = similarities[:RecallConfig.PATH1_TOP_K_IDEAS]
            print(f"  找到 {len(similarities)} 个相似Idea，选择Top-{RecallConfig.PATH1_TOP_K_IDEAS}")

        # Step 3: 直接从Idea节点获取pattern_ids并计算得分
        pattern_scores = defaultdict(float)

        for idea_id, similarity in top_ideas:
            idea = self.idea_id_to_idea[idea_id]
            pattern_ids = idea.get('pattern_ids', [])

            # 打印Idea的前300个字符用于调试
            idea_desc = idea.get('description', '')[:300]
            print(f"  - [{idea_id}] {idea_desc}... (相似度={similarity:.3f}, {len(pattern_ids)}个Pattern)")

            # V3版本: 直接使用Idea节点中的pattern_ids
            for pattern_id in pattern_ids:
                # 得分 = 相似度 (Paper质量暂时默认0.5，已集成在相似度中)
                pattern_scores[pattern_id] += similarity

        # 排序并只保留Top-K个Pattern
        sorted_patterns = sorted(pattern_scores.items(), key=lambda x: x[1], reverse=True)
        top_patterns = dict(sorted_patterns[:RecallConfig.PATH1_FINAL_TOP_K])

        print(f"  ✓ 召回 {len(pattern_scores)} 个Pattern，保留Top-{RecallConfig.PATH1_FINAL_TOP_K}")
        return top_patterns, top_ideas

    # ===================== 路径2: Idea → Domain → Pattern =====================

    def _recall_path2_domain_patterns(self, user_idea: str, top_ideas: List[Tuple[str, float]] = None) -> Dict[str, float]:
        """路径2: 通过领域相关性召回Pattern

        流程:
          1. 使用路径1召回的 Top-1 Idea 的 Domain
          2. 在这些Domain中找到表现好的Pattern
          3. 按Domain相关性和Pattern效果加权计算得分

        Args:
            user_idea: 用户输入的Idea描述
            top_ideas: 路径1召回的Top Ideas [(idea_id, similarity), ...]

        返回: {pattern_id: score}
        """
        print("\n🌍 [路径2] 领域相关性召回...")

        # Step 1: 通过最相似Idea的Domain（与 simple_recall_demo.py 一致）
        domain_scores = []

        # 如果提供了top_ideas，使用Top-1 Idea的Domain
        if top_ideas:
            top_idea_id = top_ideas[0][0]
            top_idea = self.idea_id_to_idea.get(top_idea_id)

            if top_idea and self.G.has_node(top_idea['idea_id']):
                for successor in self.G.successors(top_idea['idea_id']):
                    edge_data = self.G[top_idea['idea_id']][successor]
                    if edge_data.get('relation') == 'belongs_to':
                        domain_id = successor
                        weight = edge_data.get('weight', 0.5)
                        domain_scores.append((domain_id, weight))

        # Fallback: 如果没有找到Domain，重新计算最相似的Idea
        if not domain_scores:
            print("  未找到直接关联的Domain，重新计算最相似Idea...")
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

            # 打印Domain详细信息
            domain_name = domain.get('name', 'N/A')
            paper_count = domain.get('paper_count', 0)
            sub_domains = domain.get('sub_domains', [])
            sub_domain_str = ', '.join(sub_domains[:5])  # 只显示前5个sub_domain
            if len(sub_domains) > 5:
                sub_domain_str += f"... (共{len(sub_domains)}个)"

            print(f"  - {domain_id} (名称={domain_name}, 相关度={domain_weight:.3f}, 论文数={paper_count})")
            if sub_domain_str:
                print(f"    子领域: {sub_domain_str}")

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

        # 排序并只保留Top-K个Pattern（避免召回过多）
        sorted_patterns = sorted(pattern_scores.items(), key=lambda x: x[1], reverse=True)
        top_patterns = dict(sorted_patterns[:RecallConfig.PATH2_FINAL_TOP_K])

        print(f"  ✓ 召回 {len(pattern_scores)} 个Pattern，保留Top-{RecallConfig.PATH2_FINAL_TOP_K}")
        return top_patterns

    # ===================== 路径3: Idea → Paper → Pattern =====================

    def _recall_path3_similar_papers(self, user_idea: str) -> Dict[str, float]:
        """路径3: 通过相似Paper召回Pattern (V3版本 + 两阶段优化)

        流程:
          1. 【粗排】使用Jaccard快速筛选Top-N候选（N=100）
          2. 【精排】对候选使用Embedding重新排序，选择Top-K（K=20）
          3. 收集这些Paper使用的Pattern
          4. 按Paper相似度和质量加权计算得分

        注意:
          - 使用Paper的title进行相似度计算(与路径1的idea description互补)
          - V3版本Paper暂无review数据时，质量默认0.5

        返回: {pattern_id: score}
        """
        print("\n📄 [路径3] 相似Paper召回...")

        # Step 1: 粗排 - 使用Jaccard快速筛选
        if RecallConfig.TWO_STAGE_RECALL and RecallConfig.USE_EMBEDDING:
            print(f"  [粗排] 使用Jaccard快速筛选Top-{RecallConfig.COARSE_RECALL_SIZE}...")
            coarse_similarities = []

            for paper in self.papers:
                paper_title = paper.get('title', '')
                if not paper_title:
                    continue

                sim = self._compute_jaccard_similarity(user_idea, paper_title)
                if sim > 0.05:  # 降低阈值以保留更多候选
                    coarse_similarities.append((paper['paper_id'], sim))

            coarse_similarities.sort(key=lambda x: x[1], reverse=True)
            candidates = coarse_similarities[:RecallConfig.COARSE_RECALL_SIZE]

            print(f"  [精排] 使用Embedding重排Top-{RecallConfig.PATH3_TOP_K_PAPERS}...")
            # Step 2: 精排 - 对候选使用Embedding重新计算
            fine_similarities = []
            for paper_id, _ in candidates:
                paper = self.paper_id_to_paper[paper_id]
                paper_title = paper.get('title', '')

                sim = self._compute_embedding_similarity(user_idea, paper_title)
                if sim > 0.1:  # 过滤低相似度
                    quality = self._get_paper_quality(paper)
                    combined_weight = sim * quality
                    fine_similarities.append((paper_id, sim, quality, combined_weight))

            fine_similarities.sort(key=lambda x: x[3], reverse=True)
            top_papers = fine_similarities[:RecallConfig.PATH3_TOP_K_PAPERS]

            print(f"  ✓ 粗排{len(coarse_similarities)}个 → 精排{len(candidates)}个 → 最终{len(top_papers)}个")
        else:
            # 单阶段召回（原逻辑）
            similarities = []

            for paper in self.papers:
                paper_title = paper.get('title', '')
                if not paper_title:
                    continue

                sim = self._compute_text_similarity(user_idea, paper_title)
                if sim > 0.1:  # 过滤低相似度
                    quality = self._get_paper_quality(paper)
                    combined_weight = sim * quality
                    similarities.append((paper['paper_id'], sim, quality, combined_weight))

            similarities.sort(key=lambda x: x[3], reverse=True)
            top_papers = similarities[:RecallConfig.PATH3_TOP_K_PAPERS]

            print(f"  找到 {len(similarities)} 个相似Paper，选择Top-{RecallConfig.PATH3_TOP_K_PAPERS}")

        # Step 3: 收集Pattern
        pattern_scores = defaultdict(float)

        for paper_id, similarity, quality, combined_weight in top_papers:
            paper = self.paper_id_to_paper.get(paper_id, {})
            quality_source = "review" if paper.get('reviews') else "默认"
            title = paper.get('title', 'N/A')
            print(f"  - {paper_id} (相似度={similarity:.3f}, 质量={quality:.3f} [{quality_source}])")
            print(f"    标题: {title}")

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

        # 排序并只保留Top-K个Pattern
        sorted_patterns = sorted(pattern_scores.items(), key=lambda x: x[1], reverse=True)
        top_patterns = dict(sorted_patterns[:RecallConfig.PATH3_FINAL_TOP_K])

        print(f"  ✓ 召回 {len(pattern_scores)} 个Pattern，保留Top-{RecallConfig.PATH3_FINAL_TOP_K}")
        return top_patterns

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
        path1_scores, top_ideas = self._recall_path1_similar_ideas(user_idea)

        # 路径2: 领域相关性召回（使用路径1的Top Ideas）
        path2_scores = self._recall_path2_domain_patterns(user_idea, top_ideas=top_ideas)

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

