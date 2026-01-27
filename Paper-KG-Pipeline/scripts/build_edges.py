"""
构建知识图谱的边关系
用于支持三路召回策略：
  路径1: Idea → Idea → Pattern (相似Idea召回，实时计算)
  路径2: Idea → Domain → Pattern (领域相关性召回)
  路径3: Idea → Paper → Pattern (相似Paper召回)
"""

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import networkx as nx
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
NODES_REVIEW = OUTPUT_DIR / "nodes_review.json"  # 新增：Review 节点

# 输出文件
EDGES_FILE = OUTPUT_DIR / "edges.json"
GRAPH_FILE = OUTPUT_DIR / "knowledge_graph_v2.gpickle"


@dataclass
class EdgeStats:
    """边统计"""
    # 基础连接边
    paper_implements_idea: int = 0
    paper_uses_pattern: int = 0
    paper_in_domain: int = 0

    # 召回边 - 路径2: Idea → Domain → Pattern
    idea_belongs_to_domain: int = 0
    pattern_works_well_in_domain: int = 0

    # 召回边 - 路径3: Idea → Paper → Pattern
    idea_similar_to_paper: int = 0


class EdgeBuilder:
    """知识图谱边构建器"""

    def __init__(self):
        self.G = nx.DiGraph()
        self.stats = EdgeStats()

        # 加载节点数据
        print("📂 加载节点数据...")
        self.ideas = self._load_json(NODES_IDEA)
        self.patterns = self._load_json(NODES_PATTERN)
        self.domains = self._load_json(NODES_DOMAIN)
        self.papers = self._load_json(NODES_PAPER)
        self.reviews = self._load_json(NODES_REVIEW) if NODES_REVIEW.exists() else []

        print(f"  ✓ Idea: {len(self.ideas)} 个")
        print(f"  ✓ Pattern: {len(self.patterns)} 个")
        print(f"  ✓ Domain: {len(self.domains)} 个")
        print(f"  ✓ Paper: {len(self.papers)} 个")
        print(f"  ✓ Review: {len(self.reviews)} 条")

        # 构建映射索引
        self._build_indices()

    def _load_json(self, filepath: Path) -> List[Dict]:
        """加载 JSON 文件"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _build_indices(self):
        """构建索引映射"""
        print("\n🔍 构建索引映射...")

        # Idea: idea_id -> idea
        self.idea_id_to_idea = {i['idea_id']: i for i in self.ideas}

        # Pattern: pattern_id -> pattern
        self.pattern_id_to_pattern = {p['pattern_id']: p for p in self.patterns}

        # Domain: domain_name -> domain_id
        self.domain_name_to_id = {d['name']: d['domain_id'] for d in self.domains}
        # Domain: domain_id -> domain
        self.domain_id_to_domain = {d['domain_id']: d for d in self.domains}

        # Paper: paper_id -> paper
        self.paper_id_to_paper = {p['paper_id']: p for p in self.papers}

        # Review: review_id -> review
        self.review_id_to_review = {r['review_id']: r for r in self.reviews}
        # Paper: paper_id -> reviews list
        self.paper_id_to_reviews = {}
        for review in self.reviews:
            paper_id = review.get('paper_id', '')
            if paper_id:
                if paper_id not in self.paper_id_to_reviews:
                    self.paper_id_to_reviews[paper_id] = []
                self.paper_id_to_reviews[paper_id].append(review)

        print(f"  ✓ 索引构建完成")

    def build_all_edges(self):
        """构建所有边"""
        print("\n" + "="*60)
        print("🔗 开始构建边关系")
        print("="*60)

        # Step 1: 基础连接边（Paper 为中心）
        self._build_paper_edges()

        # Step 2: Review 相关边
        self._build_paper_review_edges()

        # Step 3: 召回边 - 路径2: Idea → Domain → Pattern
        self._build_idea_belongs_to_domain_edges()
        self._build_pattern_works_well_in_domain_edges()

        # Step 4: 召回边 - 路径3: Idea → Paper → Pattern
        self._build_idea_similar_to_paper_edges()

        # Step 5: 保存结果
        self._save_edges()
        self._print_stats()

    # ===================== 基础连接边 =====================

    def _build_paper_edges(self):
        """构建 Paper 的基础连接边"""
        print("\n📄 构建 Paper 基础连接边...")

        for paper in self.papers:
            paper_id = paper['paper_id']
            paper_quality = self._get_paper_quality(paper)

            # 1. Paper -[implements]-> Idea
            idea_id = paper.get('idea_id', '')
            if idea_id and idea_id in self.idea_id_to_idea:
                self.G.add_edge(
                    paper_id,
                    idea_id,
                    relation='implements'
                )
                self.stats.paper_implements_idea += 1

            # 2. Paper -[uses_pattern]-> Pattern (带质量权重)
            pattern_id = paper.get('pattern_id', '')
            if pattern_id and pattern_id in self.pattern_id_to_pattern:
                self.G.add_edge(
                    paper_id,
                    pattern_id,
                    relation='uses_pattern',
                    quality=paper_quality
                )
                self.stats.paper_uses_pattern += 1

            # 3. Paper -[in_domain]-> Domain
            domain_id = paper.get('domain_id', '')
            if domain_id and domain_id in self.domain_id_to_domain:
                self.G.add_edge(
                    paper_id,
                    domain_id,
                    relation='in_domain'
                )
                self.stats.paper_in_domain += 1

        print(f"  ✓ Paper->Idea: {self.stats.paper_implements_idea} 条")
        print(f"  ✓ Paper->Pattern: {self.stats.paper_uses_pattern} 条")
        print(f"  ✓ Paper->Domain: {self.stats.paper_in_domain} 条")

    # ===================== Review 相关边 =====================

    def _build_paper_review_edges(self):
        """构建 Paper -[has_review]-> Review 边，以及 Review -[evaluates]-> Quality 伪节点"""
        print("\n⭐ 构建 Paper <-> Review 边...")

        paper_review_count = 0

        for paper in self.papers:
            paper_id = paper['paper_id']
            review_ids = paper.get('review_ids', [])

            if not review_ids:
                continue

            for review_id in review_ids:
                # Paper -[has_review]-> Review
                review = self.review_id_to_review.get(review_id)
                if not review:
                    continue

                score_value = review.get('overall_score_value', 0.5)

                self.G.add_edge(
                    paper_id,
                    review_id,
                    relation='has_review',
                    score=score_value
                )
                paper_review_count += 1

        print(f"  ✓ Paper->Review: {paper_review_count} 条")

    # ===================== 召回边 - 路径2: Idea → Domain → Pattern =====================

    def _build_idea_belongs_to_domain_edges(self):
        """构建 Idea -[belongs_to]-> Domain 边

        权重定义: Idea 相关 Paper 在该 Domain 中的占比
        """
        print("\n🔗 构建 Idea -[belongs_to]-> Domain 边...")

        for idea in self.ideas:
            idea_id = idea['idea_id']
            source_paper_ids = idea.get('source_paper_ids', [])

            if not source_paper_ids:
                continue

            # 统计每个 Domain 中的 Paper 数量
            domain_counts = defaultdict(int)

            for paper_id in source_paper_ids:
                paper = self.paper_id_to_paper.get(paper_id)
                if not paper:
                    continue

                # V3: Paper 有 domain_id 字段
                domain_id = paper.get('domain_id', '')
                if domain_id:
                    domain_counts[domain_id] += 1

            # 计算每个 Domain 的权重并创建边
            total_papers = len(source_paper_ids)

            for domain_id, count in domain_counts.items():
                weight = count / total_papers  # 占比作为权重

                self.G.add_edge(
                    idea_id,
                    domain_id,
                    relation='belongs_to',
                    weight=weight,
                    paper_count=count,
                    total_papers=total_papers
                )
                self.stats.idea_belongs_to_domain += 1

        print(f"  ✓ 共构建 {self.stats.idea_belongs_to_domain} 条 belongs_to 边")

    def _build_pattern_works_well_in_domain_edges(self):
        """构建 Pattern -[works_well_in]-> Domain 效果边

        权重定义:
          - frequency: Pattern 在该 Domain 中的使用次数
          - effectiveness: Pattern 在该 Domain 中 Paper 的平均质量相对基线的增益
        """
        print("\n🌍 构建 Pattern -[works_well_in]-> Domain 效果边...")

        for pattern in self.patterns:
            pattern_id = pattern['pattern_id']

            # 统计每个 Domain 中使用该 Pattern 的 Paper
            domain_stats = defaultdict(lambda: {'papers': [], 'qualities': []})

            for paper in self.papers:
                # V3: Paper 有 pattern_id 字段（单个）
                if paper.get('pattern_id', '') != pattern_id:
                    continue

                paper_quality = self._get_paper_quality(paper)
                # V3: Paper 有 domain_id 字段
                domain_id = paper.get('domain_id', '')

                if domain_id:
                    domain_stats[domain_id]['papers'].append(paper['paper_id'])
                    domain_stats[domain_id]['qualities'].append(paper_quality)

            # 为每个 Domain 创建 works_well_in 边
            for domain_id, stats in domain_stats.items():
                if not stats['papers']:
                    continue

                # 计算平均质量
                qualities = stats['qualities']
                avg_quality = np.mean(qualities)

                # 计算领域基线
                domain = self.domain_id_to_domain.get(domain_id)
                if not domain:
                    continue

                # V3: 根据 domain_id 筛选 Paper
                domain_papers = [p for p in self.papers
                                if p.get('domain_id', '') == domain_id]
                domain_baseline = np.mean([self._get_paper_quality(p) for p in domain_papers]) if domain_papers else 0.7

                # 效果 = 平均质量 - 基线
                effectiveness = avg_quality - domain_baseline

                # 频率
                frequency = len(stats['papers'])

                # 置信度 (样本数越多越可信)
                confidence = min(frequency / 20, 1.0)

                # 添加边
                self.G.add_edge(
                    pattern_id,
                    domain_id,
                    relation='works_well_in',
                    frequency=frequency,
                    effectiveness=effectiveness,
                    confidence=confidence,
                    avg_quality=avg_quality,
                    baseline=domain_baseline
                )
                self.stats.pattern_works_well_in_domain += 1

        print(f"  ✓ 共构建 {self.stats.pattern_works_well_in_domain} 条 works_well_in 边")

    # ===================== 召回边 - 路径3: Idea → Paper → Pattern =====================

    def _build_idea_similar_to_paper_edges(self):
        """构建 Idea -[similar_to_paper]-> Paper 边

        权重定义:
          - similarity: Idea 描述与 Paper core_idea 的语义相似度
          - quality: Paper 的综合质量分数
          - combined_weight: similarity * quality - 结合相似度和质量
        """
        print("\n🔗 构建 Idea -[similar_to_paper]-> Paper 边...")

        for idea in self.ideas:
            idea_id = idea['idea_id']
            idea_desc = idea.get('description', '')

            if not idea_desc:
                continue

            # 与所有 Paper 计算相似度
            similarities = []

            for paper in self.papers:
                paper_id = paper['paper_id']
                # V3: Paper 的 idea 字段是字符串，而不是字典
                paper_idea = paper.get('idea', '')

                if not paper_idea:
                    continue

                # 计算语义相似度（使用简单的词袋相似度）
                similarity = self._compute_text_similarity(idea_desc, paper_idea)

                # 过滤低相似度的Paper (阈值可调)
                if similarity < 0.1:
                    continue

                paper_quality = self._get_paper_quality(paper)
                combined_weight = similarity * paper_quality

                similarities.append({
                    'paper_id': paper_id,
                    'similarity': similarity,
                    'quality': paper_quality,
                    'combined_weight': combined_weight
                })

            # 排序并只保留 Top-K 个相似 Paper (避免边太多)
            similarities.sort(key=lambda x: x['combined_weight'], reverse=True)
            top_k = 50  # 每个 Idea 最多连接 50 个相似 Paper

            for item in similarities[:top_k]:
                self.G.add_edge(
                    idea_id,
                    item['paper_id'],
                    relation='similar_to_paper',
                    similarity=item['similarity'],
                    quality=item['quality'],
                    combined_weight=item['combined_weight']
                )
                self.stats.idea_similar_to_paper += 1

        print(f"  ✓ 共构建 {self.stats.idea_similar_to_paper} 条 similar_to_paper 边")

    # ===================== 辅助函数 =====================

    def _get_paper_quality(self, paper: Dict) -> float:
        """计算 Paper 的综合质量分数

        基于 review 的评分，归一化到 [0, 1]
        """
        # V3: 使用 review_stats 中的 avg_score
        review_stats = paper.get('review_stats', {})
        avg_score = review_stats.get('avg_score', 0.5)

        # 如果没有 review 数据，返回默认值
        if avg_score == 0.5 and review_stats.get('review_count', 0) == 0:
            return 0.5

        # avg_score 已经被归一化到 [0, 1]
        return min(max(avg_score, 0.0), 1.0)

    def _compute_text_similarity(self, text1: str, text2: str) -> float:
        """计算两段文本的简单相似度

        使用词袋模型 + Jaccard 相似度
        """
        # 简单分词（按空格和标点）
        def tokenize(text):
            import re
            text = text.lower()
            tokens = re.findall(r'\b\w+\b', text)
            return set(tokens)

        tokens1 = tokenize(text1)
        tokens2 = tokenize(text2)

        if not tokens1 or not tokens2:
            return 0.0

        # Jaccard 相似度
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)

        return intersection / union if union > 0 else 0.0

    # ===================== 保存和统计 =====================

    def _save_edges(self):
        """保存边数据"""
        print("\n💾 保存边数据...")

        # 1. 保存为 JSON
        edges_data = []
        for u, v, data in self.G.edges(data=True):
            edge = {
                'source': u,
                'target': v,
                **data
            }
            edges_data.append(edge)

        with open(EDGES_FILE, 'w', encoding='utf-8') as f:
            json.dump(edges_data, f, ensure_ascii=False, indent=2)
        print(f"  ✓ 保存到: {EDGES_FILE}")

        # 2. 保存完整图谱（包含节点和边）
        import pickle
        with open(GRAPH_FILE, 'wb') as f:
            pickle.dump(self.G, f)
        print(f"  ✓ 保存到: {GRAPH_FILE}")

    def _print_stats(self):
        """打印统计信息"""
        print("\n" + "="*60)
        print("📊 边构建统计")
        print("="*60)
        print("\n【基础连接边】")
        print(f"  Paper->Idea:                {self.stats.paper_implements_idea} 条")
        print(f"  Paper->Pattern:             {self.stats.paper_uses_pattern} 条")
        print(f"  Paper->Domain:              {self.stats.paper_in_domain} 条")

        print("\n【Review 质量边】")
        review_edges = sum(1 for u, v, d in self.G.edges(data=True) if d.get('relation') == 'has_review')
        print(f"  Paper->Review:              {review_edges} 条")
        print(f"  关联论文数:                  {len([p for p in self.papers if p.get('review_ids')])} 篇")
        print(f"  总Review数:                  {len(self.reviews)} 条")

        print("\n【召回边 - 路径2: Idea → Domain → Pattern】")
        print(f"  Idea->Domain:               {self.stats.idea_belongs_to_domain} 条")
        print(f"  Pattern->Domain:            {self.stats.pattern_works_well_in_domain} 条")

        print("\n【召回边 - 路径3: Idea → Paper → Pattern】")
        print(f"  Idea->Paper:                {self.stats.idea_similar_to_paper} 条")

        print("\n【注意】路径1 (Idea → Idea → Pattern) 使用实时相似度计算，无需预构建边")

        print("-"*60)
        print(f"  总边数:                     {self.G.number_of_edges()} 条")
        print("="*60 + "\n")


def main():
    """主函数"""
    builder = EdgeBuilder()
    builder.build_all_edges()
    return builder.G


if __name__ == '__main__':
    main()