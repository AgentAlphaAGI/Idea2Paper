"""
知识图谱构建脚本 V3 - 基于ICLR数据源
从assignments和cluster_library构建知识图谱

数据源：
  - data/ICLR_25/assignments.jsonl: Paper到Pattern的分配关系
  - data/ICLR_25/cluster_library_sorted.jsonl: Pattern Cluster信息
  - data/ICLR_25/iclr_patterns_full_cn_912.jsonl: Pattern详细属性

节点类型：
  - Idea: 核心创新点（从pattern的idea字段提取）
  - Pattern: 写作套路（从cluster信息构建）
  - Domain: 研究领域（从domain字段聚合）
  - Paper: 论文（从assignments构建）

输出:
  - output/nodes_idea.json
  - output/nodes_pattern.json
  - output/nodes_domain.json
  - output/nodes_paper.json
  - output/knowledge_graph_stats.json
"""

import hashlib
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List

# 导入pipeline工具函数
sys.path.insert(0, str(Path(__file__).parent))
from pipeline.utils import call_llm, parse_json_from_llm

# ===================== 配置 =====================

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

# 输入路径
DATA_DIR = PROJECT_ROOT / "data" / "ICLR_25"
ASSIGNMENTS_FILE = DATA_DIR / "assignments.jsonl"
CLUSTER_LIBRARY_FILE = DATA_DIR / "cluster_library_sorted.jsonl"
PATTERN_DETAILS_FILE = DATA_DIR / "iclr_patterns_full.jsonl"  # 使用完整的英文版本
REVIEWS_FILE = DATA_DIR / "paper_reviews_dataset_iclr_reviews_filtered.jsonl"  # Review 数据

# 输出路径
OUTPUT_DIR = PROJECT_ROOT / "output"
NODES_IDEA = OUTPUT_DIR / "nodes_idea.json"
NODES_PATTERN = OUTPUT_DIR / "nodes_pattern.json"
NODES_DOMAIN = OUTPUT_DIR / "nodes_domain.json"
NODES_PAPER = OUTPUT_DIR / "nodes_paper.json"
NODES_REVIEW = OUTPUT_DIR / "nodes_review.json"  # 新增：Review 节点
STATS_FILE = OUTPUT_DIR / "knowledge_graph_stats.json"

# LLM API 配置
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "")
LLM_API_URL = os.getenv("LLM_API_URL", "https://api.siliconflow.cn/v1/chat/completions")
LLM_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen3-14B")


# ===================== 数据类 =====================

@dataclass
class GraphStats:
    """图谱统计信息"""
    total_nodes: int = 0
    ideas: int = 0
    patterns: int = 0
    domains: int = 0
    papers: int = 0
    reviews: int = 0


# ===================== 节点构建器 =====================

class KnowledgeGraphBuilderV3:
    """知识图谱构建器 V3 - 基于ICLR数据源"""

    def __init__(self):
        self.stats = GraphStats()

        # 节点存储
        self.idea_nodes: List[Dict] = []
        self.pattern_nodes: List[Dict] = []
        self.domain_nodes: List[Dict] = []
        self.paper_nodes: List[Dict] = []
        self.review_nodes: List[Dict] = []  # 新增：Review 节点

        # 去重映射
        self.idea_map: Dict[str, str] = {}           # idea_hash -> idea_id
        self.domain_map: Dict[str, str] = {}         # domain_name -> domain_id
        self.pattern_map: Dict[int, str] = {}        # cluster_id -> pattern_id
        self.global_pattern_map: Dict[str, str] = {} # global_pattern_id -> idea_id
        self.paper_map: Dict[str, str] = {}          # paper_id -> paper_id
        self.review_map: Dict[str, str] = {}         # 新增：review_id -> review_id

        # 中间数据
        self.paper_details: Dict[str, Dict] = {}     # paper_id -> pattern details
        self.paper_reviews_map: Dict[str, List[Dict]] = {}  # 新增：paper_id -> list of reviews

    def build(self):
        """构建完整的知识图谱"""
        print("=" * 60)
        print("🚀 开始构建知识图谱 V3 (ICLR数据源)")
        print("=" * 60)

        # Step 1: 加载数据
        print("\n【Step 1】加载数据")
        assignments = self._load_assignments()
        clusters = self._load_clusters()
        pattern_details = self._load_pattern_details()
        reviews_data = self._load_reviews()  # 新增：加载 Review 数据
        print(f"✅ 加载 {len(assignments)} 篇论文分配")
        print(f"✅ 加载 {len(clusters)} 个 Pattern Clusters")
        print(f"✅ 加载 {len(pattern_details)} 篇论文的详细Pattern")
        print(f"✅ 加载 {len(reviews_data)} 篇论文的 Review 数据")  # 新增

        # Step 2: 构建节点
        print("\n【Step 2】构建节点")
        self._build_pattern_nodes(clusters)
        self._enhance_patterns_with_llm(clusters)  # LLM增强Pattern节点
        self._build_idea_nodes(pattern_details)
        self._build_domain_nodes(assignments, clusters)
        self._build_paper_nodes(assignments, pattern_details)
        self._build_review_nodes(reviews_data)  # 新增：构建 Review 节点

        # Step 3: 建立关联
        print("\n【Step 3】建立节点关联")
        self._link_paper_to_pattern(assignments)
        self._link_paper_to_idea()
        self._link_paper_to_domain()
        self._link_paper_to_review()  # 新增：关联 Paper 和 Review
        self._link_idea_to_pattern()

        # Step 4: 保存节点
        print("\n【Step 4】保存节点")
        self._save_nodes()

        # Step 5: 统计
        print("\n【Step 5】统计信息")
        self._update_stats()
        self._save_stats()
        self._print_stats()

        print("\n" + "=" * 60)
        print("✅ 知识图谱构建完成!")
        print("=" * 60)

    # ===================== 数据加载 =====================

    def _load_assignments(self) -> List[Dict]:
        """加载论文分配数据 (assignments.jsonl)"""
        assignments = []
        if not ASSIGNMENTS_FILE.exists():
            print(f"⚠️  文件不存在: {ASSIGNMENTS_FILE}")
            return []

        with open(ASSIGNMENTS_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    assignments.append(json.loads(line))
        return assignments

    def _load_clusters(self) -> List[Dict]:
        """加载Pattern Cluster数据 (cluster_library_sorted.jsonl)"""
        clusters = []
        if not CLUSTER_LIBRARY_FILE.exists():
            print(f"⚠️  文件不存在: {CLUSTER_LIBRARY_FILE}")
            return []

        with open(CLUSTER_LIBRARY_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    clusters.append(json.loads(line))
        return clusters

    def _load_pattern_details(self) -> Dict[str, Dict]:
        """加载Pattern详细属性 (iclr_patterns_full.jsonl)"""
        details = {}
        if not PATTERN_DETAILS_FILE.exists():
            print(f"⚠️  文件不存在: {PATTERN_DETAILS_FILE}")
            return {}

        with open(PATTERN_DETAILS_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    paper_id = item.get('paper_id')
                    if paper_id:
                        details[paper_id] = item

        self.paper_details = details
        return details

    def _load_reviews(self) -> Dict[str, List[Dict]]:
        """加载Review数据 (paper_reviews_dataset_iclr_reviews_filtered.jsonl)"""
        reviews_data = {}
        if not REVIEWS_FILE.exists():
            print(f"⚠️  文件不存在: {REVIEWS_FILE}")
            return {}

        with open(REVIEWS_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    paper_id = item.get('id')  # 论文ID来自 'id' 字段
                    related_notes = item.get('related_notes', '[]')

                    # 解析 related_notes（是一个 JSON 字符串）
                    if isinstance(related_notes, str):
                        try:
                            reviews_list = json.loads(related_notes)
                        except:
                            reviews_list = []
                    else:
                        reviews_list = related_notes if isinstance(related_notes, list) else []

                    if paper_id and reviews_list:
                        reviews_data[paper_id] = reviews_list
                        self.paper_reviews_map[paper_id] = reviews_list

        return reviews_data

    # ===================== 构建节点 =====================

    def _build_pattern_nodes(self, clusters: List[Dict]):
        """构建 Pattern 节点（基于cluster信息）+ LLM增强"""
        print("\n📋 构建 Pattern 节点...")

        for cluster in clusters:
            cluster_id = cluster.get('cluster_id')
            if cluster_id == -1:  # 跳过未分配的cluster
                continue

            pattern_id = f"pattern_{cluster_id}"
            self.pattern_map[cluster_id] = pattern_id

            # 提取代表性论文的pattern信息
            exemplars = cluster.get('exemplars', [])
            exemplar_ideas = []
            exemplar_problems = []
            exemplar_solutions = []
            exemplar_stories = []  # 新增：提取story信息

            for ex in exemplars[:3]:  # 取前3个代表性论文
                if ex.get('idea'):
                    exemplar_ideas.append(ex['idea'])
                if ex.get('base_problem'):
                    exemplar_problems.append(ex['base_problem'])
                if ex.get('solution_pattern'):
                    exemplar_solutions.append(ex['solution_pattern'])
                if ex.get('story'):  # 新增：提取story
                    exemplar_stories.append(ex['story'])

            # 构建Pattern节点
            pattern_node = {
                'pattern_id': pattern_id,
                'cluster_id': cluster_id,
                'name': cluster.get('cluster_name', ''),
                'size': cluster.get('size', 0),

                # 领域信息
                'domain': cluster.get('retrieval_facets', {}).get('domain', ''),
                'sub_domains': cluster.get('retrieval_facets', {}).get('sub_domains', []),

                # 聚类质量指标
                'coherence': {
                    'centroid_mean': cluster.get('coherence', {}).get('centroid_mean', 0),
                    'centroid_p50': cluster.get('coherence', {}).get('centroid_p50', 0),
                    'pairwise_sample_mean': cluster.get('coherence', {}).get('pairwise_sample_mean', 0),
                    'pairwise_sample_p50': cluster.get('coherence', {}).get('pairwise_sample_p50', 0)
                },

                # 代表性信息（从exemplars提取，包含story）
                'summary': {
                    'representative_ideas': exemplar_ideas,
                    'common_problems': exemplar_problems,
                    'solution_approaches': exemplar_solutions,
                    'story': exemplar_stories  # 新增：story维度
                },

                # 元数据
                'exemplar_count': len(exemplars),
                'exemplar_paper_ids': [ex.get('paper_id') for ex in exemplars if ex.get('paper_id')]
            }

            self.pattern_nodes.append(pattern_node)

        print(f"  ✓ 创建 {len(self.pattern_nodes)} 个 Pattern 节点")

    def _enhance_patterns_with_llm(self, clusters: List[Dict]):
        """使用LLM增强Pattern节点的归纳性描述"""
        print("\n🤖 使用LLM增强Pattern节点...")

        # 检查API Key
        if not SILICONFLOW_API_KEY:
            print("  ⚠️  警告: SILICONFLOW_API_KEY 未设置，跳过LLM增强")
            return

        enhanced_count = 0

        for idx, pattern_node in enumerate(self.pattern_nodes):
            cluster_id = pattern_node['cluster_id']

            # 找到对应的cluster
            cluster = next((c for c in clusters if c.get('cluster_id') == cluster_id), None)
            if not cluster:
                continue

            # 收集该cluster中所有论文的Pattern信息
            exemplars = cluster.get('exemplars', [])

            # 构建Prompt
            prompt = self._build_llm_prompt_for_pattern(pattern_node, exemplars)

            # 调用LLM
            print(f"  处理 {idx+1}/{len(self.pattern_nodes)}: {pattern_node['name']} (cluster_id={cluster_id}, size={pattern_node['size']})")

            llm_response = call_llm(prompt, temperature=0.3, max_tokens=1500)

            if llm_response:
                # 解析LLM响应
                parsed_summary = parse_json_from_llm(llm_response)

                if parsed_summary and all(k in parsed_summary for k in ['representative_ideas', 'common_problems', 'solution_approaches', 'story']):
                    # 添加LLM生成的归纳性描述
                    pattern_node['llm_enhanced_summary'] = {
                        'representative_ideas': parsed_summary.get('representative_ideas', ''),
                        'common_problems': parsed_summary.get('common_problems', ''),
                        'solution_approaches': parsed_summary.get('solution_approaches', ''),
                        'story': parsed_summary.get('story', '')
                    }
                    pattern_node['llm_enhanced'] = True
                    enhanced_count += 1
                    print(f"    ✓ LLM增强成功")
                else:
                    print(f"    ⚠️  LLM响应格式错误，跳过")
                    pattern_node['llm_enhanced'] = False
            else:
                print(f"    ❌ LLM调用失败，跳过")
                pattern_node['llm_enhanced'] = False

        print(f"\n  ✓ 成功增强 {enhanced_count}/{len(self.pattern_nodes)} 个 Pattern 节点")

    def _build_llm_prompt_for_pattern(self, pattern_node: Dict, exemplars: List[Dict]) -> str:
        """为Pattern生成LLM Prompt"""

        cluster_name = pattern_node.get('name', '')
        cluster_size = pattern_node.get('size', 0)
        domain = pattern_node.get('domain', '')

        # 收集所有exemplar的信息
        all_ideas = []
        all_problems = []
        all_solutions = []
        all_stories = []

        for ex in exemplars:
            if ex.get('idea'):
                all_ideas.append(ex['idea'])
            if ex.get('base_problem'):
                all_problems.append(ex['base_problem'])
            if ex.get('solution_pattern'):
                all_solutions.append(ex['solution_pattern'])
            if ex.get('story'):
                all_stories.append(ex['story'])

        # 构建Prompt
        prompt = f"""You are a research pattern analyst. Given a cluster of {cluster_size} research papers in the "{domain}" domain, your task is to analyze the common patterns and generate a comprehensive summary.

**Cluster Name**: {cluster_name}

**Representative Ideas from papers**:
{self._format_list(all_ideas[:20])}

**Common Problems from papers**:
{self._format_list(all_problems[:20])}

**Solution Approaches from papers**:
{self._format_list(all_solutions[:20])}

**Story/Reframing from papers**:
{self._format_list(all_stories[:20])}

---

Based on the above information, please generate an **inductive summary** for this pattern cluster. For each category, provide **ONE comprehensive sentence** that captures the essence of all papers in this cluster.

Return your response in the following JSON format:

{{
  "representative_ideas": "A single comprehensive sentence summarizing the core innovative ideas across all papers in this cluster.",
  "common_problems": "A single comprehensive sentence describing the common challenges and problems addressed by papers in this cluster.",
  "solution_approaches": "A single comprehensive sentence outlining the general solution strategies and methodologies employed across this cluster.",
  "story": "A single comprehensive sentence that reframes the research narrative and explains the transformative perspective this pattern brings to the field."
}}

Make sure each sentence is detailed, informative, and captures the diversity of approaches within the cluster. Keep the response in JSON format only."""

        return prompt

    def _format_list(self, items: List[str]) -> str:
        """格式化列表为带序号的字符串"""
        if not items:
            return "(No data available)"

        return "\n".join([f"{i+1}. {item[:300]}..." if len(item) > 300 else f"{i+1}. {item}"
                         for i, item in enumerate(items[:20])])

    def _build_idea_nodes(self, pattern_details: Dict[str, Dict]):
        """构建 Idea 节点（从pattern_details的idea字段提取）"""
        print("\n💡 构建 Idea 节点...")

        for paper_id, details in pattern_details.items():
            idea_text = details.get('idea', '')
            if not idea_text:
                continue

            # 用hash去重
            idea_hash = hashlib.md5(idea_text.encode()).hexdigest()[:16]

            if idea_hash not in self.idea_map:
                idea_id = f"idea_{len(self.idea_nodes)}"
                self.idea_map[idea_hash] = idea_id

                # 提取research_patterns信息
                patterns = details.get('research_patterns', [])
                first_pattern = patterns[0] if patterns else {}

                self.idea_nodes.append({
                    'idea_id': idea_id,
                    'description': idea_text,

                    # 从research_patterns提取
                    'base_problem': first_pattern.get('base_problem', ''),
                    'solution_pattern': first_pattern.get('solution_pattern', ''),
                    'story': first_pattern.get('story', ''),
                    'application': first_pattern.get('application', ''),

                    # 领域信息
                    'domain': details.get('domain', ''),
                    'sub_domains': details.get('sub_domains', []),

                    # 来源信息
                    'source_paper_ids': [paper_id],
                    'pattern_ids': []  # 稍后填充
                })
            else:
                # 已存在的Idea，追加paper_id
                idea_id = self.idea_map[idea_hash]
                for idea_node in self.idea_nodes:
                    if idea_node['idea_id'] == idea_id:
                        if paper_id not in idea_node['source_paper_ids']:
                            idea_node['source_paper_ids'].append(paper_id)
                        break

        print(f"  ✓ 创建 {len(self.idea_nodes)} 个 Idea 节点")

    def _build_domain_nodes(self, assignments: List[Dict], clusters: List[Dict]):
        """构建 Domain 节点（聚合domain和sub_domains信息）"""
        print("\n🌐 构建 Domain 节点...")

        domain_stats = defaultdict(lambda: {
            'paper_count': 0,
            'sub_domains': set(),
            'related_patterns': set(),
            'paper_ids': []
        })

        # 从assignments聚合
        for assignment in assignments:
            domain = assignment.get('domain', '')
            if not domain:
                continue

            paper_id = assignment.get('paper_id', '')
            cluster_id = assignment.get('cluster_id', -1)

            domain_stats[domain]['paper_count'] += 1
            domain_stats[domain]['paper_ids'].append(paper_id)

            # 聚合sub_domains
            for sub_domain in assignment.get('sub_domains', []):
                domain_stats[domain]['sub_domains'].add(sub_domain)

            # 关联pattern
            if cluster_id != -1:
                domain_stats[domain]['related_patterns'].add(f"pattern_{cluster_id}")

        # 生成Domain节点
        for domain_name, stats in domain_stats.items():
            domain_id = f"domain_{len(self.domain_nodes)}"
            self.domain_map[domain_name] = domain_id

            self.domain_nodes.append({
                'domain_id': domain_id,
                'name': domain_name,
                'paper_count': stats['paper_count'],
                'sub_domains': sorted(list(stats['sub_domains'])),
                'related_pattern_ids': sorted(list(stats['related_patterns'])),
                'sample_paper_ids': stats['paper_ids'][:10]  # 保留前10个样例
            })

        print(f"  ✓ 创建 {len(self.domain_nodes)} 个 Domain 节点")

    def _build_paper_nodes(self, assignments: List[Dict], pattern_details: Dict[str, Dict]):
        """构建 Paper 节点（合并assignments和pattern_details信息）"""
        print("\n📄 构建 Paper 节点...")

        for assignment in assignments:
            paper_id = assignment.get('paper_id', '')
            if not paper_id:
                continue

            self.paper_map[paper_id] = paper_id

            # 从pattern_details获取详细信息
            details = pattern_details.get(paper_id, {})
            patterns = details.get('research_patterns', [])
            first_pattern = patterns[0] if patterns else {}

            # 构建Paper节点
            self.paper_nodes.append({
                'paper_id': paper_id,
                'title': assignment.get('paper_title', ''),

                # Pattern关联
                'global_pattern_id': assignment.get('global_pattern_id', ''),
                'cluster_id': assignment.get('cluster_id', -1),
                'cluster_prob': assignment.get('cluster_prob', 0),

                # 领域信息
                'domain': assignment.get('domain', ''),
                'sub_domains': assignment.get('sub_domains', []),

                # Idea信息（来自pattern_details）
                'idea': details.get('idea', ''),

                # Pattern详细信息
                'pattern_details': {
                    'base_problem': first_pattern.get('base_problem', ''),
                    'solution_pattern': first_pattern.get('solution_pattern', ''),
                    'story': first_pattern.get('story', ''),
                    'application': first_pattern.get('application', '')
                },

                # 关联ID（稍后填充）
                'pattern_id': '',
                'idea_id': '',
                'domain_id': ''
            })

        print(f"  ✓ 创建 {len(self.paper_nodes)} 个 Paper 节点")

    def _build_review_nodes(self, reviews_data: Dict[str, List[Dict]]):
        """构建 Review 节点"""
        print("\n⭐ 构建 Review 节点...")

        for paper_id, reviews_list in reviews_data.items():
            for review in reviews_list:
                review_id = review.get('review_id', '')
                if not review_id:
                    continue

                self.review_map[review_id] = review_id

                # 解析 overall_score
                overall_score = review.get('overall_score', '')
                score_value = self._parse_overall_score(overall_score)

                # 提取关键信息
                self.review_nodes.append({
                    'review_id': review_id,
                    'paper_id': review.get('paper_id', ''),

                    # 评分信息
                    'overall_score': overall_score,
                    'overall_score_value': score_value,  # 数值化的分数
                    'confidence': review.get('confidence'),  # 可能为 None

                    # 评价内容
                    'paper_summary': review.get('paper_summary', '')[:500],  # 限制长度
                    'strengths': review.get('strengths', ''),
                    'weaknesses': review.get('weaknesses', ''),
                    'comments': review.get('comments', ''),
                    'tldr': review.get('tldr', ''),

                    # 评价维度
                    'contribution': review.get('contribution'),
                    'correctness': review.get('correctness', ''),
                    'clarity_quality_novelty_reproducibility': review.get('clarity_quality_novelty_reproducibility', ''),
                    'recommendation': review.get('recommendation', ''),

                    # 元数据
                    'reviewer': review.get('reviewer'),
                })

        print(f"  ✓ 创建 {len(self.review_nodes)} 个 Review 节点")

    def _parse_overall_score(self, score_str: str) -> float:
        """将整体评分字符串转换为数值 (0-1)

        基础映射，后续在 review_stats 中会综合多个维度重新计算
        """
        if not score_str:
            return 0.5

        score_str = score_str.lower().strip()

        # 按数字大小映射 (1-10 的评分)
        score_mapping = {
            '10': 1.0,
            '9': 0.95,
            '8': 0.85,
            '7': 0.7,
            '6': 0.6,
            '5': 0.5,
            '4': 0.4,
            '3': 0.3,
            '2': 0.2,
            '1': 0.1,
        }

        # 先尝试提取数字（例如 "6: marginally above..." -> 0.6）
        import re
        numbers = re.findall(r'\d+', score_str)
        if numbers:
            for num_str in numbers:
                if num_str in score_mapping:
                    return score_mapping[num_str]

        # 文本映射
        text_mapping = {
            'accept': 0.8,
            'reject': 0.2,
            'borderline': 0.5,
        }

        for key, value in text_mapping.items():
            if key in score_str:
                return value

        return 0.5  # 默认中等评分

    # ===================== 建立关联 =====================

    def _link_paper_to_pattern(self, assignments: List[Dict]):
        """建立 Paper -> Pattern 关联"""
        print("\n🔗 建立 Paper -> Pattern 关联...")

        for paper_node in self.paper_nodes:
            cluster_id = paper_node.get('cluster_id', -1)
            if cluster_id != -1 and cluster_id in self.pattern_map:
                paper_node['pattern_id'] = self.pattern_map[cluster_id]

        linked_count = sum(1 for p in self.paper_nodes if p['pattern_id'])
        print(f"  ✓ {linked_count}/{len(self.paper_nodes)} 篇论文关联到Pattern")

    def _link_paper_to_idea(self):
        """建立 Paper -> Idea 关联"""
        print("\n🔗 建立 Paper -> Idea 关联...")

        for paper_node in self.paper_nodes:
            paper_id = paper_node['paper_id']
            idea_text = paper_node.get('idea', '')

            if idea_text:
                idea_hash = hashlib.md5(idea_text.encode()).hexdigest()[:16]
                if idea_hash in self.idea_map:
                    paper_node['idea_id'] = self.idea_map[idea_hash]

        linked_count = sum(1 for p in self.paper_nodes if p['idea_id'])
        print(f"  ✓ {linked_count}/{len(self.paper_nodes)} 篇论文关联到Idea")

    def _link_paper_to_domain(self):
        """建立 Paper -> Domain 关联"""
        print("\n🔗 建立 Paper -> Domain 关联...")

        for paper_node in self.paper_nodes:
            domain_name = paper_node.get('domain', '')

            if domain_name and domain_name in self.domain_map:
                paper_node['domain_id'] = self.domain_map[domain_name]

        linked_count = sum(1 for p in self.paper_nodes if p['domain_id'])
        print(f"  ✓ {linked_count}/{len(self.paper_nodes)} 篇论文关联到Domain")

    def _link_paper_to_review(self):
        """建立 Paper -> Review 关联并补充 Paper 的 Review 质量信息"""
        print("\n🔗 建立 Paper -> Review 关联...")

        # 为每个 Paper 收集关联的 Review
        paper_review_stats = defaultdict(lambda: {
            'review_ids': [],
            'avg_score': 0.0,
            'review_count': 0,
            'highest_score': 0.0,
            'lowest_score': 1.0
        })

        for review_node in self.review_nodes:
            paper_id = review_node.get('paper_id', '')
            if not paper_id:
                continue

            score_value = review_node.get('overall_score_value', 0.5)
            paper_review_stats[paper_id]['review_ids'].append(review_node['review_id'])
            paper_review_stats[paper_id]['review_count'] += 1
            paper_review_stats[paper_id]['highest_score'] = max(
                paper_review_stats[paper_id]['highest_score'], score_value
            )
            paper_review_stats[paper_id]['lowest_score'] = min(
                paper_review_stats[paper_id]['lowest_score'], score_value
            )

        # 计算平均分并更新 Paper 节点
        for paper_id, stats in paper_review_stats.items():
            if stats['review_count'] > 0:
                review_ids = stats['review_ids']
                scores = [
                    next((r['overall_score_value'] for r in self.review_nodes if r['review_id'] == rid), 0.5)
                    for rid in review_ids
                ]
                stats['avg_score'] = sum(scores) / len(scores)

            # 找到对应的 Paper 节点并补充信息
            for paper_node in self.paper_nodes:
                if paper_node['paper_id'] == paper_id:
                    paper_node['review_ids'] = stats['review_ids']
                    paper_node['review_stats'] = {
                        'review_count': stats['review_count'],
                        'avg_score': stats['avg_score'],
                        'highest_score': stats['highest_score'],
                        'lowest_score': stats['lowest_score']
                    }
                    break

        linked_count = sum(1 for p in self.paper_nodes if p.get('review_ids'))
        print(f"  ✓ {linked_count}/{len(self.paper_nodes)} 篇论文关联到Review")
        print(f"  ✓ 共关联 {len(self.review_nodes)} 条Review")

    def _link_idea_to_pattern(self):
        """建立 Idea -> Pattern 关联（通过Paper中转）"""
        print("\n🔗 建立 Idea -> Pattern 关联...")

        # 为每个Idea收集pattern_ids
        idea_to_patterns = defaultdict(set)

        for paper_node in self.paper_nodes:
            idea_id = paper_node.get('idea_id', '')
            pattern_id = paper_node.get('pattern_id', '')

            if idea_id and pattern_id:
                idea_to_patterns[idea_id].add(pattern_id)

        # 更新Idea节点
        for idea_node in self.idea_nodes:
            idea_id = idea_node['idea_id']
            idea_node['pattern_ids'] = sorted(list(idea_to_patterns[idea_id]))

        total_links = sum(len(patterns) for patterns in idea_to_patterns.values())
        print(f"  ✓ 共建立 {total_links} 个 Idea->Pattern 连接")

        if len(self.idea_nodes) > 0:
            avg_patterns = total_links / len(self.idea_nodes)
            print(f"  ✓ 平均每个 Idea 关联 {avg_patterns:.1f} 个 Pattern")

    # ===================== 保存和统计 =====================

    def _save_nodes(self):
        """保存所有节点到独立文件"""
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        with open(NODES_IDEA, 'w', encoding='utf-8') as f:
            json.dump(self.idea_nodes, f, ensure_ascii=False, indent=2)
        print(f"  ✓ {NODES_IDEA}")

        with open(NODES_PATTERN, 'w', encoding='utf-8') as f:
            json.dump(self.pattern_nodes, f, ensure_ascii=False, indent=2)
        print(f"  ✓ {NODES_PATTERN}")

        with open(NODES_DOMAIN, 'w', encoding='utf-8') as f:
            json.dump(self.domain_nodes, f, ensure_ascii=False, indent=2)
        print(f"  ✓ {NODES_DOMAIN}")

        with open(NODES_PAPER, 'w', encoding='utf-8') as f:
            json.dump(self.paper_nodes, f, ensure_ascii=False, indent=2)
        print(f"  ✓ {NODES_PAPER}")

        with open(NODES_REVIEW, 'w', encoding='utf-8') as f:
            json.dump(self.review_nodes, f, ensure_ascii=False, indent=2)
        print(f"  ✓ {NODES_REVIEW}")

    def _update_stats(self):
        """更新统计信息"""
        self.stats.ideas = len(self.idea_nodes)
        self.stats.patterns = len(self.pattern_nodes)
        self.stats.domains = len(self.domain_nodes)
        self.stats.papers = len(self.paper_nodes)
        self.stats.reviews = len(self.review_nodes)
        self.stats.total_nodes = (
            self.stats.ideas +
            self.stats.patterns +
            self.stats.domains +
            self.stats.papers +
            self.stats.reviews
        )

    def _save_stats(self):
        """保存统计信息"""
        with open(STATS_FILE, 'w', encoding='utf-8') as f:
            json.dump(asdict(self.stats), f, ensure_ascii=False, indent=2)
        print(f"  ✓ {STATS_FILE}")

    def _print_stats(self):
        """打印统计信息"""
        print("\n📊 知识图谱统计 (V3 - ICLR):")
        print("-" * 40)
        print(f"  总节点数:  {self.stats.total_nodes}")
        print(f"  Idea:      {self.stats.ideas}")
        print(f"  Pattern:   {self.stats.patterns}")
        print(f"  Domain:    {self.stats.domains}")
        print(f"  Paper:     {self.stats.papers}")
        print("-" * 40)


def main():
    """主函数"""
    builder = KnowledgeGraphBuilderV3()
    builder.build()


if __name__ == '__main__':
    main()

