"""
知识图谱构建脚本 V2 - 精简版（四类节点）
将论文抽取结果和Pattern聚类结果组装成精简的知识图谱

节点类型：
  - Idea: 核心创新点
  - Pattern: 写作套路（来自 patterns_structured.json）
  - Domain: 研究领域
  - Paper: 论文（含 Skeleton, Trick, Review 详细信息）

输入:
  - data/{conference}/*_paper_node.json: 论文抽取结果
  - output/patterns_structured.json: Pattern聚类结果

输出:
  - output/nodes_idea.json: Idea 节点
  - output/nodes_pattern.json: Pattern 节点
  - output/nodes_domain.json: Domain 节点
  - output/nodes_paper.json: Paper 节点
  - output/knowledge_graph_stats.json: 统计信息
"""

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List

# ===================== 配置 =====================

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

# 输入路径
DATA_DIR = PROJECT_ROOT / "data"
PATTERNS_FILE = PROJECT_ROOT / "output" / "patterns_structured.json"

# 输出路径
OUTPUT_DIR = PROJECT_ROOT / "output"
NODES_IDEA = OUTPUT_DIR / "nodes_idea.json"
NODES_PATTERN = OUTPUT_DIR / "nodes_pattern.json"
NODES_DOMAIN = OUTPUT_DIR / "nodes_domain.json"
NODES_PAPER = OUTPUT_DIR / "nodes_paper.json"
STATS_FILE = OUTPUT_DIR / "knowledge_graph_stats.json"

CONFERENCES = ["ACL_2017", "ARR_2022", "COLING_2020"]


# ===================== 数据类 =====================

@dataclass
class GraphStats:
    """图谱统计信息"""
    total_nodes: int = 0
    ideas: int = 0
    patterns: int = 0
    domains: int = 0
    papers: int = 0


# ===================== 节点构建器 =====================

class KnowledgeGraphBuilderV2:
    """知识图谱构建器 V2 - 四类节点版本"""

    def __init__(self):
        self.stats = GraphStats()

        # 节点存储
        self.idea_nodes: List[Dict] = []
        self.pattern_nodes: List[Dict] = []
        self.domain_nodes: List[Dict] = []
        self.paper_nodes: List[Dict] = []

        # 去重映射
        self.idea_map: Dict[str, str] = {}      # idea_hash -> idea_id
        self.domain_map: Dict[str, str] = {}    # domain_name -> domain_id
        self.pattern_map: Dict[int, str] = {}   # pattern_id -> pattern_id
        self.paper_map: Dict[str, str] = {}     # paper_id -> paper_id

    def build(self):
        """构建完整的知识图谱"""
        print("=" * 60)
        print("🚀 开始构建知识图谱 V2 (四类节点)")
        print("=" * 60)

        # Step 1: 加载数据
        print("\n【Step 1】加载数据")
        papers = self._load_papers()
        patterns = self._load_patterns()
        reviews = self._load_reviews()
        paper_to_pattern = self._load_paper_to_pattern()
        print(f"✅ 加载 {len(papers)} 篇论文, {len(patterns)} 个 Pattern, {len(reviews)} 条 Review")
        print(f"✅ Paper→Pattern 映射: {len(paper_to_pattern)} 篇论文有 Pattern")

        # Step 2: 构建节点
        print("\n【Step 2】构建节点")
        self._build_idea_nodes(papers)
        self._build_pattern_nodes(patterns)
        self._build_domain_nodes(papers)
        self._build_paper_nodes(papers, reviews, paper_to_pattern)

        # Step 2.5: 填充 Idea -> Pattern 映射
        self._link_idea_pattern()

        # Step 3: 保存节点
        print("\n【Step 3】保存节点")
        self._save_nodes()

        # Step 4: 统计
        print("\n【Step 4】统计信息")
        self._update_stats()
        self._save_stats()
        self._print_stats()

        print("\n" + "=" * 60)
        print("✅ 知识图谱构建完成!")
        print("=" * 60)

    # ===================== 数据加载 =====================

    def _load_papers(self) -> List[Dict]:
        """加载所有论文数据"""
        papers = []
        for conference in CONFERENCES:
            conf_dir = DATA_DIR / conference
            if not conf_dir.exists():
                continue

            all_papers_file = conf_dir / "_all_paper_nodes.json"
            if all_papers_file.exists():
                with open(all_papers_file, 'r', encoding='utf-8') as f:
                    conf_papers = json.load(f)
                    papers.extend(conf_papers)
                    print(f"  📂 {conference}: {len(conf_papers)} 篇")
            else:
                count = 0
                for file in conf_dir.glob("*_paper_node.json"):
                    if file.name.startswith("_"):
                        continue
                    with open(file, 'r', encoding='utf-8') as f:
                        papers.append(json.load(f))
                        count += 1
                if count > 0:
                    print(f"  📂 {conference}: {count} 篇")
        return papers

    def _load_patterns(self) -> List[Dict]:
        """加载 Pattern 数据"""
        if not PATTERNS_FILE.exists():
            print(f"⚠️  Pattern 文件不存在: {PATTERNS_FILE}")
            return []
        with open(PATTERNS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _load_reviews(self) -> List[Dict]:
        """加载所有 Review 数据"""
        reviews = []
        for conference in CONFERENCES:
            conf_dir = DATA_DIR / conference
            if not conf_dir.exists():
                continue

            all_reviews_file = conf_dir / "_all_review_nodes.json"
            if all_reviews_file.exists():
                with open(all_reviews_file, 'r', encoding='utf-8') as f:
                    conf_reviews = json.load(f)
                    reviews.extend(conf_reviews)
        return reviews

    def _load_paper_to_pattern(self) -> Dict[str, int]:
        """加载 Paper → Pattern 映射"""
        mapping_file = OUTPUT_DIR / 'paper_to_pattern.json'
        if not mapping_file.exists():
            print(f"⚠️  paper_to_pattern.json 不存在，Paper 节点将无 pattern_ids")
            return {}
        with open(mapping_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    # ===================== 构建节点 =====================

    def _build_idea_nodes(self, papers: List[Dict]):
        """构建 Idea 节点（第一遍：仅收集 Idea 和 Paper 映射）"""
        print("\n💡 构建 Idea 节点...")

        for paper in papers:
            ideal_info = paper.get('ideal', {})
            core_idea = ideal_info.get('core_idea', '')

            if not core_idea:
                continue

            # 用 hash 去重
            idea_hash = hashlib.md5(core_idea.encode()).hexdigest()[:16]

            if idea_hash not in self.idea_map:
                idea_id = f"idea_{len(self.idea_nodes)}"
                self.idea_map[idea_hash] = idea_id

                self.idea_nodes.append({
                    'idea_id': idea_id,
                    'description': core_idea,
                    'tech_stack': ideal_info.get('tech_stack', []),
                    'input_type': ideal_info.get('input_type', ''),
                    'output_type': ideal_info.get('output_type', ''),
                    'source_paper_ids': [paper.get('paper_id', '')],  # 记录来源
                    'pattern_ids': []  # 占位，稍后填充
                })
            else:
                # 已存在的 Idea，追加 paper_id
                idea_id = self.idea_map[idea_hash]
                for idea_node in self.idea_nodes:
                    if idea_node['idea_id'] == idea_id:
                        if paper.get('paper_id', '') not in idea_node['source_paper_ids']:
                            idea_node['source_paper_ids'].append(paper.get('paper_id', ''))
                        break

        print(f"  ✓ 创建 {len(self.idea_nodes)} 个 Idea 节点")

    def _build_pattern_nodes(self, patterns: List[Dict]):
        """构建 Pattern 节点（直接使用 patterns_structured.json）"""
        print("\n📋 构建 Pattern 节点...")

        for pattern in patterns:
            pattern_id = pattern.get('pattern_id')
            self.pattern_map[pattern_id] = f"pattern_{pattern_id}"

            # 提取关键信息
            self.pattern_nodes.append({
                'pattern_id': f"pattern_{pattern_id}",
                'name': pattern.get('pattern_name', ''),
                'summary': pattern.get('pattern_summary', ''),
                'writing_guide': pattern.get('writing_guide', ''),
                'cluster_size': pattern.get('metadata', {}).get('cluster_size', 0),
                'coherence_score': pattern.get('metadata', {}).get('coherence_score', 0),
                'paper_ids': pattern.get('metadata', {}).get('all_paper_ids', []),

                # 精简的 Skeleton 和 Trick 统计
                'skeleton_count': len(pattern.get('skeleton_examples', [])),
                'trick_count': len(pattern.get('common_tricks', [])),
                'top_tricks': [
                    {
                        'name': t.get('trick_name', ''),
                        'frequency': t.get('frequency', 0),
                        'percentage': t.get('percentage', '')
                    }
                    for t in pattern.get('common_tricks', [])[:5]  # 仅保留 Top 5
                ]
            })

        print(f"  ✓ 创建 {len(self.pattern_nodes)} 个 Pattern 节点")

    def _build_domain_nodes(self, papers: List[Dict]):
        """构建 Domain 节点"""
        print("\n🌐 构建 Domain 节点...")

        domain_stats = defaultdict(lambda: {
            'paper_count': 0,
            'research_objects': set(),
            'core_techniques': set(),
            'applications': set()
        })

        for paper in papers:
            domain_info = paper.get('domain', {})
            domains_list = domain_info.get('domains', [])

            for domain_name in domains_list:
                if not domain_name:
                    continue

                domain_stats[domain_name]['paper_count'] += 1

                # 聚合信息
                if domain_info.get('research_object'):
                    domain_stats[domain_name]['research_objects'].add(
                        domain_info['research_object']
                    )
                if domain_info.get('core_technique'):
                    domain_stats[domain_name]['core_techniques'].add(
                        domain_info['core_technique']
                    )
                if domain_info.get('application'):
                    domain_stats[domain_name]['applications'].add(
                        domain_info['application']
                    )

        # 生成 Domain 节点
        for domain_name, stats in domain_stats.items():
            domain_id = f"domain_{len(self.domain_nodes)}"
            self.domain_map[domain_name] = domain_id

            self.domain_nodes.append({
                'domain_id': domain_id,
                'name': domain_name,
                'paper_count': stats['paper_count'],
                'research_objects': list(stats['research_objects']),
                'core_techniques': list(stats['core_techniques']),
                'applications': list(stats['applications'])
            })

        print(f"  ✓ 创建 {len(self.domain_nodes)} 个 Domain 节点")

    def _build_paper_nodes(self, papers: List[Dict], reviews: List[Dict], paper_to_pattern: Dict[str, int]):
        """构建 Paper 节点（嵌入 Skeleton, Trick, Review）"""
        print("\n📄 构建 Paper 节点...")

        # 构建 review 映射
        review_map = defaultdict(list)
        for review in reviews:
            paper_id = review.get('paper_id', '')
            if paper_id:
                review_map[paper_id].append({
                    'reviewer': review.get('reviewer', ''),
                    'overall_score': review.get('overall_score', ''),
                    'confidence': review.get('confidence', ''),
                    'summary': review.get('paper_summary', '')[:300],
                    'strengths': review.get('strengths', '')[:300],
                    'weaknesses': review.get('weaknesses', '')[:300]
                })

        for paper in papers:
            paper_id = paper.get('paper_id', '')
            self.paper_map[paper_id] = paper_id

            # 提取基础信息
            ideal_info = paper.get('ideal', {})
            domain_info = paper.get('domain', {})
            skeleton = paper.get('skeleton', {})
            tricks = paper.get('tricks', [])

            # 构建 Paper 节点
            self.paper_nodes.append({
                'paper_id': paper_id,
                'title': paper.get('title', ''),
                'conference': paper.get('conference', ''),

                # Idea 信息
                'idea': {
                    'core_idea': ideal_info.get('core_idea', ''),
                    'tech_stack': ideal_info.get('tech_stack', []),
                    'input_type': ideal_info.get('input_type', ''),
                    'output_type': ideal_info.get('output_type', '')
                },

                # Domain 信息
                'domains': domain_info.get('domains', []),

                # Skeleton 信息（完整保留）
                'skeleton': {
                    'problem_framing': skeleton.get('problem_framing', ''),
                    'gap_pattern': skeleton.get('gap_pattern', ''),
                    'method_story': skeleton.get('method_story', ''),
                    'experiments_story': skeleton.get('experiments_story', '')
                },

                # Tricks 信息（完整保留）
                'tricks': [
                    {
                        'name': t.get('name', ''),
                        'type': t.get('type', ''),
                        'description': t.get('description', ''),
                        'purpose': t.get('purpose', ''),
                        'location': t.get('location', '')
                    }
                    for t in tricks
                ],

                # Review 信息
                'reviews': review_map.get(paper_id, []),

                # Pattern 归属（从 paper_to_pattern 映射获取）
                'pattern_ids': [f"pattern_{paper_to_pattern[paper_id]}"] if paper_id in paper_to_pattern else []
            })

        print(f"  ✓ 创建 {len(self.paper_nodes)} 个 Paper 节点")

    def _link_idea_pattern(self):
        """填充 Idea 节点的 pattern_ids（通过 Paper 中转）"""
        print("\n🔗 构建 Idea -> Pattern 映射...")

        # 遍历每个 Idea
        for idea_node in self.idea_nodes:
            idea_id = idea_node['idea_id']
            pattern_ids_set = set()

            # 找到所有实现该 Idea 的 Paper
            for paper_id in idea_node['source_paper_ids']:
                # 从 Paper 中获取 pattern_ids
                for paper_node in self.paper_nodes:
                    if paper_node['paper_id'] == paper_id:
                        pattern_ids_set.update(paper_node['pattern_ids'])
                        break

            # 填充到 Idea 节点
            idea_node['pattern_ids'] = sorted(list(pattern_ids_set))

        total_connections = sum(len(idea['pattern_ids']) for idea in self.idea_nodes)
        print(f"  ✓ 共建立 {total_connections} 个 Idea->Pattern 连接")
        print(f"  ✓ 平均每个 Idea 连接 {total_connections/len(self.idea_nodes):.1f} 个 Pattern")

    # ===================== 保存和统计 =====================

    def _save_nodes(self):
        """保存所有节点到独立文件"""
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # 保存 Idea 节点
        with open(NODES_IDEA, 'w', encoding='utf-8') as f:
            json.dump(self.idea_nodes, f, ensure_ascii=False, indent=2)
        print(f"  ✓ {NODES_IDEA}")

        # 保存 Pattern 节点
        with open(NODES_PATTERN, 'w', encoding='utf-8') as f:
            json.dump(self.pattern_nodes, f, ensure_ascii=False, indent=2)
        print(f"  ✓ {NODES_PATTERN}")

        # 保存 Domain 节点
        with open(NODES_DOMAIN, 'w', encoding='utf-8') as f:
            json.dump(self.domain_nodes, f, ensure_ascii=False, indent=2)
        print(f"  ✓ {NODES_DOMAIN}")

        # 保存 Paper 节点
        with open(NODES_PAPER, 'w', encoding='utf-8') as f:
            json.dump(self.paper_nodes, f, ensure_ascii=False, indent=2)
        print(f"  ✓ {NODES_PAPER}")

    def _update_stats(self):
        """更新统计信息"""
        self.stats.ideas = len(self.idea_nodes)
        self.stats.patterns = len(self.pattern_nodes)
        self.stats.domains = len(self.domain_nodes)
        self.stats.papers = len(self.paper_nodes)
        self.stats.total_nodes = (
            self.stats.ideas +
            self.stats.patterns +
            self.stats.domains +
            self.stats.papers
        )

    def _save_stats(self):
        """保存统计信息"""
        with open(STATS_FILE, 'w', encoding='utf-8') as f:
            json.dump(asdict(self.stats), f, ensure_ascii=False, indent=2)
        print(f"  ✓ {STATS_FILE}")

    def _print_stats(self):
        """打印统计信息"""
        print("\n📊 知识图谱统计 (V2):")
        print("-" * 40)
        print(f"  总节点数:  {self.stats.total_nodes}")
        print(f"  Idea:      {self.stats.ideas}")
        print(f"  Pattern:   {self.stats.patterns}")
        print(f"  Domain:    {self.stats.domains}")
        print(f"  Paper:     {self.stats.papers}")
        print("-" * 40)


def main():
    """主函数"""
    builder = KnowledgeGraphBuilderV2()
    builder.build()


if __name__ == '__main__':
    main()