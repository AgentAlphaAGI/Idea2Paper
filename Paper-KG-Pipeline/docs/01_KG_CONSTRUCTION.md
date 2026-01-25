# 知识图谱构建文档

## 📋 概述

本文档详细说明了 Idea2Paper 项目中知识图谱的构建过程,包括数据源、节点、边的定义、构建流程、参数配置和运行方式。

---

## 1. 数据源

### 1.1 输入文件

| 文件 | 路径 | 说明 | 数据量 |
|------|------|------|--------|
| **assignments.jsonl** | `data/ICLR_25/assignments.jsonl` | Paper到Pattern的分配关系 | 8,285条 |
| **cluster_library_sorted.jsonl** | `data/ICLR_25/cluster_library_sorted.jsonl` | Pattern Cluster信息 | 124条 |
| **iclr_patterns_full.jsonl** | `data/ICLR_25/iclr_patterns_full.jsonl` | Pattern详细属性(英文完整版) | 8,310条 |

### 1.2 数据结构示例

**assignments.jsonl**:
```json
{
  "paper_id": "RUzSobdYy0V",
  "paper_title": "Quantifying and Mitigating...",
  "global_pattern_id": "g0",
  "pattern_id": "p0",
  "domain": "Fairness & Accountability",
  "sub_domains": ["Label Noise", "Disparity Metrics"],
  "cluster_id": 9,
  "cluster_prob": 0.384
}
```

**cluster_library_sorted.jsonl**:
```json
{
  "cluster_id": 24,
  "cluster_name": "Reframing Graph Learning Scalability",
  "size": 331,
  "coherence": {
    "centroid_mean": 0.668,
    "pairwise_sample_mean": 0.461
  },
  "exemplars": [...]
}
```

---

## 2. 节点定义

### 2.1 节点类型概览

| 节点类型 | 数量 | 主要数据源 | 作用 |
|---------|------|-----------|------|
| **Idea** | 8,284 | `iclr_patterns_full.jsonl` | 论文的核心创新点 |
| **Pattern** | 124 | `cluster_library_sorted.jsonl` | 写作套路/方法模板 |
| **Domain** | 98 | `assignments.jsonl`(聚合) | 研究领域 |
| **Paper** | 8,285 | `assignments.jsonl` + pattern details | 具体论文 |

### 2.2 Pattern节点

**数据源**: `cluster_library_sorted.jsonl` + LLM增强

**关键字段**:
```json
{
  "pattern_id": "pattern_24",
  "cluster_id": 24,
  "name": "Reframing Graph Learning Scalability",
  "size": 331,
  "domain": "Machine Learning",
  "sub_domains": ["Graph Neural Networks", ...],
  "coherence": {...},

  "summary": {
    "representative_ideas": ["idea1", "idea2", ...],
    "common_problems": ["problem1", ...],
    "solution_approaches": ["solution1", ...],
    "story": ["story1", ...]
  },

  "llm_enhanced_summary": {
    "representative_ideas": "归纳性总结(单句)...",
    "common_problems": "归纳性总结(单句)...",
    "solution_approaches": "归纳性总结(单句)...",
    "story": "归纳性总结(单句)..."
  },

  "llm_enhanced": true,
  "exemplar_count": 6
}
```

**构建逻辑**:
```python
def _build_pattern_nodes(clusters):
    for cluster in clusters:
        if cluster_id == -1:
            continue  # 跳过未分配

        pattern_node = {
            'pattern_id': f"pattern_{cluster_id}",
            'name': cluster['cluster_name'],
            'size': cluster['size'],
            'coherence': cluster['coherence'],
            'summary': extract_from_exemplars(cluster)
        }
```

### 2.3 Idea节点

**数据源**: `iclr_patterns_full.jsonl`

**关键字段**:
```json
{
  "idea_id": "idea_0",
  "description": "通过分析标签错误对群体差异指标的影响...",
  "base_problem": "在群体差异指标评估中...",
  "solution_pattern": "提出一种方法估计...",
  "story": "将标签错误问题从模型性能影响扩展到...",
  "application": "高风险决策系统的公平性审计...",
  "domain": "Fairness & Accountability",
  "sub_domains": ["Label Noise", ...],
  "source_paper_ids": ["RUzSobdYy0V"],
  "pattern_ids": ["pattern_9"]
}
```

**去重策略**: MD5 hash前16位

**构建逻辑**:
```python
def _build_idea_nodes(pattern_details):
    for paper_id, details in pattern_details.items():
        idea_text = details['idea']
        idea_hash = hashlib.md5(idea_text.encode()).hexdigest()[:16]

        if idea_hash not in self.idea_map:
            idea_node = {
                'idea_id': f"idea_{len(self.idea_nodes)}",
                'description': idea_text,
                ...
            }
```

### 2.4 Domain节点

**数据源**: `assignments.jsonl`(聚合)

**关键字段**:
```json
{
  "domain_id": "domain_0",
  "name": "Fairness & Accountability",
  "paper_count": 69,
  "sub_domains": ["Label Noise", "Bias Mitigation", ...],
  "related_pattern_ids": ["pattern_9", "pattern_15", ...],
  "sample_paper_ids": ["RUzSobdYy0V", ...]
}
```

**构建逻辑**:
```python
def _build_domain_nodes(assignments):
    domain_stats = defaultdict(lambda: {
        'paper_count': 0,
        'sub_domains': set(),
        'related_patterns': set()
    })

    for assignment in assignments:
        domain = assignment['domain']
        domain_stats[domain]['paper_count'] += 1
        domain_stats[domain]['sub_domains'].update(assignment['sub_domains'])
```

### 2.5 Paper节点

**数据源**: `assignments.jsonl` + `iclr_patterns_full.jsonl`

**关键字段**:
```json
{
  "paper_id": "RUzSobdYy0V",
  "title": "Quantifying and Mitigating...",
  "global_pattern_id": "g0",
  "cluster_id": 9,
  "cluster_prob": 0.384,
  "domain": "Fairness & Accountability",
  "sub_domains": [...],
  "idea": "核心想法描述(字符串)",
  "pattern_details": {...},
  "pattern_id": "pattern_9",
  "idea_id": "idea_0",
  "domain_id": "domain_0"
}
```

---

## 3. 边定义

### 3.1 边分类

| 边类型 | 用途 | 数量 |
|--------|------|------|
| **基础连接边** | 建立实体间基本关系 | ~25,000 |
| **召回辅助边** | 支持三路召回策略 | ~420,000 |

### 3.2 基础连接边

#### (1) Paper → Idea (`implements`)
```python
G.add_edge(
    paper['paper_id'],
    paper['idea_id'],
    relation='implements'
)
```

#### (2) Paper → Pattern (`uses_pattern`)
```python
G.add_edge(
    paper['paper_id'],
    paper['pattern_id'],
    relation='uses_pattern',
    quality=paper_quality  # [0, 1]
)
```

**质量评分计算**:
```python
def _get_paper_quality(paper):
    reviews = paper.get('reviews', [])
    if reviews:
        scores = [r['overall_score'] for r in reviews]
        avg_score = np.mean(scores)
        return (avg_score - 1) / 9  # 归一化到[0,1]
    return 0.5  # 默认值(V3当前无review数据)
```

#### (3) Paper → Domain (`in_domain`)
```python
G.add_edge(
    paper['paper_id'],
    paper['domain_id'],
    relation='in_domain'
)
```

### 3.3 召回辅助边

#### (1) Idea → Domain (`belongs_to`)

**权重定义**: Idea相关Paper在该Domain的占比

```python
for idea in ideas:
    domain_counts = defaultdict(int)
    for paper_id in idea['source_paper_ids']:
        paper = paper_id_to_paper[paper_id]
        domain_counts[paper['domain_id']] += 1

    total_papers = len(idea['source_paper_ids'])
    for domain_id, count in domain_counts.items():
        weight = count / total_papers

        G.add_edge(
            idea['idea_id'],
            domain_id,
            relation='belongs_to',
            weight=weight,  # [0, 1]
            paper_count=count
        )
```

#### (2) Pattern → Domain (`works_well_in`)

**权重定义**:
- `effectiveness`: Pattern在该Domain的效果增益(相对基线) [-1, 1]
- `confidence`: 基于样本数的置信度 [0, 1]

```python
for pattern in patterns:
    domain_papers = defaultdict(list)
    for paper_id in pattern['sample_paper_ids']:
        paper = paper_id_to_paper[paper_id]
        domain_papers[paper['domain_id']].append(paper)

    for domain_id, papers in domain_papers.items():
        qualities = [_get_paper_quality(p) for p in papers]
        avg_quality = np.mean(qualities)

        all_domain_papers = get_papers_in_domain(domain_id)
        domain_baseline = np.mean([_get_paper_quality(p) for p in all_domain_papers])

        effectiveness = avg_quality - domain_baseline  # [-1, 1]
        frequency = len(papers)
        confidence = min(frequency / 20, 1.0)  # [0, 1]

        G.add_edge(
            pattern['pattern_id'],
            domain_id,
            relation='works_well_in',
            frequency=frequency,
            effectiveness=effectiveness,
            confidence=confidence
        )
```

#### (3) Idea → Paper (`similar_to_paper`)

**注意**: 此边在V3.1版本中**已预构建但未直接使用**。路径3召回改为**实时计算**用户Idea与Paper Title的相似度。

---

## 4. 构建流程

### 4.1 整体流程

```
┌─────────────────────────────────────────────────────────────┐
│               【知识图谱构建完整流程】                        │
└─────────────────────────────────────────────────────────────┘

【阶段1: 数据加载】(约1秒)
    │
    ├─ 加载 assignments.jsonl (8,285篇论文)
    ├─ 加载 cluster_library_sorted.jsonl (124个Pattern Cluster)
    └─ 加载 iclr_patterns_full.jsonl (8,310条Pattern详情)
    │
    ▼

【阶段2: 节点构建】(约2分钟)
    │
    ├─ 1. Pattern节点 (124个)
    │     ├─ 从cluster_library提取基础信息
    │     ├─ 提取exemplars的ideas/problems/solutions/stories
    │     └─ 生成初步Pattern节点
    │     ↓
    ├─ 2. LLM增强Pattern (124个,约10分钟)
    │     ├─ 为每个Pattern调用LLM
    │     ├─ 生成归纳性总结(4个维度)
    │     │   ├─ representative_ideas
    │     │   ├─ common_problems
    │     │   ├─ solution_approaches
    │     │   └─ story
    │     └─ 添加llm_enhanced_summary字段
    │     ↓
    ├─ 3. Idea节点 (8,284个)
    │     ├─ 从pattern_details提取idea字段
    │     ├─ MD5 hash去重
    │     └─ 提取base_problem/solution_pattern/story/application
    │     ↓
    ├─ 4. Domain节点 (98个)
    │     ├─ 从assignments聚合domain信息
    │     ├─ 收集sub_domains
    │     ├─ 统计paper_count
    │     └─ 关联related_pattern_ids
    │     ↓
    └─ 5. Paper节点 (8,285个)
          ├─ 合并assignments和pattern_details
          ├─ 提取title/domain/sub_domains/idea
          └─ 保留cluster_id/global_pattern_id
    │
    ▼

【阶段3: 建立关联】(约1秒)
    │
    ├─ Paper → Pattern关联
    │    └─ 通过cluster_id映射到pattern_id
    │        覆盖率: 5,981/8,285 (72.2%)
    │
    ├─ Paper → Idea关联
    │    └─ 通过idea文本的MD5 hash映射
    │        覆盖率: 8,284/8,285 (100%)
    │
    ├─ Paper → Domain关联
    │    └─ 通过domain名称映射到domain_id
    │        覆盖率: 8,285/8,285 (100%)
    │
    └─ Idea → Pattern关联
         └─ 通过Paper中转建立连接
             ├─ 收集每个Idea关联的所有Paper
             ├─ 提取这些Paper的pattern_id
             └─ 填充Idea.pattern_ids字段
             平均每个Idea关联0.7个Pattern
    │
    ▼

【阶段4: 保存节点】(约1秒)
    │
    ├─ 输出 nodes_idea.json (8,284个)
    ├─ 输出 nodes_pattern.json (124个)
    ├─ 输出 nodes_domain.json (98个)
    ├─ 输出 nodes_paper.json (8,285个)
    └─ 输出 knowledge_graph_stats.json
    │
    ▼

【阶段5: 构建边】(约2-3分钟)
    │
    ├─ 基础连接边
    │    ├─ Paper → Idea (implements) 8,284条
    │    ├─ Paper → Pattern (uses_pattern) 5,981条
    │    └─ Paper → Domain (in_domain) 8,285条
    │
    ├─ 召回辅助边 - 路径2
    │    ├─ Idea → Domain (belongs_to)
    │    │   └─ 权重: Idea相关Paper在该Domain的占比
    │    │
    │    └─ Pattern → Domain (works_well_in)
    │        ├─ effectiveness: Pattern在Domain的效果增益
    │        └─ confidence: 基于样本数的置信度
    │
    └─ 召回辅助边 - 路径3
         └─ (实时计算,不预构建)
    │
    ▼

【阶段6: 保存图谱】(约1秒)
    │
    ├─ 输出 edges.json
    └─ 输出 knowledge_graph_v2.gpickle
    │
    ▼

✅ 构建完成
   ├─ 总节点: 16,791个
   ├─ 总边数: 444,872条
   └─ 总耗时: 约15-18分钟
```

### 4.2 关键步骤

#### Step 1: 加载数据
```python
assignments = _load_assignments()      # 8,285条
clusters = _load_clusters()            # 124个
pattern_details = _load_pattern_details()  # 8,310条
```

#### Step 2: 构建节点
```python
_build_pattern_nodes(clusters)         # 124个Pattern
_enhance_patterns_with_llm(clusters)   # LLM增强
_build_idea_nodes(pattern_details)     # 8,284个Idea
_build_domain_nodes(assignments)       # 98个Domain
_build_paper_nodes(assignments, pattern_details)  # 8,285个Paper
```

#### Step 3: 建立关联
```python
_link_paper_to_pattern(assignments)    # Paper → Pattern
_link_paper_to_idea()                  # Paper → Idea
_link_paper_to_domain()                # Paper → Domain
_link_idea_to_pattern()                # Idea → Pattern(通过Paper中转)
```

#### Step 4: 构建边
```python
_build_paper_edges()                   # 基础连接边
_build_idea_belongs_to_domain_edges()  # 召回边-路径2
_build_pattern_works_well_in_domain_edges()
_build_idea_similar_to_paper_edges()   # 召回边-路径3
```

#### Step 5: 保存结果
```python
_save_nodes()  # 保存4类节点JSON
_save_edges()  # 保存edges.json
_save_graph()  # 保存knowledge_graph_v2.gpickle
```

---

## 5. LLM增强机制

### 5.1 增强目标

为每个Pattern cluster生成归纳性总结,既保留具体示例,也提供全局概述。

### 5.2 Prompt设计

```python
def _build_llm_prompt_for_pattern(pattern_node, exemplars):
    prompt = f"""
你是一个学术研究专家。请基于以下{len(exemplars)}篇论文的Pattern信息，
为Pattern Cluster "{pattern_node['name']}" 生成归纳性总结。

【论文Pattern信息】
{format_exemplars(exemplars)}

【任务】
请生成4个维度的归纳性总结(每个1句话，80-120字)：
1. representative_ideas: 代表性研究想法
2. common_problems: 共同解决的问题
3. solution_approaches: 解决方法特点
4. story: 研究叙事框架

返回JSON格式。
"""
    return prompt
```

### 5.3 API配置

```python
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")
LLM_API_URL = "https://api.siliconflow.cn/v1/chat/completions"
LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"
```

---

## 6. 参数配置

### 6.1 路径配置

```python
# 数据输入路径
DATA_DIR = PROJECT_ROOT / "data" / "ICLR_25"
ASSIGNMENTS_FILE = DATA_DIR / "assignments.jsonl"
CLUSTER_LIBRARY_FILE = DATA_DIR / "cluster_library_sorted.jsonl"
PATTERN_DETAILS_FILE = DATA_DIR / "iclr_patterns_full.jsonl"

# 输出路径
OUTPUT_DIR = PROJECT_ROOT / "output"
NODES_IDEA = OUTPUT_DIR / "nodes_idea.json"
NODES_PATTERN = OUTPUT_DIR / "nodes_pattern.json"
NODES_DOMAIN = OUTPUT_DIR / "nodes_domain.json"
NODES_PAPER = OUTPUT_DIR / "nodes_paper.json"
EDGES_FILE = OUTPUT_DIR / "edges.json"
GRAPH_FILE = OUTPUT_DIR / "knowledge_graph_v2.gpickle"
```

### 6.2 LLM配置

```python
# API密钥(环境变量)
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")

# API端点
LLM_API_URL = "https://api.siliconflow.cn/v1/chat/completions"

# 模型选择
LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"  # 节点构建
# 或 "Qwen/Qwen3-14B"  # Pipeline生成
```

### 6.3 边构建配置

```python
# Pattern-Domain边权重计算
BASELINE_SAMPLE_SIZE = 20  # confidence达到1.0的样本数阈值

# Paper质量评分(无review数据时)
DEFAULT_PAPER_QUALITY = 0.5  # [0, 1]
```

---

## 7. 运行方式

### 7.1 环境准备

**依赖安装**:
```bash
cd /Users/gaoge/code/mycode/Idea2Paper/Paper-KG-Pipeline
pip install -r requirements.txt
```

**环境变量设置**:
```bash
export SILICONFLOW_API_KEY="your_api_key_here"
```

### 7.2 构建节点

**命令**:
```bash
python scripts/build_entity_v3.py
```

**输出**:
```
output/
├── nodes_idea.json           # 8,284个Idea节点
├── nodes_pattern.json        # 124个Pattern节点
├── nodes_domain.json         # 98个Domain节点
├── nodes_paper.json          # 8,285个Paper节点
└── knowledge_graph_stats.json # 统计信息
```

**执行时间**: 约10-15分钟(含LLM增强)

### 7.3 构建边

**命令**:
```bash
python scripts/build_edges.py
```

**输出**:
```
output/
├── edges.json                # 边数据(JSON格式)
└── knowledge_graph_v2.gpickle # 完整图谱(NetworkX格式)
```

**执行时间**: 约2-3分钟

### 7.4 验证图谱

**Python交互式验证**:
```python
import json
import pickle

# 加载节点
with open('output/nodes_pattern.json') as f:
    patterns = json.load(f)
print(f"Pattern数量: {len(patterns)}")

# 加载图谱
with open('output/knowledge_graph_v2.gpickle', 'rb') as f:
    G = pickle.load(f)
print(f"节点数: {G.number_of_nodes()}")
print(f"边数: {G.number_of_edges()}")
```

---

## 8. 输出统计

### 8.1 节点统计

```
总节点数:  9,411
  - Idea:      8,284 (100%覆盖率)
  - Pattern:   124
  - Domain:    98
  - Paper:     8,285
```

### 8.2 边统计

```
【基础连接边】
  Paper→Idea:      8,284 条
  Paper→Pattern:   5,981 条 (72.2%覆盖率)
  Paper→Domain:    8,285 条

【召回边 - 路径2】
  Idea→Domain:     ~15,000 条
  Pattern→Domain:  ~3,500 条

【召回边 - 路径3】
  (实时计算，无预构建边)

总边数: 444,872 条
```

### 8.3 数据质量

```
✅ Idea覆盖率: 100% (8,284/8,285)
✅ Pattern覆盖率: 72.2% (基于cluster分配)
✅ LLM增强: 124/124 Pattern节点
✅ 聚类质量: 可量化评估(coherence指标)
```

---

## 9. 故障排查

### 9.1 常见问题

**Q: LLM API调用失败**
```
错误: Connection timeout / API key invalid
解决:
1. 检查网络连接
2. 验证SILICONFLOW_API_KEY环境变量
3. 检查API额度
```

**Q: 内存不足**
```
错误: MemoryError
解决:
1. 减少LLM增强的exemplar数量(默认20→10)
2. 分批处理Pattern节点
```

**Q: 输出文件已存在**
```
行为: 自动覆盖
建议: 备份重要的output/文件后再运行
```

### 9.2 日志查看

构建过程会输出详细日志:
```
🚀 开始构建知识图谱 V3 (ICLR数据源)
【Step 1】加载数据
  ✅ 加载 8285 篇论文分配
【Step 2】构建节点
  ✓ 创建 124 个 Pattern 节点
  ✓ LLM增强: 124/124 完成
【Step 3】建立节点关联
  ✓ 共建立 8284 个 Idea->Pattern 连接
【Step 4】保存节点
【Step 5】统计信息
✅ 知识图谱构建完成!
```

---

## 10. 扩展与优化

### 10.1 数据源扩展

**添加新会议数据**:
1. 准备与ICLR格式一致的JSONL文件
2. 修改`DATA_DIR`路径
3. 重新运行`build_entity_v3.py`

### 10.2 Review数据补充

**当前状态**: Paper节点暂无review数据，质量分默认0.5

**改进方案**:
```python
# 在assignments.jsonl中添加reviews字段
{
  "paper_id": "xxx",
  "reviews": [
    {"overall_score": 7},
    {"overall_score": 8}
  ],
  ...
}

# 质量评分将自动生效
```

### 10.3 性能优化

**LLM增强加速**:
```python
# 并行处理Pattern
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=5) as executor:
    futures = [executor.submit(_enhance_single_pattern, p)
               for p in pattern_nodes]
```

---

## 11. 总结

### 核心成果

✅ 成功基于ICLR数据源构建知识图谱
✅ 实现100% Idea覆盖率
✅ 引入LLM增强,为每个Pattern生成归纳性总结
✅ 保留聚类质量指标(coherence)
✅ 代码模块化,易于扩展

### 技术特性

✅ **LLM集成**: 使用SiliconFlow API增强Pattern描述
✅ **Prompt工程**: 结构化Prompt设计
✅ **容错机制**: 自动JSON解析和修复
✅ **双层描述**: 具体示例+全局总结

### 扩展性

✅ 支持增量更新
✅ 可适配其他会议数据源
✅ 为召回系统提供完整节点基础

---

**生成时间**: 2026-01-25
**版本**: V3.1
**作者**: Idea2Paper Team

