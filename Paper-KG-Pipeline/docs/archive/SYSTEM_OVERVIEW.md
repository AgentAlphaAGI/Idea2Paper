# Idea2Pattern 知识图谱召回系统 - 完整说明

## 📋 目录

1. [系统概述](#系统概述)
2. [数据源与节点构建](#数据源与节点构建)
3. [知识图谱边构建](#知识图谱边构建)
4. [三路召回策略](#三路召回策略)
5. [多路融合与精排](#多路融合与精排)
6. [当前局限与改进方向](#当前局限与改进方向)

---

## 系统概述

### 核心目标
**输入**: 用户的研究 Idea 描述
**输出**: Top-10 最相关的研究 Pattern (写作套路/方法模板)

### 技术架构
```
用户 Idea
    ↓
知识图谱 (16,790 节点, 444,872 条边)
    ↓
三路并行召回
    ├─ 路径1: 相似 Idea → Pattern (权重 0.4)
    ├─ 路径2: 相关 Domain → Pattern (权重 0.2)
    └─ 路径3: 相似 Paper → Pattern (权重 0.4)
    ↓
加权融合 + 精排
    ↓
Top-10 Pattern 推荐
```

### 数据规模 (V3版本)
- **数据源**: ICLR 2025 论文数据集
- **Idea 节点**: 8,284 个 (每篇论文的核心想法)
- **Pattern 节点**: 124 个 (基于聚类的写作套路)
- **Domain 节点**: 98 个 (研究领域)
- **Paper 节点**: 8,285 篇论文
- **总边数**: 444,872 条

---

## 数据源与节点构建

### 1. 数据源

#### 输入文件
1. **`assignments.jsonl`** - 论文分配信息
   - `paper_id`: 论文唯一标识
   - `paper_title`: 论文标题
   - `domain`: 主领域 (如 "Natural Language Processing")
   - `sub_domains`: 子领域列表 (如 ["Text Classification", "Transformers"])
   - `cluster_id`: 聚类ID (对应 Pattern)
   - `global_pattern_id`: 全局 Pattern ID

2. **`clusters.jsonl`** - Pattern 聚类信息
   - `cluster_id`: 聚类ID
   - `size`: 聚类大小 (包含多少篇论文)
   - `summary`: Pattern 摘要信息 (示例、技巧等)

3. **`pattern_details.jsonl`** - 论文详细 Pattern
   - `paper_id`: 论文ID
   - `idea`: 核心想法描述 (字符串)
   - `research_patterns`: 研究模式详情 (base_problem, solution_pattern, story 等)

4. **`iclr_patterns_full_cn_912.jsonl`** - LLM 增强的 Pattern 总结
   - `cluster_id`: 聚类ID
   - `representative_ideas`: LLM 生成的归纳总结
   - `common_tricks`: 常见技巧
   - `naming_suggestion`: Pattern 命名建议

### 2. 节点构建流程

#### Pattern 节点 (124个)
**来源**: `clusters.jsonl` + LLM 增强

**构建逻辑**:
```python
for cluster in clusters:
    pattern_node = {
        'pattern_id': f"pattern_{cluster_id}",
        'name': llm_summary['naming_suggestion'],  # LLM生成的名称
        'size': cluster['size'],  # 论文数量
        'summary': cluster['summary'],  # 原始摘要
        'llm_enhanced_summary': {  # LLM增强
            'representative_ideas': "...",
            'common_tricks': [...],
            'application_scenarios': [...]
        }
    }
```

**关键字段**:
- `pattern_id`: 如 `pattern_5`
- `name`: 如 "Reframing Zero-Shot Generalization"
- `size`: 聚类大小,反映该 Pattern 的流行度
- `llm_enhanced`: 是否经过 LLM 增强 (912/124 已增强)

---

#### Idea 节点 (8,284个)
**来源**: `pattern_details.jsonl`

**构建逻辑**:
```python
for paper in pattern_details:
    idea_node = {
        'idea_id': f"idea_{index}",
        'description': paper['idea'],  # 核心想法(字符串)
        'source_paper_ids': [paper['paper_id']],
        'pattern_ids': []  # 后续关联填充
    }
```

**关键字段**:
- `idea_id`: 如 `idea_42`
- `description`: 核心想法描述 (平均长度 150-300 字符)
- `pattern_ids`: 该 Idea 使用的 Pattern 列表 (关联后填充)

**特点**:
- 每篇论文对应一个 Idea 节点
- V3 版本中 `idea` 是简单字符串,而非嵌套字典

---

#### Domain 节点 (98个)
**来源**: `assignments.jsonl` (聚合)

**构建逻辑**:
```python
domain_stats = defaultdict(lambda: {
    'paper_count': 0,
    'sub_domains': set(),
    'related_patterns': set()
})

for assignment in assignments:
    domain = assignment['domain']
    domain_stats[domain]['paper_count'] += 1
    domain_stats[domain]['sub_domains'].update(assignment['sub_domains'])
    domain_stats[domain]['related_patterns'].add(f"pattern_{cluster_id}")

# 生成 Domain 节点
for domain_name, stats in domain_stats.items():
    domain_node = {
        'domain_id': f"domain_{index}",
        'name': domain_name,
        'paper_count': stats['paper_count'],
        'sub_domains': sorted(list(stats['sub_domains'])),
        'related_pattern_ids': sorted(list(stats['related_patterns']))
    }
```

**关键字段**:
- `domain_id`: 如 `domain_0`
- `name`: 如 "Natural Language Processing"
- `paper_count`: 该领域的论文数量
- `sub_domains`: 子领域列表 (通常 10-50 个)
- `related_pattern_ids`: 该领域相关的 Pattern 列表

**示例**:
```json
{
  "domain_id": "domain_1",
  "name": "Computer Vision",
  "paper_count": 1076,
  "sub_domains": ["3D Reconstruction", "Object Detection", "Image Synthesis", ...],
  "related_pattern_ids": ["pattern_10", "pattern_25", ...]
}
```

---

#### Paper 节点 (8,285个)
**来源**: `assignments.jsonl` + `pattern_details.jsonl` (合并)

**构建逻辑**:
```python
for assignment in assignments:
    paper_id = assignment['paper_id']
    details = pattern_details.get(paper_id, {})

    paper_node = {
        'paper_id': paper_id,
        'title': assignment['paper_title'],
        'domain': assignment['domain'],
        'sub_domains': assignment['sub_domains'],
        'idea': details.get('idea', ''),  # 字符串
        'global_pattern_id': assignment['global_pattern_id'],
        'cluster_id': assignment['cluster_id'],

        # 后续关联填充
        'pattern_id': '',
        'idea_id': '',
        'domain_id': ''
    }
```

**关键字段**:
- `paper_id`: 如 `Kn-HA8DFik` (ICLR 论文 ID)
- `title`: 论文标题
- `idea`: 核心想法描述 (字符串,用于相似度计算)
- `pattern_id`: 关联的 Pattern
- `idea_id`: 关联的 Idea
- `domain_id`: 关联的 Domain

**当前缺失字段**:
- `reviews`: ⚠️ 暂无 Review 数据,质量分默认 0.5

---

## 知识图谱边构建

### 边的分类

| 边类型 | 用途 | 数量 |
|--------|------|------|
| **基础连接边** | 建立实体间基本关系 | ~25,000 |
| **召回辅助边** | 支持三路召回策略 | ~420,000 |

---

### 1. 基础连接边

#### (1) Paper → Idea (`implements`)
**用途**: 表示论文实现了某个核心 Idea

**构建逻辑**:
```python
for paper in papers:
    G.add_edge(
        paper['paper_id'],
        paper['idea_id'],
        relation='implements'
    )
```

**权重**: 无权重 (布尔关系)

---

#### (2) Paper → Pattern (`uses_pattern`)
**用途**: 表示论文使用了某个写作 Pattern

**构建逻辑**:
```python
for paper in papers:
    paper_quality = _get_paper_quality(paper)  # 基于 review 评分

    G.add_edge(
        paper['paper_id'],
        paper['pattern_id'],
        relation='uses_pattern',
        quality=paper_quality  # [0, 1]
    )
```

**权重**:
- `quality`: Paper 的综合质量分数 (0-1)
  - **有 Review 数据时**: `(avg_review_score - 1) / 9`
  - **无 Review 数据时** (V3 当前状态): `0.5` (默认值)

**示例**:
```json
{
  "source": "Kn-HA8DFik",
  "target": "pattern_5",
  "relation": "uses_pattern",
  "quality": 0.5
}
```

---

#### (3) Paper → Domain (`in_domain`)
**用途**: 表示论文属于某个研究领域

**构建逻辑**:
```python
for paper in papers:
    G.add_edge(
        paper['paper_id'],
        paper['domain_id'],
        relation='in_domain'
    )
```

**权重**: 无权重 (布尔关系)

---

### 2. 召回辅助边

#### (1) Idea → Domain (`belongs_to`)
**用途**: 支持路径2召回 (领域相关性召回)

**构建逻辑**:
```python
for idea in ideas:
    domain_counts = defaultdict(int)

    # 统计 Idea 相关 Paper 在各 Domain 的分布
    for paper_id in idea['source_paper_ids']:
        paper = paper_id_to_paper[paper_id]
        domain_counts[paper['domain_id']] += 1

    # 创建边,权重为占比
    total_papers = len(idea['source_paper_ids'])
    for domain_id, count in domain_counts.items():
        weight = count / total_papers  # 占比作为权重

        G.add_edge(
            idea['idea_id'],
            domain_id,
            relation='belongs_to',
            weight=weight,  # [0, 1]
            paper_count=count
        )
```

**权重**:
- `weight`: Idea 相关 Paper 在该 Domain 的占比 (0-1)

**示例**:
```json
{
  "source": "idea_42",
  "target": "domain_2",
  "relation": "belongs_to",
  "weight": 0.75,
  "paper_count": 3,
  "total_papers": 4
}
```

---

#### (2) Pattern → Domain (`works_well_in`)
**用途**: 支持路径2召回,表示 Pattern 在某 Domain 的效果

**构建逻辑**:
```python
for pattern in patterns:
    # 按 Domain 分组统计 Pattern 的使用情况
    domain_papers = defaultdict(list)

    for paper_id in pattern['sample_paper_ids']:
        paper = paper_id_to_paper[paper_id]
        domain_papers[paper['domain_id']].append(paper)

    # 为每个 Domain 计算效果指标
    for domain_id, papers in domain_papers.items():
        # 计算 Pattern 在该 Domain 的平均质量
        qualities = [_get_paper_quality(p) for p in papers]
        avg_quality = np.mean(qualities)

        # 计算该 Domain 的基线质量
        all_domain_papers = get_papers_in_domain(domain_id)
        domain_baseline = np.mean([_get_paper_quality(p) for p in all_domain_papers])

        # 效果增益 = Pattern平均质量 - Domain基线
        effectiveness = avg_quality - domain_baseline  # [-1, 1]

        # 置信度 = 基于样本数
        frequency = len(papers)
        confidence = min(frequency / 20, 1.0)  # [0, 1]

        G.add_edge(
            pattern['pattern_id'],
            domain_id,
            relation='works_well_in',
            frequency=frequency,
            effectiveness=effectiveness,
            confidence=confidence,
            avg_quality=avg_quality,
            baseline=domain_baseline
        )
```

**权重**:
- `effectiveness`: Pattern 在该 Domain 的效果增益 (相对基线) [-1, 1]
  - **正值**: Pattern 在该 Domain 效果好于平均水平
  - **负值**: Pattern 在该 Domain 效果低于平均水平
- `confidence`: 基于样本数的置信度 [0, 1]
  - 样本数 ≥ 20 时,置信度达到 1.0
  - 样本数越少,置信度越低

**示例**:
```json
{
  "source": "pattern_5",
  "target": "domain_2",
  "relation": "works_well_in",
  "frequency": 15,
  "effectiveness": 0.12,
  "confidence": 0.75,
  "avg_quality": 0.82,
  "baseline": 0.70
}
```

---

#### (3) Idea → Paper (`similar_to_paper`)
**用途**: 支持路径3召回 (相似 Paper 召回)

**注意**: 该边在当前版本(V3.1)中**已预构建但未直接使用**。路径3召回改为**实时计算**用户Idea与Paper Title的相似度,以避免与路径1重复。

**边构建逻辑** (保留用于未来扩展):
```python
for idea in ideas:
    similarities = []

    # 计算与所有 Paper 的相似度
    for paper in papers:
        similarity = compute_text_similarity(
            idea['description'],
            paper['idea']
        )

        if similarity < 0.1:  # 过滤低相似度
            continue

        paper_quality = _get_paper_quality(paper)
        combined_weight = similarity * paper_quality

        similarities.append({
            'paper_id': paper['paper_id'],
            'similarity': similarity,
            'quality': paper_quality,
            'combined_weight': combined_weight
        })

    # 排序并只保留 Top-50 (避免边过多)
    similarities.sort(key=lambda x: x['combined_weight'], reverse=True)

    for item in similarities[:50]:
        G.add_edge(
            idea['idea_id'],
            item['paper_id'],
            relation='similar_to_paper',
            similarity=item['similarity'],
            quality=item['quality'],
            combined_weight=item['combined_weight']
        )
```

**权重**:
- `similarity`: Idea 与 Paper 的语义相似度 [0, 1]
- `quality`: Paper 质量分数 [0, 1]
- `combined_weight`: `similarity × quality` [0, 1]

**路径3实际召回**:
- 不使用预构建的边,而是实时计算用户Idea与Paper **Title**的相似度
- 这样确保路径1(基于Idea Description)和路径3(基于Paper Title)互补

---

## 三路召回策略

### 设计理念: 三路互补

三路召回从不同维度捕捉用户需求,避免重复和信息冗余:

| 路径 | 匹配对象 | 捕捉维度 | 权重 | 典型场景 |
|------|---------|---------|------|---------|
| **路径1** | Idea Description | **核心思想/概念**相似性 | 0.4 | 用户描述与历史成功案例的核心思路一致 |
| **路径2** | Domain & Sub-domains | **领域泛化**能力 | 0.2 | 用户Idea属于某领域,该领域有验证有效的Pattern |
| **路径3** | Paper Title | **研究主题/具体问题**相似性 | 0.4 | 用户想解决的具体问题与某些论文标题表述类似 |

**互补性说明**:
- **路径1 vs 路径3**:
  - 路径1关注"想法本质"(如 "使用蒸馏提升模型效率")
  - 路径3关注"研究方向"(如 "Cross-Domain Text Classification with Transformers")
  - 即使Idea相同,论文标题可能聚焦不同应用场景
- **路径2的泛化作用**: 即使用户Idea是全新的,只要属于某个成熟领域,也能召回该领域通用的有效Pattern

---

### 路径1: 相似 Idea 召回 (Idea → Idea → Pattern)

#### 召回流程
```
用户 Idea (文本)
    ↓ [粗排] Jaccard 筛选 Top-100
候选 Idea (100个)
    ↓ [精排] Embedding 重排 Top-10
相似 Idea (10个)
    ↓ 直接获取 idea.pattern_ids
Pattern 集合
    ↓ 按相似度加权累加
Top-10 Pattern (得分字典)
```

#### 两阶段召回优化

**为什么需要两阶段?**
- 全量 Embedding 检索: 8,284 次 API 调用,耗时 **~7 分钟** ❌
- 两阶段召回: 100 次 API 调用,耗时 **~10 秒** ✅ (提速 40 倍)

**粗排阶段** (Jaccard):
```python
# 对所有 Idea 快速计算 Jaccard 相似度
coarse_similarities = []
for idea in ideas:  # 8,284 个
    sim = compute_jaccard_similarity(user_idea, idea['description'])
    if sim > 0:
        coarse_similarities.append((idea_id, sim))

# 排序并取 Top-100
coarse_similarities.sort(reverse=True)
candidates = coarse_similarities[:100]
```

**精排阶段** (Embedding):
```python
# 对候选 Idea 使用 Embedding 重新计算
fine_similarities = []
for idea_id, _ in candidates:  # 100 个
    idea = idea_id_to_idea[idea_id]
    sim = compute_embedding_similarity(user_idea, idea['description'])
    if sim > 0:
        fine_similarities.append((idea_id, sim))

# 排序并取 Top-10
fine_similarities.sort(reverse=True)
top_ideas = fine_similarities[:10]
```

**Embedding API**:
- 使用 Qwen3-Embedding-4B 模型
- 计算余弦相似度: `cosine_sim = dot(emb1, emb2) / (norm(emb1) * norm(emb2))`

---

#### Pattern 得分计算

**算分逻辑**:
```python
pattern_scores = defaultdict(float)

for idea_id, similarity in top_10_ideas:
    idea = idea_id_to_idea[idea_id]

    # V3 版本: 直接从 Idea 节点获取 pattern_ids
    for pattern_id in idea['pattern_ids']:
        # 得分 = 相似度 (多个 Idea 使用同一 Pattern 时会累加)
        pattern_scores[pattern_id] += similarity

# 排序并只保留 Top-10
sorted_patterns = sorted(pattern_scores.items(), reverse=True)
top_patterns = dict(sorted_patterns[:10])
```

**关键点**:
- 如果多个相似 Idea 都使用了同一个 Pattern,得分会**累加**
- 最终只保留得分最高的 **Top-10 个 Pattern**

**示例**:
```
用户 Idea: "使用 Transformer 进行文本分类"

相似 Idea_1 (相似度 0.8) → [pattern_5, pattern_10]
相似 Idea_2 (相似度 0.7) → [pattern_5, pattern_20]
相似 Idea_3 (相似度 0.6) → [pattern_10]

路径1得分:
  pattern_5:  0.8 + 0.7 = 1.5
  pattern_10: 0.8 + 0.6 = 1.4
  pattern_20: 0.7 = 0.7
```

---

### 路径2: 领域相关召回 (Idea → Domain → Pattern)

#### 召回流程
```
用户 Idea (文本)
    ↓ 关键词匹配 Domain name
相关 Domain (Top-5)
    ↓ 反向查找 Pattern → Domain 边
在 Domain 中表现好的 Pattern
    ↓ 按 effectiveness & confidence 加权
Top-5 Pattern (得分字典)
```

#### Domain 匹配逻辑

**方法1: 关键词匹配** (优先)
```python
domain_scores = []
user_tokens = set(user_idea.lower().split())

for domain in domains:
    domain_name = domain['name']
    domain_tokens = set(domain_name.lower().split())

    # 简单的词汇重叠
    match_score = len(user_tokens & domain_tokens) / max(len(user_tokens), 1)

    if match_score > 0:
        domain_scores.append((domain_id, match_score))

# 排序并取 Top-5
domain_scores.sort(reverse=True)
top_domains = domain_scores[:5]
```

**方法2: 通过相似 Idea 的 Domain** (备选)
```python
if not domain_scores:  # 如果没有直接匹配
    # 找到最相似的 Idea
    similarities = [(idea, compute_similarity(user_idea, idea['description']))
                    for idea in ideas]
    top_idea = max(similarities, key=lambda x: x[1])[0]

    # 获取该 Idea 的 Domain (通过 belongs_to 边)
    for successor in G.successors(top_idea['idea_id']):
        edge_data = G[top_idea['idea_id']][successor]
        if edge_data['relation'] == 'belongs_to':
            domain_id = successor
            weight = edge_data['weight']
            domain_scores.append((domain_id, weight))
```

---

#### Pattern 得分计算

**算分逻辑**:
```python
pattern_scores = defaultdict(float)

for domain_id, domain_weight in top_5_domains:
    # 反向查找: 哪些 Pattern 在该 Domain 中表现好?
    for predecessor in G.predecessors(domain_id):
        edge_data = G[predecessor][domain_id]

        if edge_data['relation'] == 'works_well_in':
            pattern_id = predecessor
            effectiveness = edge_data['effectiveness']  # [-1, 1]
            confidence = edge_data['confidence']  # [0, 1]

            # 得分 = Domain相关度 × 效果 × 置信度
            # max(effectiveness, 0.1) 避免负值
            score = domain_weight * max(effectiveness, 0.1) * confidence
            pattern_scores[pattern_id] += score

# 排序并只保留 Top-5 (辅助通道)
sorted_patterns = sorted(pattern_scores.items(), reverse=True)
top_patterns = dict(sorted_patterns[:5])
```

**关键点**:
- `effectiveness` 负值时取 0.1,避免惩罚过度
- 最终只保留 **Top-5 个 Pattern** (路径2 是辅助通道)

**示例**:
```
用户 Idea: "使用 distillation 技术进行跨领域文本分类"

匹配 Domain:
  domain_2 (Natural Language Processing, 相关度=0.25)

在 domain_2 中表现好的 Pattern:
  pattern_5  (effectiveness=0.12, confidence=0.75)
  pattern_10 (effectiveness=0.08, confidence=0.60)

路径2得分:
  pattern_5:  0.25 × 0.12 × 0.75 = 0.0225
  pattern_10: 0.25 × 0.10 × 0.60 = 0.0150  (max(0.08, 0.1) = 0.1)
```

---

### 路径3: 相似 Paper 召回 (Idea → Paper → Pattern)

#### 召回流程
```
用户 Idea (文本)
    ↓ [粗排] Jaccard 筛选 Top-100 (基于 Paper Title)
候选 Paper (100个)
    ↓ [精排] Embedding 重排 Top-20 (基于 Paper Title)
相似 Paper (20个)
    ↓ 查找 Paper → Pattern 边
Pattern 集合
    ↓ 按 similarity × quality 加权
Top-10 Pattern (得分字典)
```

**设计理念**:
- **路径1** 使用 Idea Description 计算相似度 → 捕捉**核心思想/概念**的相似性
- **路径3** 使用 Paper Title 计算相似度 → 捕捉**研究主题/具体问题**的相似性
- 两者互补,避免重复

#### 两阶段召回优化

**粗排阶段** (Jaccard):
```python
coarse_similarities = []
for paper in papers:  # 8,285 个
    paper_title = paper['title']  # 使用论文标题
    sim = compute_jaccard_similarity(user_idea, paper_title)

    if sim > 0.05:  # 降低阈值保留更多候选
        coarse_similarities.append((paper_id, sim))

# 排序并取 Top-100
coarse_similarities.sort(reverse=True)
candidates = coarse_similarities[:100]
```

**精排阶段** (Embedding):
```python
fine_similarities = []
for paper_id, _ in candidates:  # 100 个
    paper = paper_id_to_paper[paper_id]
    paper_title = paper['title']  # 使用论文标题

    sim = compute_embedding_similarity(user_idea, paper_title)

    if sim > 0.1:  # 过滤低相似度
        quality = _get_paper_quality(paper)
        combined_weight = sim * quality
        fine_similarities.append((paper_id, sim, quality, combined_weight))

# 按综合权重排序并取 Top-20
fine_similarities.sort(key=lambda x: x[3], reverse=True)
top_papers = fine_similarities[:20]
```

---

#### Pattern 得分计算

**算分逻辑**:
```python
pattern_scores = defaultdict(float)

for paper_id, similarity, paper_quality, combined_weight in top_20_papers:
    # 从图谱中查找 Paper 使用的 Pattern
    for successor in G.successors(paper_id):
        edge_data = G[paper_id][successor]

        if edge_data['relation'] == 'uses_pattern':
            pattern_id = successor
            pattern_quality = edge_data['quality']  # Paper质量

            # 得分 = 相似度 × Paper质量 × Pattern质量
            score = combined_weight * pattern_quality
            pattern_scores[pattern_id] += score

# 排序并只保留 Top-10
sorted_patterns = sorted(pattern_scores.items(), reverse=True)
top_patterns = dict(sorted_patterns[:10])
```

**关键点**:
- 综合考虑 Paper 与用户 Idea 的相似度、Paper 质量、Pattern 质量
- 如果多个相似 Paper 都使用了同一个 Pattern,得分会**累加**
- 最终只保留 **Top-10 个 Pattern**

**示例**:
```
用户 Idea: "使用 Transformer 进行文本分类"

相似 Paper:
  Paper_1 (相似度=0.85, 质量=0.5) → pattern_5 (质量=0.5)
  Paper_2 (相似度=0.78, 质量=0.5) → pattern_5 (质量=0.5)
  Paper_3 (相似度=0.72, 质量=0.5) → pattern_10 (质量=0.5)

路径3得分:
  pattern_5:  (0.85×0.5)×0.5 + (0.78×0.5)×0.5 = 0.2125 + 0.195 = 0.4075
  pattern_10: (0.72×0.5)×0.5 = 0.18
```

---

## 多路融合与精排

### 融合策略

#### 路径权重
```python
PATH1_WEIGHT = 0.4  # 相似 Idea 召回 (重要)
PATH2_WEIGHT = 0.2  # 领域相关召回 (辅助)
PATH3_WEIGHT = 0.4  # 相似 Paper 召回 (重要)
```

**权重设计理由**:
- **路径1 (0.4)**: 直接利用历史成功经验,最可靠
- **路径2 (0.2)**: 领域泛化能力强,但较粗粒度,作为辅助
- **路径3 (0.4)**: 细粒度匹配,质量导向,与路径1同等重要

---

#### 按 Pattern 聚合得分

**融合逻辑**:
```python
# 收集三路召回的所有 Pattern
all_patterns = set(path1_scores.keys()) | set(path2_scores.keys()) | set(path3_scores.keys())

# 计算每个 Pattern 的最终得分
final_scores = {}
for pattern_id in all_patterns:
    score1 = path1_scores.get(pattern_id, 0.0) * PATH1_WEIGHT
    score2 = path2_scores.get(pattern_id, 0.0) * PATH2_WEIGHT
    score3 = path3_scores.get(pattern_id, 0.0) * PATH3_WEIGHT

    final_scores[pattern_id] = score1 + score2 + score3

# 排序并返回 Top-10
ranked = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
top_10 = ranked[:10]
```

**关键点**:
1. **各路独立算分**: 每条路径独立计算 Pattern 得分,互不影响
2. **加权线性融合**: 按预定义权重简单相加
3. **Top-K 精排**: 最终返回得分最高的 10 个 Pattern

---

#### 结果示例

**召回结果日志**:
```
================================================================================
📊 召回结果 Top-10
================================================================================

【Rank 1】 pattern_111
  名称: Reframing Zero-Shot Generalization
  最终得分: 0.6571
  - 路径1 (相似Idea):   0.5257 (占比 80.0%)
  - 路径2 (领域相关):   0.0000 (占比 0.0%)
  - 路径3 (相似Paper):  0.1314 (占比 20.0%)
  聚类大小: 22 篇论文
  归纳总结: This cluster explores innovative methods to enhance zero-shot generalization...

【Rank 2】 pattern_110
  名称: Reframing Few Shot Learning Robustness
  最终得分: 0.4990
  - 路径1 (相似Idea):   0.3036 (占比 60.8%)
  - 路径2 (领域相关):   0.0000 (占比 0.0%)
  - 路径3 (相似Paper):  0.1954 (占比 39.2%)
  聚类大小: 24 篇论文
  归纳总结: This cluster introduces innovative frameworks to enhance few-shot learning...

...
```

**分数解读**:
- `pattern_111` 主要由**路径1** (相似 Idea) 贡献 (80%)
- `pattern_110` 在**路径1** 和**路径3** 均有贡献

---

## 当前局限与改进方向

### 1. Review 数据缺失

#### 现状
- ⚠️ **Paper 节点暂无 Review 评分数据**
- 所有 Paper 质量分默认为 **0.5**
- 导致**路径3** 和 **Paper→Pattern 边权重**失去质量区分能力

#### 影响
- **Paper→Pattern 边**: `quality` 全部为 0.5,无法反映真实论文质量
- **路径3 召回**: 只能基于相似度,无法优先推荐高质量 Paper 的 Pattern

#### 改进方案
```python
# 当补充 Review 数据后,质量评分将自动生效
def _get_paper_quality(paper):
    reviews = paper.get('reviews', [])
    if reviews:
        scores = [r['overall_score'] for r in reviews]
        avg_score = np.mean(scores)
        return (avg_score - 1) / 9  # 归一化到 [0, 1]
    return 0.5  # 默认值
```

**下一步**:
- 补充 ICLR 2025 的 Review 数据
- 重新运行 `build_edges.py` 更新边权重
- 召回质量将自动提升,无需修改代码

---

### 2. Domain 粒度过粗

#### 现状
- 98 个 Domain,粒度较大 (如 "Natural Language Processing" 包含 1000+ 篇论文)
- 200+ 个 sub_domains,分布不均匀
- **路径2** 召回时,Domain 匹配不够精确

#### 影响
- 关键词匹配容易失败 (Domain name 通常只有 2-3 个词)
- Domain 内 Pattern 过多,区分度不足

#### 改进方案

**方案1: Domain 分层聚合**
```python
# 构建 Domain 层级结构
hierarchy = {
    'Natural Language Processing': {
        'Text Classification': [...],
        'Machine Translation': [...],
        'Question Answering': [...]
    },
    'Computer Vision': {
        '3D Reconstruction': [...],
        'Object Detection': [...]
    }
}

# 召回时先匹配大领域,再匹配子领域
main_domain = match_main_domain(user_idea)
sub_domain = match_sub_domain(user_idea, main_domain)
```

**方案2: 使用 sub_domains 进行精细匹配**
```python
# 扩展关键词匹配到 sub_domains
for domain in domains:
    all_tokens = set(domain['name'].lower().split())
    all_tokens.update([s.lower() for s in domain['sub_domains']])

    match_score = len(user_tokens & all_tokens) / max(len(user_tokens), 1)
```

**方案3: 基于 Embedding 的 Domain 检索**
```python
# 使用 Embedding 计算用户 Idea 与 Domain 的语义相似度
domain_embeddings = precompute_domain_embeddings()  # 预计算
user_embedding = get_embedding(user_idea)

similarities = []
for domain_id, domain_emb in domain_embeddings.items():
    sim = cosine_similarity(user_embedding, domain_emb)
    similarities.append((domain_id, sim))

top_domains = sorted(similarities, reverse=True)[:5]
```

---

### 3. Pattern 命名与总结

#### 现状
- 124 个 Pattern 中,**912 个已通过 LLM 增强**
- 增强内容包括:
  - `representative_ideas`: 代表性想法归纳
  - `common_tricks`: 常见技巧列表
  - `naming_suggestion`: Pattern 命名建议

#### 问题
- 部分 Pattern 命名可能不够直观
- 总结内容可能需要根据实际使用情况迭代优化

#### 改进方案
- 根据召回效果反馈,调整 Pattern 命名
- 使用更强的 LLM 模型 (如 GPT-4) 重新生成总结
- 考虑加入用户反馈机制,持续优化 Pattern 描述

---

### 4. 召回效率优化

#### 现状
- **两阶段召回**: 从 ~7 分钟优化到 ~27 秒 (提速 13 倍)
- 仍有优化空间

#### 进一步优化方案

**方案1: Embedding 缓存**
```python
# 预计算所有 Idea 和 Paper 的 Embedding
idea_embeddings = precompute_all_embeddings(ideas)
paper_embeddings = precompute_all_embeddings(papers)

# 召回时直接使用缓存
user_embedding = get_embedding(user_idea)
similarities = [cosine_similarity(user_embedding, idea_emb)
                for idea_emb in idea_embeddings]
```

**方案2: 向量数据库**
- 使用 Faiss/Milvus 等向量数据库
- 支持高效的 ANN (近似最近邻) 检索
- 召回速度可进一步提升到 **~1-3 秒**

**方案3: GPU 加速**
- 使用 GPU 批量计算 Embedding 相似度
- 适合大规模实时召回场景

---

### 5. 多模态支持

#### 当前状态
- 仅支持文本 Idea 和 Paper 的匹配
- 未利用论文的其他信息 (如图表、公式、代码等)

#### 未来扩展
- 支持多模态 Embedding (文本 + 图像 + 代码)
- 引入论文的图表、算法伪代码等作为辅助特征
- 提升召回的准确性和多样性

---

### 6. 动态更新机制

#### 当前状态
- 知识图谱为静态数据,需手动重新构建
- 无法实时吸收新论文

#### 改进方案
- **增量更新**: 支持新论文动态加入图谱
- **在线学习**: 根据用户反馈调整 Pattern 权重
- **版本管理**: 支持图谱的版本回滚和 A/B 测试

---

## 总结

### 系统亮点
1. ✅ **完整的知识图谱**: 16,790 节点,444,872 条边,全面覆盖 ICLR 2025 数据
2. ✅ **三路召回策略**: 兼顾相似度、领域和质量,召回全面且准确
3. ✅ **两阶段优化**: 提速 13 倍,实现秒级召回
4. ✅ **LLM 增强**: 912 个 Pattern 经过 LLM 归纳总结,可读性强
5. ✅ **可扩展架构**: 模块化设计,易于增加新数据源和召回路径

### 待改进点
1. ⚠️ **补充 Review 数据**: 提升质量评分的准确性
2. ⚠️ **优化 Domain 匹配**: 引入层级结构或 Embedding 匹配
3. ⚠️ **向量数据库**: 进一步提升召回效率
4. ⚠️ **动态更新**: 支持增量更新和在线学习

---

## 附录

### 核心文件说明

| 文件 | 说明 |
|------|------|
| `scripts/build_entity_v3.py` | 节点构建脚本 |
| `scripts/build_edges.py` | 边构建脚本 |
| `scripts/recall_system.py` | 召回系统实现 (类封装版本) |
| `scripts/simple_recall_demo.py` | 召回系统 Demo (单测试版本) |
| `output/nodes_*.json` | 节点数据文件 |
| `output/edges.json` | 边数据文件 |
| `output/knowledge_graph_v2.gpickle` | NetworkX 图谱文件 |

### 参数配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `PATH1_TOP_K_IDEAS` | 10 | 路径1召回的相似 Idea 数量 |
| `PATH1_FINAL_TOP_K` | 10 | 路径1最终保留的 Pattern 数量 |
| `PATH2_TOP_K_DOMAINS` | 5 | 路径2召回的相关 Domain 数量 |
| `PATH2_FINAL_TOP_K` | 5 | 路径2最终保留的 Pattern 数量 |
| `PATH3_TOP_K_PAPERS` | 20 | 路径3召回的相似 Paper 数量 |
| `PATH3_FINAL_TOP_K` | 10 | 路径3最终保留的 Pattern 数量 |
| `FINAL_TOP_K` | 10 | 最终返回的 Pattern 数量 |
| `COARSE_RECALL_SIZE` | 100 | 粗排候选数量 |
| `TWO_STAGE_RECALL` | True | 是否启用两阶段召回 |
| `USE_EMBEDDING` | True | 是否使用 Embedding (推荐) |

---

**文档版本**: V3.0
**更新日期**: 2026-01-22
**作者**: Idea2Pattern Team

