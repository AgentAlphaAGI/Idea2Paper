# 知识图谱边类型说明

本文档详细说明了 Idea2Pattern 知识图谱中所有边的类型、用途和权重定义。

---

## 📋 目录

1. [基础连接边](#基础连接边)
2. [三路召回策略](#三路召回策略)
3. [权重计算公式总结](#权重计算公式总结)

---

## 基础连接边

这些边用于建立实体之间的基本关系，为召回路径提供基础结构。

### 1. Paper -[implements]-> Idea

**用途**: 表示某篇 Paper 实现了某个核心 Idea。

**权重**: 无权重（布尔关系）

**构建逻辑**:
- 通过 Paper 的 `source_paper_ids` 字段与 Idea 节点匹配
- 每个 Paper 只链接到一个 Idea

**示例**:
```json
{
  "source": "ACL_2017_104",
  "target": "idea_0",
  "relation": "implements"
}
```

---

### 2. Paper -[uses_pattern]-> Pattern

**用途**: 表示某篇 Paper 使用了某个写作 Pattern。

**权重**:
- `quality`: Paper 的综合质量分数 (0-1)

**构建逻辑**:
- 从 Paper 的 `pattern_ids` 字段获取关联的 Pattern
- 质量分数基于 Review 评分归一化

**质量分数计算**:
```python
quality = (avg_review_score - 1) / 9  # 归一化到 [0, 1]
```

**示例**:
```json
{
  "source": "ACL_2017_104",
  "target": "pattern_5",
  "relation": "uses_pattern",
  "quality": 0.78
}
```

---

### 3. Paper -[in_domain]-> Domain

**用途**: 表示某篇 Paper 属于某个研究领域。

**权重**: 无权重（布尔关系）

**构建逻辑**:
- 从 Paper 的 `domains` 字段获取所属领域
- 一篇 Paper 可以属于多个 Domain

**示例**:
```json
{
  "source": "ACL_2017_104",
  "target": "domain_12",
  "relation": "in_domain"
}
```

---

## 三路召回策略

### 路径1: Idea → Idea → Pattern (相似Idea召回)

**召回流程**:
```
用户输入新Idea → 实时计算与图谱中所有Idea的相似度 → Top-K相似Idea → 这些Idea的pattern_ids
```

**不需要预构建边**:
- ❌ 不需要 `Idea → Idea` 边
- ✅ Idea 节点已有 `pattern_ids` 字段

**实时计算相似度**:
```python
def find_similar_ideas(user_idea_text, top_k=10):
    similarities = []
    for idea in graph_ideas:
        sim = compute_similarity(user_idea_text, idea['description'])
        similarities.append((idea['idea_id'], sim))

    # 返回Top-K相似Idea
    top_ideas = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]

    # 收集这些Idea的Pattern
    patterns = set()
    for idea_id, sim in top_ideas:
        patterns.update(graph_ideas[idea_id]['pattern_ids'])

    return patterns
```

**权重定义**:
- `similarity`: 实时计算的语义相似度 (0-1)
- `pattern_relevance`: Idea使用该Pattern的Paper的平均质量

---

### 路径2: Idea → Domain → Pattern (领域相关性召回)

**召回流程**:
```
用户输入新Idea → 找到相关Domain → 该Domain中表现好的Pattern
```

#### 2.1 Idea -[belongs_to]-> Domain

**用途**: 表示某个 Idea 主要属于哪些研究领域。

**权重**:
- `weight`: Idea 相关 Paper 在该 Domain 中的占比 (0-1)
- `paper_count`: 该 Domain 中的 Paper 数量
- `total_papers`: Idea 的所有 Paper 总数

**构建逻辑**:
1. 统计 Idea 的所有 `source_paper_ids`
2. 统计这些 Paper 在各 Domain 中的分布
3. 计算每个 Domain 的占比作为权重

**权重计算**:
```python
weight = paper_count_in_domain / total_papers
```

**示例**:
```json
{
  "source": "idea_42",
  "target": "domain_12",
  "relation": "belongs_to",
  "weight": 0.75,
  "paper_count": 3,
  "total_papers": 4
}
```

#### 2.2 Pattern -[works_well_in]-> Domain

**用途**: 表示某个 Pattern 在某个领域中的使用效果。

**权重**:
- `frequency`: Pattern 在该 Domain 中的使用次数
- `effectiveness`: Pattern 在该 Domain 中的效果增益（相对基线）
- `confidence`: 置信度 (0-1)，基于样本数
- `avg_quality`: Pattern 在该 Domain 中 Paper 的平均质量
- `baseline`: 该 Domain 的质量基线

**构建逻辑**:
1. 统计使用该 Pattern 且属于该 Domain 的所有 Paper
2. 计算这些 Paper 的平均质量
3. 计算该 Domain 所有 Paper 的平均质量作为基线
4. 效果增益 = 平均质量 - 基线

**权重计算**:
```python
effectiveness = avg_quality - baseline
confidence = min(frequency / 20, 1.0)
```

**示例**:
```json
{
  "source": "pattern_5",
  "target": "domain_12",
  "relation": "works_well_in",
  "frequency": 15,
  "effectiveness": 0.12,
  "confidence": 0.75,
  "avg_quality": 0.82,
  "baseline": 0.70
}
```

**召回使用**:
```python
# 1. 找到用户Idea最相关的Domain
user_idea_domains = find_related_domains(user_idea)

# 2. 在这些Domain中找效果最好的Pattern
patterns = []
for domain in user_idea_domains:
    domain_patterns = G.predecessors(domain, relation='works_well_in')
    ranked = sorted(domain_patterns,
        key=lambda p: G[p][domain]['effectiveness'] * G[p][domain]['confidence'],
        reverse=True)
    patterns.extend(ranked[:10])
```

---

### 路径3: Idea → Paper → Pattern (相似Paper召回)

**召回流程**:
```
用户输入新Idea → 找到实现相似Idea的高质量Paper → 这些Paper使用的Pattern
```

#### 3.1 Idea -[similar_to_paper]-> Paper

**用途**: 表示某个 Idea 与某篇 Paper 的核心思想相似。

**权重**:
- `similarity`: 语义相似度 (0-1)
- `quality`: Paper 的综合质量分数 (0-1)
- `combined_weight`: 综合权重 = similarity × quality

**构建逻辑**:
1. 计算 Idea 描述与所有 Paper 的 core_idea 的相似度
2. 过滤低相似度的 Paper (阈值 0.1)
3. 计算综合权重
4. 每个 Idea 只保留 Top-50 相似 Paper

**相似度计算**:
```python
# 使用 Jaccard 相似度（词袋模型）
similarity = |tokens1 ∩ tokens2| / |tokens1 ∪ tokens2|
combined_weight = similarity * quality
```

**示例**:
```json
{
  "source": "idea_42",
  "target": "ACL_2017_150",
  "relation": "similar_to_paper",
  "similarity": 0.65,
  "quality": 0.82,
  "combined_weight": 0.533
}
```

**召回使用**:
```python
# 1. 找到与用户Idea最相似的Paper
similar_papers = find_similar_papers(user_idea, top_k=20)

# 2. 收集这些Paper使用的Pattern
patterns = set()
for paper_id, combined_weight in similar_papers:
    paper_patterns = G.successors(paper_id, relation='uses_pattern')
    for pattern_id in paper_patterns:
        # 考虑Paper质量作为Pattern的权重
        pattern_weight = combined_weight * G[paper_id][pattern_id]['quality']
        patterns.add((pattern_id, pattern_weight))

# 3. 按权重排序
ranked_patterns = sorted(patterns, key=lambda x: x[1], reverse=True)
```

---

## 权重计算公式总结

| 边类型 | 关键权重 | 计算公式 | 取值范围 |
|--------|---------|---------|---------|
| `Paper → Pattern` | `quality` | `(avg_review - 1) / 9` | [0, 1] |
| `Idea → Domain` | `weight` | `paper_count / total_papers` | [0, 1] |
| `Pattern → Domain` | `effectiveness` | `avg_quality - baseline` | [-1, 1] |
| `Pattern → Domain` | `confidence` | `min(frequency / 20, 1.0)` | [0, 1] |
| `Idea → Paper` | `similarity` | `Jaccard(tokens1, tokens2)` | [0, 1] |
| `Idea → Paper` | `combined_weight` | `similarity × quality` | [0, 1] |

---

## 完整召回示意图

```
用户输入: 新 Idea
    |
    |-- 路径1: Idea → Idea → Pattern (实时计算)
    |      |
    |      |-- 计算相似度 → Top-K相似Idea
    |      |-- 获取 Idea.pattern_ids → Pattern
    |      |
    |      └── 得分: similarity × pattern使用频率
    |
    |-- 路径2: Idea → Domain → Pattern
    |      |
    |      |-- [belongs_to] → Domain (weight)
    |      |-- [works_well_in] → Pattern (effectiveness, confidence)
    |      |
    |      └── 得分: weight × effectiveness × confidence
    |
    └-- 路径3: Idea → Paper → Pattern
           |
           |-- [similar_to_paper] → Paper (similarity, quality)
           |-- [uses_pattern] → Pattern (quality)
           |
           └── 得分: similarity × quality_paper × quality_pattern
```

---

## 使用示例

### 完整召回流程

```python
def recall_patterns(user_idea_text):
    """三路召回Pattern"""

    all_patterns = {}

    # 路径1: 相似Idea召回
    similar_ideas = find_similar_ideas(user_idea_text, top_k=10)
    for idea_id, similarity in similar_ideas:
        for pattern_id in graph.nodes[idea_id]['pattern_ids']:
            score = similarity * 0.4  # 路径1权重
            all_patterns[pattern_id] = all_patterns.get(pattern_id, 0) + score

    # 路径2: 领域相关召回
    related_domains = find_related_domains(user_idea_text, top_k=5)
    for domain_id, domain_weight in related_domains:
        patterns = G.predecessors(domain_id, relation='works_well_in')
        for pattern_id in patterns:
            edge = G[pattern_id][domain_id]
            score = domain_weight * edge['effectiveness'] * edge['confidence'] * 0.3
            all_patterns[pattern_id] = all_patterns.get(pattern_id, 0) + score

    # 路径3: 相似Paper召回
    similar_papers = find_similar_papers(user_idea_text, top_k=20)
    for paper_id, combined_weight in similar_papers:
        patterns = G.successors(paper_id, relation='uses_pattern')
        for pattern_id in patterns:
            pattern_quality = G[paper_id][pattern_id]['quality']
            score = combined_weight * pattern_quality * 0.3
            all_patterns[pattern_id] = all_patterns.get(pattern_id, 0) + score

    # 排序返回Top-K
    ranked = sorted(all_patterns.items(), key=lambda x: x[1], reverse=True)
    return ranked[:10]
```

---

## 注意事项

1. **路径1不需要预构建边**: Idea → Idea 的相似度是实时计算的，因为用户输入的是新Idea
2. **相似度计算**: 当前使用简单的 Jaccard 相似度，后续可升级为语义嵌入模型（如 BERT）
3. **Top-K 限制**: `Idea → Paper` 边只保留 Top-50，避免图过于稠密
4. **质量归一化**: Review 评分假设范围为 1-10，需根据实际数据调整
5. **置信度阈值**: Pattern 在 Domain 中至少 20 个样本才能达到满置信度

---

## 文件生成

- **脚本**: `scripts/build_edges.py`
- **输出**: `output/edges.json`, `output/knowledge_graph_v2.gpickle`
- **运行**: `python scripts/build_edges.py`

