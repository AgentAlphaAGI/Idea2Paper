# 三路召回系统文档

## 📋 概述

本文档详细说明了基于知识图谱的三路召回系统,包括召回策略、相似度计算、多路融合、参数配置和运行方式。

---

## 1. 系统架构

### 1.1 核心目标

**输入**: 用户的研究Idea描述(文本)
**输出**: Top-10最相关的研究Pattern(写作套路/方法模板)

### 1.2 技术架构

```
┌──────────────────────────────────────────────────────────────────┐
│                    【三路召回系统架构】                            │
└──────────────────────────────────────────────────────────────────┘

用户输入Idea (文本描述)
    │
    ├────────────────────────────────────────────────────────────┐
    │                  三路并行召回 (约27秒)                      │
    ├────────────────────────────────────────────────────────────┤
    │                                                              │
    │  ┌──────────────┬──────────────┬──────────────┐           │
    │  │   路径1      │    路径2     │    路径3     │           │
    │  │ 相似Idea召回 │ 领域相关召回 │ 相似Paper召回│           │
    │  │  (权重0.4)   │  (权重0.2)   │  (权重0.4)   │           │
    │  └──────────────┴──────────────┴──────────────┘           │
    │        │              │              │                      │
    │        │              │              │                      │
    │  ┌─────▼──────┐  ┌───▼────┐  ┌──────▼─────┐              │
    │  │【粗排阶段】│  │【Domain】│  │【粗排阶段】│              │
    │  │ Jaccard   │  │ 匹配    │  │ Jaccard   │              │
    │  └───────────┘  └────────┘  └────────────┘              │
    │        │              │              │                      │
    │  遍历8,284个    使用Top-1      遍历8,285个                │
    │  Idea描述       Idea的Domain    Paper标题                 │
    │  词袋模型       关键词匹配      词袋模型                   │
    │  快速过滤       查图谱边        快速过滤                   │
    │        │              │              │                      │
    │  Top-100个      Top-5个        Top-100个                  │
    │  候选Idea       Domain         候选Paper                  │
    │        │              │              │                      │
    │  ┌─────▼──────┐  ┌───▼────┐  ┌──────▼─────┐              │
    │  │【精排阶段】│  │【Pattern】│  │【精排阶段】│              │
    │  │ Embedding │  │ 召回    │  │ Embedding │              │
    │  └───────────┘  └────────┘  └────────────┘              │
    │        │              │              │                      │
    │  100次API调用   查works_well  100次API调用                │
    │  语义相似度     _in边        语义相似度                    │
    │  精确重排       效果加权      × Paper质量                  │
    │        │              │              │                      │
    │  Top-10个       Top-K个       Top-20个                    │
    │  相似Idea       Pattern       相似Paper                   │
    │        │              │              │                      │
    │  ┌─────▼──────┐  ┌───▼────┐  ┌──────▼─────┐              │
    │  │【Pattern】 │  │【Pattern】│  │【Pattern】 │              │
    │  │  提取     │  │  得分   │  │  提取     │              │
    │  └───────────┘  └────────┘  └────────────┘              │
    │        │              │              │                      │
    │  直接获取Idea   Domain相关度   查Paper→Pattern             │
    │  .pattern_ids   × effectiveness  uses_pattern边            │
    │  按相似度加权   × confidence   相似度×质量加权              │
    │        │              │              │                      │
    │  Pattern得分    Pattern得分    Pattern得分                 │
    │  字典           字典           字典                         │
    │        │              │              │                      │
    └────────┼──────────────┼──────────────┼────────────────────┘
             │              │              │
             └──────────────┴──────────────┘
                          │
                          ▼
               ┌──────────────────────┐
               │   【多路融合】        │
               └──────────────────────┘
                          │
                score = path1 × 0.4
                      + path2 × 0.2
                      + path3 × 0.4
                          │
                          ▼
                  按融合得分排序
                          │
                          ▼
               ┌──────────────────────┐
               │   Top-10 Pattern     │
               │   返回给用户         │
               └──────────────────────┘
```

**架构说明**:
- **横向**: 三路并行执行,互不干扰
- **纵向**: 每路内部两阶段优化(粗排→精排)
- **融合**: 加权求和,确保多样性

### 1.3 数据规模

```
知识图谱统计:
  - Idea节点:    8,284 个
  - Pattern节点: 124 个
  - Domain节点:  98 个
  - Paper节点:   8,285 个
  - 总边数:      444,872 条
```

---

## 2. 三路召回策略

### 2.1 设计理念

三路召回从不同维度捕捉用户需求,避免重复和信息冗余:

| 路径 | 匹配对象 | 捕捉维度 | 权重 | 典型场景 |
|------|---------|---------|------|---------|
| **路径1** | Idea Description | 核心思想/概念相似性 | 0.4 | 用户描述与历史成功案例的核心思路一致 |
| **路径2** | Domain & Sub-domains | 领域泛化能力 | 0.2 | 用户Idea属于某领域,该领域有验证有效的Pattern |
| **路径3** | Paper Title | 研究主题/具体问题相似性 | 0.4 | 用户想解决的具体问题与某些论文标题表述类似 |

**互补性说明**:
- **路径1 vs 路径3**: 路径1关注"想法本质",路径3关注"研究方向"
- **路径2的泛化作用**: 即使用户Idea是全新的,只要属于某个成熟领域,也能召回该领域通用的有效Pattern

---

## 3. 路径1: 相似Idea召回

### 3.1 召回流程

```
用户Idea (文本)
    ↓ [粗排] Jaccard快速筛选
候选Idea (Top-100)
    ↓ [精排] Embedding重排
相似Idea (Top-10)
    ↓ 直接获取 idea.pattern_ids
Pattern集合
    ↓ 按相似度加权累加
Top-10 Pattern (得分字典)
```

### 3.2 两阶段召回优化

**为什么需要两阶段?**
- 全量Embedding检索: 8,284次API调用,耗时**~7分钟** ❌
- 两阶段召回: 100次API调用,耗时**~10秒** ✅ (提速40倍)

**粗排阶段(Jaccard)**:
```python
def compute_jaccard_similarity(text1, text2):
    """计算Jaccard相似度(词袋模型)"""
    # 分词
    tokens1 = set(text1.lower().split())
    tokens2 = set(text2.lower().split())

    # Jaccard = 交集/并集
    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)

    return intersection / union if union > 0 else 0.0

# 粗排: 快速筛选Top-100
coarse_similarities = []
for idea in ideas:  # 8,284个
    sim = compute_jaccard_similarity(user_idea, idea['description'])
    if sim > 0:
        coarse_similarities.append((idea_id, sim))

coarse_similarities.sort(reverse=True)
candidates = coarse_similarities[:100]  # 粗排Top-100
```

**精排阶段(Embedding)**:
```python
def compute_embedding_similarity(text1, text2):
    """使用Qwen3-Embedding-4B计算语义相似度"""
    # 获取Embedding
    emb1 = get_embedding(text1)  # API调用
    emb2 = get_embedding(text2)  # API调用

    # 余弦相似度
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

# 精排: 对候选使用Embedding重排
fine_similarities = []
for idea_id, _ in candidates:  # 100个
    idea = idea_id_to_idea[idea_id]
    sim = compute_embedding_similarity(user_idea, idea['description'])
    if sim > 0:
        fine_similarities.append((idea_id, sim))

fine_similarities.sort(reverse=True)
top_ideas = fine_similarities[:10]  # 精排Top-10
```

### 3.3 Pattern得分计算

```python
pattern_scores = defaultdict(float)

for idea_id, similarity in top_10_ideas:
    idea = idea_id_to_idea[idea_id]

    # V3版本: 直接从Idea节点获取pattern_ids
    for pattern_id in idea['pattern_ids']:
        # 得分 = 相似度 (多个Idea使用同一Pattern时会累加)
        pattern_scores[pattern_id] += similarity

# 排序并只保留Top-10
sorted_patterns = sorted(pattern_scores.items(), reverse=True)
top_patterns = dict(sorted_patterns[:10])
```

**示例**:
```
用户Idea: "使用Transformer进行文本分类"

相似Idea_1 (相似度0.8) → [pattern_5, pattern_10]
相似Idea_2 (相似度0.7) → [pattern_5, pattern_20]
相似Idea_3 (相似度0.6) → [pattern_10]

路径1得分:
  pattern_5:  0.8 + 0.7 = 1.5
  pattern_10: 0.8 + 0.6 = 1.4
  pattern_20: 0.7 = 0.7
```

---

## 4. 路径2: 领域相关召回

### 4.1 召回流程

```
用户Idea (文本)
    ↓ 关键词匹配Domain name
相关Domain (Top-5)
    ↓ 反向查找Pattern→Domain边
在Domain中表现好的Pattern
    ↓ 按effectiveness & confidence加权
Top-5 Pattern (得分字典)
```

### 4.2 Domain匹配逻辑

**方法1: 关键词匹配**(优先):
```python
def match_domains(user_idea, domains):
    domain_scores = []
    user_tokens = set(user_idea.lower().split())

    for domain in domains:
        domain_name = domain['name']
        domain_tokens = set(domain_name.lower().split())

        # 词汇重叠
        match_score = len(user_tokens & domain_tokens) / max(len(user_tokens), 1)

        if match_score > 0:
            domain_scores.append((domain['domain_id'], match_score))

    domain_scores.sort(reverse=True)
    return domain_scores[:5]  # Top-5
```

**方法2: 通过相似Idea的Domain**(备选):
```python
if not domain_scores:
    # 找到最相似的Idea
    similarities = [(idea, compute_similarity(user_idea, idea['description']))
                    for idea in ideas]
    top_idea = max(similarities, key=lambda x: x[1])[0]

    # 获取该Idea的Domain (通过belongs_to边)
    for successor in G.successors(top_idea['idea_id']):
        edge_data = G[top_idea['idea_id']][successor]
        if edge_data['relation'] == 'belongs_to':
            domain_id = successor
            weight = edge_data['weight']
            domain_scores.append((domain_id, weight))
```

### 4.3 Pattern得分计算

```python
pattern_scores = defaultdict(float)

for domain_id, domain_weight in top_5_domains:
    # 反向查找: 哪些Pattern在该Domain中表现好?
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

# 排序并只保留Top-5 (辅助通道)
sorted_patterns = sorted(pattern_scores.items(), reverse=True)
top_patterns = dict(sorted_patterns[:5])
```

**边权重说明**:
- `effectiveness`: Pattern在该Domain的效果增益(相对基线) [-1, 1]
  - 正值: Pattern在该Domain效果好于平均水平
  - 负值: Pattern在该Domain效果低于平均水平
- `confidence`: 基于样本数的置信度 [0, 1]
  - 样本数≥20时,置信度达到1.0

---

## 5. 路径3: 相似Paper召回

### 5.1 召回流程

```
用户Idea (文本)
    ↓ [粗排] Jaccard筛选(基于Paper Title)
候选Paper (Top-100)
    ↓ [精排] Embedding重排(基于Paper Title)
相似Paper (Top-20)
    ↓ 查找Paper→Pattern边
Pattern集合
    ↓ 按similarity × quality加权
Top-10 Pattern (得分字典)
```

### 5.2 设计理念

**路径1 vs 路径3的互补性**:
- **路径1**: 使用Idea Description计算相似度 → 捕捉**核心思想/概念**的相似性
- **路径3**: 使用Paper Title计算相似度 → 捕捉**研究主题/具体问题**的相似性

### 5.3 两阶段召回优化

**粗排阶段(Jaccard)**:
```python
coarse_similarities = []
for paper in papers:  # 8,285个
    paper_title = paper['title']  # 使用论文标题
    sim = compute_jaccard_similarity(user_idea, paper_title)

    if sim > 0.05:  # 降低阈值保留更多候选
        coarse_similarities.append((paper_id, sim))

coarse_similarities.sort(reverse=True)
candidates = coarse_similarities[:100]  # 粗排Top-100
```

**精排阶段(Embedding)**:
```python
fine_similarities = []
for paper_id, _ in candidates:  # 100个
    paper = paper_id_to_paper[paper_id]
    paper_title = paper['title']  # 使用论文标题

    sim = compute_embedding_similarity(user_idea, paper_title)

    if sim > 0.1:  # 过滤低相似度
        # 获取Paper质量 (优先使用 review_stats.avg_score)
        quality = _get_paper_quality(paper)  # [0, 1]
        combined_weight = sim * quality  # 结合相似度和质量
        fine_similarities.append((paper_id, sim, quality, combined_weight))

fine_similarities.sort(key=lambda x: x[3], reverse=True)
top_papers = fine_similarities[:20]  # 精排Top-20
```

### 5.4 Pattern得分计算

```python
pattern_scores = defaultdict(float)

for paper_id, similarity, paper_quality, combined_weight in top_20_papers:
    # 从图谱中查找Paper使用的Pattern
    for successor in G.successors(paper_id):
        edge_data = G[paper_id][successor]

        if edge_data['relation'] == 'uses_pattern':
            pattern_id = successor
            pattern_quality = edge_data['quality']  # Paper的Review质量

            # 得分 = (相似度 × Paper质量) × Pattern质量
            # paper_quality 来自 review_stats.avg_score
            score = combined_weight * pattern_quality
            pattern_scores[pattern_id] += score

# 排序并只保留Top-10
sorted_patterns = sorted(pattern_scores.items(), reverse=True)
top_patterns = dict(sorted_patterns[:10])
```

---

## 6. 多路融合与精排

### 6.1 融合策略

```python
# 路径权重配置
PATH1_WEIGHT = 0.4  # 相似Idea召回 (重要)
PATH2_WEIGHT = 0.2  # 领域相关召回 (辅助)
PATH3_WEIGHT = 0.4  # 相似Paper召回 (重要)
```

**权重设计理由**:
- **路径1 (0.4)**: 直接利用历史成功经验,最可靠
- **路径2 (0.2)**: 领域泛化能力强,但较粗粒度,作为辅助
- **路径3 (0.4)**: 细粒度匹配,质量导向,与路径1同等重要

### 6.2 按Pattern聚合得分

```python
# 收集三路召回的所有Pattern
all_patterns = set(path1_scores.keys()) | set(path2_scores.keys()) | set(path3_scores.keys())

# 计算每个Pattern的最终得分
final_scores = {}
for pattern_id in all_patterns:
    score1 = path1_scores.get(pattern_id, 0.0) * PATH1_WEIGHT
    score2 = path2_scores.get(pattern_id, 0.0) * PATH2_WEIGHT
    score3 = path3_scores.get(pattern_id, 0.0) * PATH3_WEIGHT

    final_scores[pattern_id] = score1 + score2 + score3

# 排序并返回Top-10
ranked = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
top_10 = ranked[:10]
```

### 6.3 结果示例

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

【Rank 2】 pattern_110
  名称: Reframing Few Shot Learning Robustness
  最终得分: 0.4990
  - 路径1 (相似Idea):   0.3036 (占比 60.8%)
  - 路径2 (领域相关):   0.0000 (占比 0.0%)
  - 路径3 (相似Paper):  0.1954 (占比 39.2%)
  聚类大小: 24 篇论文
```

---

## 7. 参数配置

### 7.1 召回参数

```python
class RecallConfig:
    """召回系统配置"""
    # 路径1: 相似Idea召回
    PATH1_TOP_K_IDEAS = 10         # 召回前K个最相似的Idea
    PATH1_FINAL_TOP_K = 10         # 最终只保留Top-K个Pattern

    # 路径2: 领域相关召回
    PATH2_TOP_K_DOMAINS = 5        # 召回前K个最相关的Domain
    PATH2_FINAL_TOP_K = 5          # 最终只保留Top-K个Pattern

    # 路径3: 相似Paper召回
    PATH3_TOP_K_PAPERS = 20        # 召回前K个最相似的Paper
    PATH3_FINAL_TOP_K = 10         # 最终只保留Top-K个Pattern

    # 各路召回的权重
    PATH1_WEIGHT = 0.4             # 路径1权重(相似Idea - 重要)
    PATH2_WEIGHT = 0.2             # 路径2权重(领域相关 - 辅助)
    PATH3_WEIGHT = 0.4             # 路径3权重(相似Paper - 重要)

    # 最终召回的Top-K
    FINAL_TOP_K = 10

    # 相似度计算方式
    USE_EMBEDDING = True           # 使用embedding(推荐)

    # 两阶段召回优化
    TWO_STAGE_RECALL = True        # 启用两阶段召回(大幅提速)
    COARSE_RECALL_SIZE = 100       # 粗召回数量(Jaccard)
    FINE_RECALL_SIZE = 20          # 精排数量(Embedding)
```

### 7.2 Embedding API配置

```python
# API端点
EMBEDDING_API_URL = "https://api.siliconflow.cn/v1/embeddings"

# 模型选择
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-4B"

# API密钥
EMBEDDING_API_KEY = os.getenv("SILICONFLOW_API_KEY")
```

---

## 8. 运行方式

### 8.1 独立运行召回系统

**命令**:
```bash
cd /Users/gaoge/code/mycode/Idea2Paper/Paper-KG-Pipeline
python scripts/simple_recall_demo.py "你的研究Idea描述"
```

**示例**:
```bash
python scripts/simple_recall_demo.py "使用蒸馏技术完成Transformer跨领域文本分类任务"
```

**输出**:
```
🎯 三路召回系统 Demo
================================================================================
【用户Idea】
使用蒸馏技术完成Transformer跨领域文本分类任务

🔍 [路径1] 相似Idea召回...
  [粗排] 使用Jaccard快速筛选Top-100...
  [精排] 使用Embedding重排Top-10...
  ✓ 粗排8284个 → 精排100个 → 最终10个

🌍 [路径2] 领域相关性召回...
  找到 3 个相关Domain
  ✓ 召回 34 个Pattern，保留Top-5

📄 [路径3] 相似Paper召回...
  [粗排] 使用Jaccard快速筛选Top-100...
  [精排] 使用Embedding重排Top-20...
  ✓ 粗排171个 → 精排100个 → 最终20个

🔗 融合三路召回结果...

📊 召回结果 Top-10
【Rank 1】 pattern_11 - 模型压缩与知识蒸馏
  最终得分: 0.1312
  ...
```

### 8.2 作为类使用

```python
from recall_system import RecallSystem

# 初始化召回系统
system = RecallSystem()

# 执行召回
user_idea = "你的研究Idea"
results = system.recall(user_idea, verbose=True)

# 处理结果
for pattern_id, pattern_info, score in results:
    print(f"Pattern: {pattern_info['name']}, Score: {score:.4f}")
```

### 8.3 集成到Pipeline

```python
# 在idea2story_pipeline.py中使用
from recall_system import RecallSystem

recall_system = RecallSystem()
recall_results = recall_system.recall(user_idea, verbose=True)

# recall_results格式: [(pattern_id, pattern_info, score), ...]
```

---

## 9. 性能优化

### 9.1 召回速度对比

| 模式 | 描述 | 时间 | API调用次数 |
|------|------|------|-----------|
| **全量Embedding** | 对所有8,284个Idea用Embedding计算 | ~7分钟 | 8,284次 |
| **两阶段召回** | Jaccard粗排100→Embedding精排10 | ~27秒 | 100次 |
| **提速比** | - | **13倍** | - |

### 9.2 进一步优化方案

**方案1: Embedding缓存**:
```python
# 预计算所有Idea和Paper的Embedding
idea_embeddings = precompute_all_embeddings(ideas)
paper_embeddings = precompute_all_embeddings(papers)

# 召回时直接使用缓存
user_embedding = get_embedding(user_idea)
similarities = [cosine_similarity(user_embedding, idea_emb)
                for idea_emb in idea_embeddings]
```

**方案2: 向量数据库**:
```python
# 使用Faiss/Milvus等向量数据库
import faiss

# 构建索引
index = faiss.IndexFlatIP(embedding_dim)
index.add(idea_embeddings)

# ANN检索
D, I = index.search(user_embedding, k=10)  # Top-10
```
预期提速: **~1-3秒**

**方案3: GPU加速**:
```python
# 使用GPU批量计算Embedding相似度
import torch

user_emb = torch.tensor(user_embedding).cuda()
all_embs = torch.tensor(idea_embeddings).cuda()

similarities = torch.matmul(user_emb, all_embs.T)
```

---

## 10. 故障排查

### 10.1 常见问题

**Q: 召回结果全是高分Pattern**
```
原因: 路径2权重过高,导致热门Pattern得分虚高
解决: 降低PATH2_WEIGHT (0.2 → 0.1)
```

**Q: Embedding API超时**
```
原因: 网络问题或API限流
解决:
1. 增加重试机制
2. 添加请求延迟(time.sleep(0.1))
3. 使用缓存避免重复请求
```

**Q: 召回速度慢**
```
原因: TWO_STAGE_RECALL=False或USE_EMBEDDING=False
解决: 确保config中启用两阶段召回和Embedding
```

**Q: 路径1得分为0**
```
原因: 用户Idea与所有历史Idea相似度极低
检查: 打印相似度分布,确认是否有匹配的Idea
```

### 10.2 调试模式

```python
# 启用详细日志
results = system.recall(user_idea, verbose=True)

# 查看中间结果
print(f"路径1召回Pattern数: {len(path1_scores)}")
print(f"路径2召回Pattern数: {len(path2_scores)}")
print(f"路径3召回Pattern数: {len(path3_scores)}")

# 查看相似度分布
for idea_id, sim in top_ideas:
    print(f"Idea {idea_id}: {sim:.3f}")
```

---

## 11. 评估指标

### 11.1 召回质量评估

**相关性评估**:
```python
# 人工标注Top-10结果的相关性(0-1)
relevance_scores = []
for pattern in top_10:
    score = manual_annotation(pattern, user_idea)
    relevance_scores.append(score)

avg_relevance = np.mean(relevance_scores)
print(f"平均相关性: {avg_relevance:.2f}")
```

**多样性评估**:
```python
# 计算Top-10 Pattern的cluster size分布
cluster_sizes = [p['size'] for p in top_10_patterns]
diversity_score = np.std(cluster_sizes) / np.mean(cluster_sizes)
print(f"多样性得分(变异系数): {diversity_score:.2f}")
```

### 11.2 性能监控

```python
import time

start = time.time()
results = system.recall(user_idea)
elapsed = time.time() - start

print(f"召回耗时: {elapsed:.2f}秒")
print(f"API调用次数: {api_call_count}")
```

---

## 12. 扩展与定制

### 12.1 自定义权重

```python
# 在recall_system.py中修改
class RecallConfig:
    PATH1_WEIGHT = 0.5  # 提高路径1权重
    PATH2_WEIGHT = 0.1  # 降低路径2权重
    PATH3_WEIGHT = 0.4
```

### 12.2 添加新的召回路径

**示例: 路径4 - 相似技术栈召回**:
```python
def _recall_path4_similar_techniques(self, user_idea):
    """路径4: 通过技术栈相似度召回"""
    # 提取技术关键词
    techniques = extract_techniques(user_idea)

    # 匹配Pattern的common_tricks
    pattern_scores = defaultdict(float)
    for pattern in self.patterns:
        tricks = pattern.get('common_tricks', [])
        overlap = len(set(techniques) & set(tricks))
        pattern_scores[pattern['pattern_id']] = overlap

    return pattern_scores
```

### 12.3 领域特化

```python
# 针对特定领域(如NLP)调整参数
if domain == "Natural Language Processing":
    RecallConfig.PATH1_WEIGHT = 0.5  # NLP领域更依赖历史经验
    RecallConfig.PATH2_WEIGHT = 0.1
```

---

## 13. 总结

### 系统亮点

✅ **三路互补召回**: 兼顾相似度、领域和质量
✅ **两阶段优化**: 提速13倍,实现秒级召回
✅ **质量导向召回**: 路径3结合Review质量评分,提升召回准确性
✅ **LLM增强Pattern**: 124个Pattern经过LLM归纳总结
✅ **可扩展架构**: 易于添加新召回路径
✅ **完整监控**: 详细的日志和评估指标

### 技术特性

✅ **Embedding + Jaccard混合策略**: 平衡精度和速度
✅ **图谱结构化召回**: 利用边权重精确计算得分
✅ **多维度质量评分**: 综合overall_score、confidence、contribution、correctness
✅ **实时计算**: 路径3避免预构建冗余边

### 待改进

⚠️ **优化Domain匹配**: 引入层级结构或Embedding匹配
⚠️ **向量数据库**: 进一步提升召回效率到1-3秒
⚠️ **在线学习**: 根据用户反馈调整权重
⚠️ **扩展Review数据**: 整合更多会议的评审数据

---

**生成时间**: 2026-01-25
**版本**: V3.1
**作者**: Idea2Paper Team

