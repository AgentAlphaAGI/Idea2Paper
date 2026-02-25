# Path4: Agentic Search 召回 — 简化方案 (v2)

> 状态：Draft v2 | 2026-02-25

---

## 方案总述

现有 Idea2Paper 的三路召回（相似 Idea / 领域相关 / 相似 Paper）完全依赖本地 KG 的存量数据。
当用户提出的 idea 涉及 KG 构建之后的新工作时，三路召回存在"盲区"。

**Agentic Search（第四路）是一条完全独立的"联网检索 → pattern 抽取 → 排序"链路。**
它的核心思路是：

1. 根据用户 idea 自动生成检索 query，通过学术 API（Semantic Scholar）在 **CS 顶会白名单**
   范围内搜索近期论文；不在白名单中的论文直接丢弃，不参与后续任何环节。
2. 对搜到的白名单内论文，去除 KG 中已有的存量，仅保留"KG 外的新论文"。
3. 对这些新论文用 LLM 抽取 paper-level pattern（problem / gap / solution / story），
   再通过 embedding 聚类合成 cluster-level pattern——产出结构与 `nodes_pattern.json` 对齐。
4. 在 Path4 **内部**对这些 pattern 做独立排序（因为 Path4 不具备 KG 三路的相似度/文章质量等信号，
   不能直接用原始分数与三路混排）。
5. 最后，用**基于 rank 的归一化分数**将 Path4 的 pattern 与三路召回的 pattern 合并到统一排序池。

简言之：**Path4 是一条并行的、自治的召回通路，
它与原有三路通过"各自独立排序 → rank 归一化 → 合并排序池"进行交汇，
而不是在原始分数层面做加权融合。**

---

## 1. Agentic Search 流水线

```
user_idea
    │
    ▼
┌──────────┐
│ Planner  │  LLM 生成 5-8 条英文 query（核心方法 / 相关任务 / 对比方向）
└────┬─────┘
     │
     ▼
┌──────────┐  Semantic Scholar API
│ Searcher │  ◆ 硬过滤：仅保留 venue ∈ 白名单 的论文（其余直接丢弃）
│          │  ◆ 去重：跨 query 去重 + 与 KG 存量 title 去重
└────┬─────┘
     │  novel papers（仅白名单 & KG 外）
     ▼
┌───────────┐
│ Extractor │  LLM 逐篇/批量抽取 paper-level pattern
│           │  （base_problem / gap / solution / story / packaging）
└────┬──────┘
     │
     ▼
┌───────────┐
│ Clusterer │  Embedding 聚类 → 簇级 LLM 合成
│           │  产出 cluster-level pattern（对齐 nodes_pattern.json）
└────┬──────┘
     │
     ▼
┌────────────┐
│ Path4      │  在 Path4 内部独立排序
│ Ranker     │  产出 path4_ranked: [(pattern_id, rank), ...]
└────────────┘
     │
     ▼
  合并排序池（见第 3 节）
```

### 1.1 Planner

- 输入：`user_idea`（+ 可选的 `IdeaBrief`）
- 输出：5-8 条检索 query，每条标注 intent（core_method / related_task / broader_context）
- 实现：单次 LLM 调用，模板 prompt

### 1.2 Searcher

- 对每条 query 调用 Semantic Scholar `/paper/search`，限制 `fieldsOfStudy=Computer Science`、
  `year=近1-2年`
- **Venue 硬过滤**：对返回的每篇论文检查其 `venue` / `publicationVenue` 是否命中白名单；
  **不命中者直接丢弃**，不给分、不参与后续任何环节
- 跨 query 按 paperId 去重
- 与 KG 存量按 title 近似匹配去重（保留"KG 外的新论文"）

#### CS 顶会白名单（可配置）

```
ML/AI:    ICLR, ICML, NeurIPS, AAAI
CV:       CVPR, ICCV, ECCV
NLP:      ACL, EMNLP, NAACL
Data/IR:  KDD, WWW, SIGIR
其他:     CoRL, RSS, OSDI, SOSP, ...（可扩展）
```

> arXiv 预印本默认不在白名单中（因为无同行评审质量信号）；
> 可通过配置开关纳入。

### 1.3 Extractor

- 对 Searcher 输出的每篇论文（title + abstract + venue + year），用 LLM 抽取 pattern：
  - `base_problem`：核心问题
  - `gap_pattern`：现有方法的 gap
  - `solution_pattern`：解决方案
  - `story`：叙事/包装定位
  - `packaging_strategy`：卖点策略类型
  - `domain` / `sub_domains`
- 批量处理（3-5 篇一批），降低 LLM 调用次数

### 1.4 Clusterer

- 将 paper-level pattern 的关键字段拼接后 embedding，用 KMeans 聚类
  （论文数少时可跳过聚类，每篇自成一簇）
- 对每个簇用 LLM 合成 cluster-level pattern，输出结构对齐 `nodes_pattern.json`：
  `name / summary / llm_enhanced_summary / exemplar_papers / ...`
- 每个 pattern 标记 `source: "agentic_search"`

### 1.5 Path4 内部排序

Path4 产出的 pattern 没有 KG 三路召回那样的"相似 Idea 分数"、"Paper 质量"、"领域边权重"等信号。
因此 **Path4 在内部用自己的维度独立排序**：

```
path4_score(pattern) = α · cluster_relevance + β · avg_paper_recency + γ · cluster_size_norm
```

| 维度 | 含义 |
|------|------|
| `cluster_relevance` | 簇中论文与 user_idea 的平均 embedding 相似度 |
| `avg_paper_recency` | 簇中论文的平均时效性（越新越高） |
| `cluster_size_norm` | 簇大小归一化（更多论文汇聚 = pattern 更稳健） |

排序后产出 **path4 的 rank 序列**：rank 1 = 最优，rank 2 = 次优，……

---

## 2. 原有三路召回的 rank 化

现有三路融合后的 `final_scores` 已经是一个排好序的 pattern 列表。
将其转化为 rank 序列即可：

```
三路融合排序后: pattern_A (score=0.82), pattern_B (score=0.71), pattern_C (score=0.65), ...
  → rank 序列: pattern_A → rank 1, pattern_B → rank 2, pattern_C → rank 3, ...
```

---

## 3. 合并排序：Rank-Based Fusion

### 3.1 为什么不用原始分数直接加权

两条通路的分数**量纲不同、分布不同**：
- 三路召回的分数来自 embedding 相似度 × 图谱边权重 × 质量分，值域和分布由 KG 决定
- Path4 的分数来自联网论文的 relevance × recency × cluster_size，值域完全不同

直接做 `w_old * score_old + w_new * score_new` 会导致：
一条通路的绝对分数碰巧高就"淹没"另一条，无法公平比较。

### 3.2 Reciprocal Rank Fusion (RRF)

采用经典的 **Reciprocal Rank Fusion** 来合并两条通路的排序：

```
RRF_score(pattern) = Σ  1 / (k + rank_in_path_i)
                    path_i ∈ {三路融合, Path4}
```

其中 `k` 是平滑常数（通常取 60），作用是避免 rank 1 的分数过于"尖锐"。

**具体步骤**：

1. **三路融合**按现有逻辑跑完，得到排序序列 `ranked_old`，取 Top-N（如 N=15）
2. **Path4** 按内部逻辑跑完，得到排序序列 `ranked_p4`，取 Top-M（如 M=8）
3. 对所有出现过的 pattern，计算 RRF 分数：

```python
k = 60  # 平滑常数

rrf_scores = {}
for pattern_id, rank in ranked_old:
    rrf_scores[pattern_id] = rrf_scores.get(pattern_id, 0) + 1.0 / (k + rank)
for pattern_id, rank in ranked_p4:
    rrf_scores[pattern_id] = rrf_scores.get(pattern_id, 0) + 1.0 / (k + rank)

final_ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
```

4. 取最终 Top-K（如 K=10）作为召回结果

### 3.3 通路级别的权重控制

如果希望控制"三路 vs Path4"的整体贡献比例，可以在 RRF 前对各通路乘以通路权重：

```python
w_old = 0.7   # 三路召回的通路权重
w_p4  = 0.3   # Path4 的通路权重

for pattern_id, rank in ranked_old:
    rrf_scores[pattern_id] += w_old * (1.0 / (k + rank))
for pattern_id, rank in ranked_p4:
    rrf_scores[pattern_id] += w_p4 * (1.0 / (k + rank))
```

这样 `w_old` 和 `w_p4` 控制的是**通路的话语权**而不是原始分数的缩放，更加稳定和可解释。

### 3.4 合并排序的架构图

```
┌─────────────────────────────┐     ┌──────────────────────────┐
│   三路召回 (Path1+2+3)       │     │   Path4: Agentic Search   │
│                             │     │                          │
│  Path1 scores ─┐            │     │  Planner → Searcher      │
│  Path2 scores ─┤→ 加权融合   │     │    → Extractor           │
│  Path3 scores ─┘  → 排序    │     │    → Clusterer           │
│                    ↓        │     │    → 内部排序             │
│           ranked_old        │     │         ↓                │
│        (rank 1,2,3,...)     │     │     ranked_p4            │
└─────────────┬───────────────┘     └──────────┬───────────────┘
              │                                │
              └──────────┬─────────────────────┘
                         ▼
              ┌─────────────────────┐
              │  Reciprocal Rank    │
              │  Fusion (RRF)       │
              │                     │
              │  RRF(p) = Σ w_i /   │
              │          (k+rank_i) │
              └──────────┬──────────┘
                         │
                         ▼
                  Final Top-K
                  (统一结果)
```

---

## 4. 关键设计决策摘要

| 决策 | 选择 | 理由 |
|------|------|------|
| Venue 过滤策略 | **硬过滤**（不在白名单 → 直接丢弃） | 减少噪声、控制质量，CS 顶会已足够覆盖高质量信号 |
| 两条通路的融合方式 | **Rank-based (RRF)**，不做原始分数加权 | 两条通路分数量纲不同，rank 融合更公平、更鲁棒 |
| Path4 内部排序 | 独立维度（relevance + recency + cluster_size） | Path4 没有 KG 的质量/边权信号，不应与三路共用打分体系 |
| 数据源 | Semantic Scholar（MVP） | 免费、有 abstract、venue/year 过滤好用 |
| pattern 输出格式 | 对齐 `nodes_pattern.json` | 下游 packaging/critic 不需要改动 |
| 失败策略 | Path4 任何故障 → 返回空，不阻塞其他三路 | 增量功能不应降低系统稳定性 |

---

## 5. 配置与开关

```python
# 总开关
PATH4_ENABLE = True

# Rank 融合参数
PATH4_RRF_K = 60                  # RRF 平滑常数
PATH4_RRF_WEIGHT = 0.3            # Path4 在 RRF 中的通路权重（三路 = 1 - 0.3 = 0.7）
PATH4_TOP_M = 8                   # Path4 参与 RRF 的 pattern 数

# 搜索预算
PATH4_MAX_QUERIES = 8             # Planner 最多 query 数
PATH4_YEAR_RANGE = "2024-2026"
PATH4_VENUE_WHITELIST = "cs_top"  # cs_top | custom

# 缓存
PATH4_CACHE_TTL_DAYS = 7
```

---

## 6. 落地阶段

| 阶段 | 内容 | 周期 |
|------|------|------|
| **Phase 1 (MVP)** | Planner + Searcher(S2, 白名单硬过滤) + Extractor + Clusterer + 内部排序 + RRF 合并 | 1 周 |
| **Phase 2** | Backtrack 机制、多数据源、IdeaPackager evidence fallback、前端展示 | 2 周 |
| **Phase 3** | PDF 深抽取、HDBSCAN、增量缓存、Path4 采纳率统计 | 持续 |
