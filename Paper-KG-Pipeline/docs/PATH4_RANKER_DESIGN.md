# Path4 Ranker — pattern 内部排序 (v0.1)

> 对齐方案：`docs/PATH4_AGENTIC_SEARCH_DESIGN_v2.md` 的 1.5 Path4 内部排序  
> 范围：对 `path4_patterns.json` 中的 paper-level pattern 做独立打分排序，产出 `path4_ranked`  
> 约束：不涉及 pattern 聚类；跳过聚类时每篇 pattern `size=1`

---

## 0. 为什么需要独立排序

Path4 pattern 没有 KG 三路召回的图谱边权重、embedding 存量等信号，
不能直接与三路原始分数混排。因此先在 Path4 内部独立打分，
再通过 RRF 与三路做 rank 级融合（见总体方案 §2-3）。

---

## 1. 打分信号

### 1.1 信号一览

| 信号 | 含义 | 数据来源 | 当前状态 |
|------|------|---------|---------|
| `relevance` | 论文与 user_idea 的语义相关度 | embed(user_idea) vs embed(summary_text)，Ranker 运行时计算 | 无需改动 |
| `recency` | 发表年份（越新越高） | `PaperPattern.year`，Extractor 已写入 | 已有 ✓ |
| `impact` | 引用数（log 变换，质量代理） | OpenAlex `cited_by_count` → `PaperPattern.citation_count` | 已有 ✓ |

> **关于 Review 评分**：OpenAlex 不提供同行评审打分（OpenReview 有 ICLR/NeurIPS 打分，但需额外 API 且覆盖不全）。因为已对 venue 做硬过滤，所有 pattern 均来自顶会同行评审论文，使用引用数作为质量代理已足够。OpenReview 评分可作为 Phase 2 可选增强。

### 1.2 `citation_count` 字段来源

OpenAlex Searcher 已返回 `cited_by_count`，并在 Searcher 阶段写入 `PaperStub.citation_count`；
Extractor 会从 `PaperStub` 复制到 `PaperPattern.citation_count`：

```python
# PaperPattern dataclass 新增
citation_count: int = 0
```

```python
# Extractor 构建 PaperPattern 时：
citation_count = stub.citation_count  # 从 PaperStub 直接复制
```

对已有的 `path4_patterns.json`，缺失此字段时默认为 `0`，不影响排序逻辑（仅 impact 维度退化）。

---

## 2. 打分公式

### 2.1 各维度归一化

对当前 pattern 集合做 **min-max 归一化**（batch 内归一，不跨 run 比较）：

```
recency_norm(p)  = (year(p) - year_min) / max(year_max - year_min, 1)
impact_raw(p)    = log(1 + citation_count(p))
impact_norm(p)   = (impact_raw(p) - imp_min) / max(imp_max - imp_min, 1e-6)
relevance(p)     = cosine_sim(embed(user_idea), embed(summary_text(p)))   ∈ [0, 1]
```

`summary_text(p)` = 拼接 `representative_ideas + common_problems + solution_approaches`（前 200 词即可）。

### 2.2 加权求和

```
path4_score(p) = α · relevance(p) + β · recency_norm(p) + γ · impact_norm(p)
```

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `α` | 0.5 | 相关度权重（最重要） |
| `β` | 0.3 | 时效性权重 |
| `γ` | 0.2 | 引用质量权重 |

三个参数均可通过 `PipelineConfig` 配置项覆盖。

---

## 3. 输入 / 输出接口

- **输入**：
  - `patterns: List[PaperPattern]`（Extractor 输出）
  - `user_idea: str`（用于计算 relevance）
- **输出**：
  - `path4_ranked: List[tuple[str, int]]`，即 `[(pattern_id, rank), ...]`，rank 从 1 开始
  - 每个 `PaperPattern` 附加打分明细字段（可选落盘，便于调试）：
    ```json
    "rank_scores": {
      "relevance": 0.82,
      "recency_norm": 0.75,
      "impact_norm": 0.40,
      "path4_score": 0.69
    }
    ```

---

## 4. 实现概要

```python
class Path4Ranker:
    def rank(
        self,
        patterns: List[PaperPattern],
        user_idea: str,
    ) -> List[tuple[str, int]]:
        if not patterns:
            return []

        # 1. embed user_idea（单次调用）
        idea_emb = embed(user_idea)

        # 2. 各维度原始值
        years         = [p.year or 0 for p in patterns]
        impacts_raw   = [math.log(1 + (p.citation_count or 0)) for p in patterns]
        relevances    = [cosine_sim(idea_emb, embed(summary_text(p))) for p in patterns]

        # 3. min-max 归一化
        recency_norms = minmax_norm(years)
        impact_norms  = minmax_norm(impacts_raw)

        # 4. 加权求和 → 排序
        scores = [
            ALPHA * rel + BETA * rec + GAMMA * imp
            for rel, rec, imp in zip(relevances, recency_norms, impact_norms)
        ]
        ranked_indices = sorted(range(len(patterns)), key=lambda i: scores[i], reverse=True)

        return [(patterns[i].pattern_id, rank + 1) for rank, i in enumerate(ranked_indices)]
```

**Embedding 调用优化**：pattern 数量通常 ≤ 20，所有 `summary_text` 可批量送入 embedding API，一次调用完成。

---

## 5. 边界与失败处理

| 场景 | 处理 |
|------|------|
| patterns 为空 | 返回 `[]`，不阻塞后续 RRF |
| embedding 调用失败 | relevance 全部退化为 0，仅用 recency + impact 排序，记 warning |
| year 全部为 0（数据缺失） | recency_norm 全为 0，不影响其他维度 |
| citation_count 全为 0 | impact_norm 全为 0，等价于 `γ=0`，不影响排序 |
| 所有 score 相等（极端情况） | 保持原始顺序（稳定排序） |

---

## 6. 与 pipeline 的衔接

```
List[PaperPattern]  （Extractor 输出 / path4_patterns.json）
    │
    ▼
┌─────────────────┐
│  Path4 Ranker   │  输入: patterns + user_idea
│                 │  → embed user_idea（1次）
│                 │  → batch embed summary_texts
│                 │  → 加权打分 → 排序
└────────┬────────┘
         │
         ▼
  path4_ranked: [(pattern_id, rank), ...]
         │
         ▼
  RRF 融合（与三路召回合并，见总体方案 §3）
```

---

## 7. 决策摘要

| 决策点 | 选择 | 理由 |
|--------|------|------|
| Review 评分 | 不引入（暂） | OpenAlex 不提供；顶会硬过滤已保证基础质量；引用数作为代理已足够 |
| 引用数处理 | log(1 + count) + min-max 归一化 | 压缩长尾分布；与 year 归一化量纲对齐 |
| 相关度计算时机 | Ranker 运行时（不在 Extractor 存储） | user_idea 只有在 run 时才确定；summary_text 已是语义压缩后的文本，embedding 效果好 |
| 向量模型 | 复用 Pipeline 已有 embedding 配置 | 与现有三路召回共用，避免多模型维护 |
| 归一化范围 | batch 内 min-max | Path4 不跨 run 比较，batch 内归一更稳定 |
