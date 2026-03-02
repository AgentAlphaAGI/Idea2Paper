# Path4 arXiv Track — 当前方案说明

> 对齐实现：`src/idea2paper/agentic_search/searcher.py` / `ranker.py` / `config.py`  
> 状态：Implemented | 2026-03-02

---

## 1. 目标与范围

arXiv Track 是 Agentic Search 的预印本补充召回通路，与 OpenAlex 并行执行。

- OpenAlex 通路负责顶会白名单内的已发表论文召回。
- arXiv 通路负责前沿 preprint 召回，不使用 venue 白名单。
- 两路在 Searcher 阶段合并去重，进入统一 Extractor 和 Ranker。

---

## 2. Searcher 方案（OpenAlex + arXiv）

`AgenticSearcher.search()` 对每条 planner query 执行两路检索：

1. OpenAlex 路：`_search_openalex()`，再做 venue whitelist 过滤。  
2. arXiv 路：`_search_arxiv_raw()` + `_filter_and_score_arxiv()`。

最终输出是统一 `PaperStub` 列表，含来源统计。

### 2.1 arXiv 检索语法与 fallback

arXiv 使用 Atom API：`https://export.arxiv.org/api/query`。

每条 query 的 arXiv 检索策略：

- 先构造核心短语（默认取 query 的前 3 个有效词）。
- 首先查询 `ti:"<core phrase>"`（标题精确短语）。
- 若返回条数 `< 5`，fallback 到 `abs:"<core phrase>"`（摘要精确短语）。
- 当 fallback 结果更大时采用 fallback 结果，否则保留原结果。

说明：

- 这一步是召回策略，不是质量过滤。
- fallback 触发不代表“arXiv 没有相关论文”，仅表示标题短语匹配偏严格。

### 2.2 arXiv 过滤与预筛

arXiv 原始结果会经过“硬过滤 + 软预筛”。

硬过滤（任一不满足即丢弃）：

- arXiv ID 有效。
- 不与已见 arXiv ID 重复。
- `published.year >= AGENTIC_SEARCH_ARXIV_MIN_YEAR`（默认 2025）。
- `abstract` 长度 >= 100。
- 若开启分类过滤：`primary_category` 在白名单内（默认 CS 子分类）。

软预筛分数：

```text
prescreen = 0.6 * recency + 0.2 * revision + 0.2 * category
```

- `recency`：相对 `min_year` 的时间新鲜度（上限按 2 年封顶）。
- `revision`：版本号映射（v1=0，v4+=1.0）。
- `category`：核心类别 1.0，扩展类别 0.7，其它 0.4。

低于阈值 `AGENTIC_SEARCH_ARXIV_MIN_PRESCREEN_SCORE`（默认 0.2）会被丢弃。  
通过后按 `prescreen` 排序，保留每 query 的 top-N（默认 5）。

### 2.3 去重逻辑

- OpenAlex 内部：按 OpenAlex paper_id 去重。
- arXiv 内部：按 arXiv ID 去重。
- 跨源：基于已收集的 arXiv ID 做去重（OpenAlex 优先）。

---

## 3. Pattern 抽取与统一格式

Extractor 对合并后的论文统一抽取 paper-level pattern。

- arXiv 来源 pattern 与 OpenAlex 来源 pattern 格式一致。
- 区分字段主要是 `venue_norm`（arXiv 为 `"arXiv"`）与引用量（通常为 0）。
- 输出文件：`output/agenticSearch_patterns.json`。

---

## 4. Ranker 方案（当前实现）

Ranker 统一打分公式：

```text
agenticSearch_score = alpha * relevance + beta * recency_norm + gamma * impact_norm
```

默认权重：

- `alpha=0.5`
- `beta=0.3`
- `gamma=0.2`

其中：

- `relevance`：idea 与 pattern summary 的 embedding cosine。
- `recency_norm`：按 pattern 年份做全局 min-max。
- `impact_norm`：按 `log(1 + citation_count)` 做全局 min-max。

---

## 5. 来源公平重排（6:4）

为避免单一来源占据 head 排名，Ranker 在打分后执行 source-mix 头部重排。

重排位置：`Path4Ranker.rank()` 内部，原始排序完成后执行。

重排规则：

- 仅作用于头部 `top_n`（默认对齐 `AGENTIC_SEARCH_RRF_TOP_M`）。
- 按比例分配来源名额（默认 OpenAlex:arXiv = 0.6:0.4）。
- 两侧不足时名额自动 spillover 给另一侧。
- 头部按交错方式混排，保持双方可见性。
- 重排后回写 `rank_scores.rank`（供下游 RRF 使用）。

---

## 6. 与 KG 融合（RRF）

主流程中，Agentic Search 结果通过 `fuse_old_and_agentic_search()` 与 KG 召回融合。

- KG 输入：`top_n_old`（默认 10）。
- Agentic 输入：`top_m_agentic`（默认 8，来自重排后的 head）。
- 融合：Weighted RRF（`weight_old=0.7`, `weight_agentic=0.3`，默认）。
- 输出：`final_top_k`（默认 10）。

说明：

- 融合阶段不再额外按来源做硬配额，来源平衡由 Ranker 的 source-mix 先行控制。

---

## 7. 关键配置项（当前）

### 7.1 arXiv 检索与过滤

- `AGENTIC_SEARCH_ARXIV_ENABLE`（默认 `True`）
- `AGENTIC_SEARCH_ARXIV_MIN_YEAR`（默认 `2025`）
- `AGENTIC_SEARCH_ARXIV_MAX_RESULTS_PER_QUERY`（默认 `25`）
- `AGENTIC_SEARCH_ARXIV_TOP_N_PER_QUERY`（默认 `5`）
- `AGENTIC_SEARCH_ARXIV_REQUEST_INTERVAL_SEC`（默认 `3.0`）
- `AGENTIC_SEARCH_ARXIV_MIN_PRESCREEN_SCORE`（默认 `0.2`）
- `AGENTIC_SEARCH_ARXIV_CATEGORY_FILTER_ENABLE`（默认 `True`）
- `AGENTIC_SEARCH_ARXIV_CS_CATEGORIES`（默认 `cs.LG, cs.AI, cs.CL, cs.CV, cs.IR, cs.RO, cs.NE`）

### 7.2 source-mix 公平重排

- `AGENTIC_SEARCH_SOURCE_MIX_ENABLE`（默认 `True`）
- `AGENTIC_SEARCH_SOURCE_MIX_RATIO_OPENALEX`（默认 `0.6`）
- `AGENTIC_SEARCH_SOURCE_MIX_RATIO_ARXIV`（默认 `0.4`）
- `AGENTIC_SEARCH_SOURCE_MIX_TOP_N`（默认 `AGENTIC_SEARCH_RRF_TOP_M`）

### 7.3 与 KG 融合（RRF）

- `AGENTIC_SEARCH_RRF_K`（默认 `60`）
- `AGENTIC_SEARCH_RRF_WEIGHT_OLD`（默认 `0.7`）
- `AGENTIC_SEARCH_RRF_WEIGHT_AGENTIC`（默认 `0.3`）
- `AGENTIC_SEARCH_RRF_TOP_N_OLD`（默认 `10`）
- `AGENTIC_SEARCH_RRF_TOP_M`（默认 `8`）
- `AGENTIC_SEARCH_RRF_FINAL_TOP_K`（默认 `10`）

---

## 8. 产物与调试文件

- `output/agenticSearch_patterns.json`：Ranker 后的 pattern（含 source-mix 结果）。
- `output/agenticSearch_venue_debug.json`：OpenAlex 侧 venue 命中/未命中诊断。
- `output/agenticSearch_result_*.json`：完整 pipeline 结果（含 per-query OA/arXiv 统计）。

