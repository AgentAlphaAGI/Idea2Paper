# Path4 Extractor — abstract → paper-level pattern 抽取 (v0.1)

> 对齐方案：`docs/PATH4_AGENTIC_SEARCH_DESIGN_v2.md` 的 1.3 Extractor  
> 范围：从 `PaperStub`（含 abstract）出发，抽取与 `nodes_pattern.json` 对齐的 paper-level pattern  
> 约束：不涉及 KG 去重；每篇单独产出一个 pattern（skip 聚类，留 v0.2）

---

## 0. 内容来源：abstract

原始 KG 的 pattern（`base_problem / solution_pattern / story`）也是从 abstract 抽取的，因此 Path4 直接使用 Searcher 已拉回的 abstract，无需下载 PDF 或获取全文。

**Searcher 阶段已有字段，直接复用：**

| 字段 | 内容 |
|------|------|
| `title` | 论文标题 |
| `abstract` | 摘要（通常 150-300 词，S2 返回） |
| `venue_norm` / `year` | 会议 + 年份 |

若 abstract 为空，跳过该论文（记 `skip_reason: "no_abstract"`）。

---

## 1. 抽取接口

### 1.1 输入 / 输出

- **输入**：`List[PaperStub]`（Searcher 输出）
- **输出**：`List[PaperPattern]`，字段对齐 `nodes_pattern.json`

### 1.2 `PaperPattern` 数据结构

```json
{
  "pattern_id": "p4_<paper_id_short>",
  "source": "agentic_search",
  "paper_id": "<S2 paperId>",
  "title": "...",
  "venue_norm": "ICLR",
  "year": 2024,
  "url": "...",
  "name": "≤8词的叙事名（LLM生成）",
  "domain": "Machine Learning",
  "sub_domains": ["Tool Use", "Code Generation", "Planning"],
  "size": 1,
  "summary": {
    "base_problem":     "...",
    "gap_pattern":      "...",
    "solution_pattern": "...",
    "story":            "...",
    "application":      "..."
  },
  "packaging_strategy": "contrast | extend | reframe | apply"
}
```

> `size=1` 表示 paper-level（未聚类）。后续聚类时 size 变大，`summary` 子结构保持一致。  
> 字段对齐 `nodes_pattern.json` 的 `summary` 结构，下游不需要改动。

---

## 2. 抽取策略

### 2.1 批量 LLM 调用（3-5 篇/批）

单篇 abstract 约 200 tokens，5 篇一批约 1200 tokens，一次 LLM 调用可以完成，节省调用次数。

**Prompt 结构：**

```
System:
You are a research paper analyst. Extract structured research patterns from paper abstracts.

User:
Below are {n} papers. For EACH paper output one JSON object in the array.

## Papers
[1] Title: {title}  Venue: {venue_norm} {year}
    Abstract: {abstract}
[2] ...

## Output Format (JSON array only, no markdown, order must match input)
[
  {
    "index": 1,
    "name": "<≤8-word narrative name>",
    "domain": "...",
    "sub_domains": ["...", "..."],
    "base_problem": "The core problem addressed (2-3 sentences)",
    "gap_pattern": "What existing methods fail to do (2-3 sentences)",
    "solution_pattern": "The proposed solution and key mechanism (2-3 sentences)",
    "story": "The narrative framing / positioning angle (1-2 sentences)",
    "application": "Target domain or downstream use case (1 sentence)",
    "packaging_strategy": "contrast | extend | reframe | apply"
  },
  ...
]
```

### 2.2 失败处理

| 场景 | 处理 |
|------|------|
| 批量 JSON 解析失败 | 自动拆成单篇重试一次 |
| 单篇重试仍失败 | skip，记 `skip_reason: "llm_parse_failed"` |
| abstract 为空 | skip，记 `skip_reason: "no_abstract"` |

所有 skip 记录到 `extraction_stats`，不阻塞链路。

### 2.3 字段缺失容忍

- 任意字段缺失时用空字符串填充，不报错
- `packaging_strategy` 无法识别时默认 `"extend"`

---

## 3. 决策摘要

| 决策点 | 选择 | 理由 |
|--------|------|------|
| 内容来源 | abstract（S2 已有） | 与原 KG pattern 提取方式一致 |
| 批量大小 | 3-5 篇/批 | 平衡 token 成本与 JSON 解析稳定性 |
| 失败处理 | 单篇重试一次，仍失败则 skip | 不阻塞链路 |
| 输出格式 | 对齐 `nodes_pattern.json` summary 字段 | 下游不改 |
| 聚类 | 跳过，每篇 `size=1` | v0.2 补充 |

---

## 4. 与 pipeline 的衔接

```
PaperStub list (Searcher 输出，abstract 已包含)
    │
    ▼
┌──────────────────┐
│  LLM Extractor   │  批量 3-5 篇/次 → PaperPattern list
│                  │  JSON 失败 → 单篇重试 → skip
└────────┬─────────┘
         │
         ▼
  List[PaperPattern]
  落盘 path4_patterns.json
         │
         ▼
   (v0.2) Clusterer → Path4 Ranker → RRF 融合
```

`path4_patterns.json` 格式与 `nodes_pattern.json` 兼容，通过 `source: "agentic_search"` 字段区分来源。
