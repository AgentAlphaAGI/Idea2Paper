# Path4 Agentic Search — Planner & Searcher 具体设计 (v0.1)

> 对齐方案：`docs/PATH4_AGENTIC_SEARCH_DESIGN_v2.md` 的 1.1 Planner / 1.2 Searcher  
> 范围：只讨论「query 生成」与「在哪里/怎么搜」；暂不包含去重/缓存/抽取/排序/融合细节  

---

## 0. 总体目标（文字性描述）

Path4 的 Planner + Searcher 要做的事可以概括为一句话：

**把用户的研究 idea 转成一组“可覆盖核心术语 + 可控噪声”的英文检索查询（Planner），并仅在 CS 顶会白名单范围内，从结构化学术数据源拉回一批“高相关、近两年”的候选论文（Searcher），为后续 pattern 抽取提供输入。**

这里的关键约束来自 v2 方案的两条原则：

- **白名单硬过滤**：venue 不在白名单 → 直接跳过（不给分、不进入后续链路）
- **rank 融合**：Path4 后续会独立排序，因此 Planner/Searcher 更关注“召回质量与覆盖”，而不是与 KG 三路共享打分体系

---

## 1. Planner：如何生成搜索 query

### 1.1 输入 / 输出接口（高层）

- **输入**
  - `user_idea: str`
  - （可选）`idea_brief: dict`：如果 Pipeline 已启用 `IdeaPackager.parse_raw_idea()`，Planner 可读取：
    - `keywords_en`
    - `problem_definition`
    - `technical_plan`
    - `constraints`

- **输出**
  - `queries: List[QuerySpec]`，建议 5–8 条为主，最多 10 条
  - `search_intent` 标注（便于审计与后续调参），例如：
    - `core_method`：核心方法/模型/损失/结构
    - `task_setting`：任务/场景/数据形态/约束
    - `evaluation`：benchmark/metric/setting（可选，避免过拟合到 benchmark 名）
    - `related_area`：同源方法族/近邻领域
    - `contrast`：反向词（如“without supervision / efficient / long context”）

> 备注：Planner 只负责生成 query，不负责选择数据源与 API 参数（那是 Searcher 的工作）。

### 1.2 Query 生成策略（推荐：LLM + 规则混合）

Planner 最稳的落地方式不是“纯 prompt”，而是 **LLM 生成候选 → 规则约束与裁剪**：

#### Step A：抽取“检索锚点”（anchors）

从 `user_idea` / `idea_brief` 里得到三类 anchors（每类 3–8 个词组）：

- **Method anchors**：模型/算法/训练范式/模块（例如 “retrieval-augmented generation”, “diffusion policy”, “speculative decoding”）
- **Task anchors**：任务/输入输出/场景（例如 “long-context reasoning”, “video understanding”, “3D reconstruction”）
- **Constraint anchors**：约束/卖点（例如 “efficient”, “robust”, “low-resource”, “privacy”）

> 如果 `idea_brief.keywords_en` 存在，优先用它作为 anchors 的来源（更贴近用户 intent）。

#### Step B：生成 query 模板（LLM）

让 LLM 按固定模板输出 8–12 条候选 query，然后做裁剪到 5–8 条：

**模板约束（非常重要，降低噪声）**

- 每条 query **只包含 2–4 个 anchor**（避免把一句话全部塞进去导致召回稀释）
- 强制英文、禁止引号语法/高级检索语法（兼容 S2 的 plain-text search）
- 避免出现过宽词（如 “deep learning”, “neural network”）作为唯一关键词
- 允许 1–2 条 query 包含“对比/反向”词（例如 “efficient”, “without labels”）用于召回更贴近 gap 的论文

**Planner LLM 的输出 JSON（示意）**

```json
{
  "anchors": {
    "method": ["...", "..."],
    "task": ["...", "..."],
    "constraint": ["...", "..."]
  },
  "candidates": [
    {"q": "method_anchor + task_anchor", "intent": "core_method", "must_have": ["..."]},
    {"q": "task_anchor + constraint_anchor + method_anchor", "intent": "task_setting", "must_have": ["..."]}
  ]
}
```

#### Step C：规则裁剪与多样性控制

从候选 query 中选出最终 query 列表：

- **多样性**：intent 至少覆盖 `core_method` 与 `task_setting`，其余 intent 可选
- **去重**：query token 集合 Jaccard > 0.8 的视为重复，只保留更短、更“锚点密度高”的那条
- **长度**：建议 6–14 个英文词；过长会削弱 S2 的相关性排序

### 1.3 Planner 的两种运行模式

- **Fast mode（默认）**：一次 LLM 调用生成候选 + 规则裁剪
- **Backtrack mode（可选）**：如果 Searcher 的结果在白名单硬过滤后几乎为空（例如 < 5 篇），Planner 再生成 2–3 条“更宽松”的 query（例如丢掉 constraint anchors，只保留 method+task），但仍坚持白名单硬过滤

> 注意：Backtrack 的目的是避免 query 过窄导致“搜不到”，不是放松白名单质量门槛。

---

## 2. Searcher：怎么搜、搜哪里（含 Google Scholar 可行性）

### 2.1 “直接搜 Google Scholar 可以吗？”

不建议把 Google Scholar 作为你系统的主检索源，原因是工程上不可控：

- **没有官方 API**，且 `robots.txt` 明确禁止抓取 `/search` 等路径（程序化爬取极易被封/不稳定）
- 结果页结构经常变化，解析成本高，且不具备稳定的字段（venue/year/pdf/abstract）保证
- 从合规与可复现角度，Scholar 更适合作为“人工辅助入口”，而不是 pipeline 的核心依赖

结论：**Planner/Searcher 的 MVP 应以结构化学术 API 为主（Semantic Scholar），而不是 Google Scholar。**

### 2.2 数据源选择（MVP）

按“可落地 + 字段齐全 + 可过滤 + 可复现”排序：

1. **Semantic Scholar Academic Graph API（主力）**
   - 优点：有 `venue / publicationVenue / year / abstract / openAccessPdf` 等字段；支持 `year`、`fieldsOfStudy` 过滤；结果较干净
2. （后续可加）OpenAlex / OpenReview / arXiv
   - v0.1 先不引入，避免扩大不确定性面

> 参考：Gemini Deep Research 的公开描述强调“反复规划→搜索→读→发现缺口→再搜索”的迭代式研究工作流。
> 我们在 v0.1 里只借鉴其 **planning+iterative search 的结构**，但数据源仍选择可工程化的学术 API。

### 2.3 Searcher 的调用策略（Semantic Scholar）

对每条 query：

- endpoint：`GET /graph/v1/paper/search`
- 过滤参数（建议）：
  - `fieldsOfStudy=Computer Science`
  - `year=2024-2026`（或“近两年”配置项）
  - `limit=50`（单 query 拉回上限）
  - `fields=` 建议至少包含：
    - `paperId,externalIds,title,abstract,venue,publicationVenue,year,publicationDate,authors,url,isOpenAccess,openAccessPdf,citationCount,influentialCitationCount,tldr`

**为什么不在 API 层强行加 `venue=`？**

- S2 的 `venue` 过滤是“字符串匹配”，实际 venue 命名有多种别名；API 层过滤容易漏召回
- 我们采用 v2 要求的 **白名单硬过滤**，放在结果后处理阶段做“更可靠”的规范化匹配

### 2.4 白名单硬过滤（核心）

对每篇返回论文做 venue 规范化（只要一个稳定规则即可）：

1. 取 `publicationVenue.name`（若存在）否则取 `venue` 字符串
2. lower + 去标点 + 去多空格
3. 用白名单 alias 表做“包含匹配”（例如包含 `\"international conference on machine learning\"` 或 `\"icml\"`）

**硬过滤规则：**

- 命中白名单 → 保留
- 不命中 → 直接丢弃（不进入 Extractor，不参与任何 ranking）

### 2.5 结果输出（不含去重的 v0.1）

由于你说“暂时不用管去重”，此阶段 Searcher 的输出可以非常简单：

- `papers: List[PaperStub]`
  - `paper_id, title, abstract, venue_norm, year, url, external_ids, source_query`

> 后续去重（跨 query、与 KG 存量）可以在 Searcher 或独立 Dedup 模块补上；\n+> v0.1 先把“能稳定搜到白名单内论文”打通。

---

## 3. 建议的最小落地（v0.1）

为了让这一步尽快可跑，建议先做以下“最小可用”决策：

- Planner：一次 LLM 生成 8 条候选 query → 规则裁剪成 6 条
- Searcher：只用 Semantic Scholar；每条 query 拉回 Top 50；白名单硬过滤后合并（可先不 dedup）
- 输出：保证每篇论文至少有 `title + abstract + venue_norm + year + url`，便于后续 Extractor 抽取

