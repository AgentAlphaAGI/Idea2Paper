# 论文生成链路（Paper Workflow）

这是一个“从一句想法到可编译 LaTeX/Overleaf 工程”的多阶段工作流：选套路/（可选）查重 → 写稿 →（可选）多 Reviewer 评审回退 → 导出 Overleaf 工程并本地编译验证。

本仓库的核心入口是 CLI：`python -m api.paper_cli workflow ...`

---

## 1. 拉取后需要做什么（最短路径）

1) 安装 Python 依赖（见下文）
2) 安装系统依赖：Pandoc + LaTeX 工具链（见下文，WSL/Ubuntu 给了完整命令）
3) 复制 `.env` 并填写 API Key（最少只需要 `OPENAI_API_KEY`）
4) 修改 `config_paper.yaml`（至少确认 `output_format` / 模板 / 导出目录）
5) 跑一次端到端：

```bash
python -m api.paper_cli workflow "我想写一篇关于代码库检索与打包的论文" --config config_paper.yaml --snapshot-path knowledge_graph.json
```

输出工程默认在 `outputs/overleaf/`，会包含 `overleaf.zip`。

---

## 2. 安装依赖

### 2.1 Python 依赖（pip）

建议使用虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### 2.2 系统依赖（LaTeX + Pandoc）

`requirements.txt` 只包含 Python 包；LaTeX/Pandoc 是系统依赖，需要用系统包管理器安装。

#### WSL / Ubuntu（推荐）

```bash
sudo apt-get update
sudo apt-get install -y \
  pandoc \
  latexmk \
  texlive-xetex \
  texlive-latex-extra \
  texlive-fonts-recommended \
  texlive-fonts-extra \
  chktex \
  lacheck
```

说明：
- 你遇到过的 `inconsolata.sty not found`，通常通过安装 `texlive-fonts-extra` 解决。
- 若你只想导出 Overleaf 工程、不在本机编译，可在配置里将 `latex_require_tools: false` 或 `latex_final_compile: false`，跳过本地工具链依赖。

#### macOS（示例）

- Pandoc：`brew install pandoc`
- LaTeX：安装 MacTeX（或 `mactex-no-gui`），并确保 `latexmk` 可用。

---

## 3. 配置说明

### 3.1 `.env` / `.env.example`（环境变量）

复制一份：

```bash
cp .env.example .env
```

最小必填：
- `OPENAI_API_KEY`：LLM 调用的 API Key（支持 OpenAI-compatible 服务）

### 3.1.1 `.env.example` 变量逐项说明

全局 LLM（所有 agent 的默认值）：
- `OPENAI_API_KEY`：默认 LLM API Key（未给某个 agent 单独配置时会用它）。
- `OPENAI_MODEL`：默认聊天模型名（如 `gpt-4o-mini`）。
- `OPENAI_BASE_URL`：OpenAI-compatible 网关地址（例如你自建/第三方网关）；为空则使用默认 OpenAI。
- `OPENAI_TEMPERATURE`：LLM 采样温度（注意：这不是 KG 选套路的温度）。
- `OPENAI_TIMEOUT`：LLM 请求超时秒数。

提示词目录：
- `PROMPT_DIR`：非论文模块的提示词目录覆盖（如果你不用其他模块可以忽略）。
- `PAPER_PROMPT_DIR`：论文链路提示词目录覆盖；一般不手动设，CLI 会根据 `paper_language` 自动指向 `prompts_paper_en/` 或 `prompts_paper/`。

引用检索：
- `S2_API_KEY`：Semantic Scholar API Key（当 `citations_provider=semantic_scholar` 且 `citations_require_api_key: true` 时必填）。

Pandoc：
- `PANDOC_BIN`：pandoc 可执行文件路径覆盖（默认直接用 `pandoc`）。

按 Agent 单独覆盖（高级用法；优先级高于全局 `OPENAI_*`）：
- 规则：程序会优先读取 `PREFIX_OPENAI_*`，没有才回退到全局 `OPENAI_*`。
- `PREFIX` 来自 agent 名称的大写（例如 `paper_section_writer` → `PAPER_SECTION_WRITER_*`）。

各 agent 的变量（每组含义相同：Key/Model/Base URL/Temperature/Timeout）：

- 解析输入（`paper_parse`）：
  - `PAPER_PARSE_OPENAI_API_KEY`
  - `PAPER_PARSE_OPENAI_MODEL`
  - `PAPER_PARSE_OPENAI_BASE_URL`
  - `PAPER_PARSE_OPENAI_TEMPERATURE`
  - `PAPER_PARSE_OPENAI_TIMEOUT`

- 写故事骨架与整体草稿（`paper_story`）：
  - `PAPER_STORY_OPENAI_API_KEY`
  - `PAPER_STORY_OPENAI_MODEL`
  - `PAPER_STORY_OPENAI_BASE_URL`
  - `PAPER_STORY_OPENAI_TEMPERATURE`
  - `PAPER_STORY_OPENAI_TIMEOUT`

- 章节写作（`paper_section_writer`）：
  - `PAPER_SECTION_WRITER_OPENAI_API_KEY`
  - `PAPER_SECTION_WRITER_OPENAI_MODEL`
  - `PAPER_SECTION_WRITER_OPENAI_BASE_URL`
  - `PAPER_SECTION_WRITER_OPENAI_TEMPERATURE`
  - `PAPER_SECTION_WRITER_OPENAI_TIMEOUT`

- 引用需求审阅（`paper_citation_need`）：
  - `PAPER_CITATION_NEED_OPENAI_API_KEY`
  - `PAPER_CITATION_NEED_OPENAI_MODEL`
  - `PAPER_CITATION_NEED_OPENAI_BASE_URL`
  - `PAPER_CITATION_NEED_OPENAI_TEMPERATURE`
  - `PAPER_CITATION_NEED_OPENAI_TIMEOUT`

- 引用检索/候选选择（`paper_citation_retrieval`）：
  - `PAPER_CITATION_RETRIEVAL_OPENAI_API_KEY`
  - `PAPER_CITATION_RETRIEVAL_OPENAI_MODEL`
  - `PAPER_CITATION_RETRIEVAL_OPENAI_BASE_URL`
  - `PAPER_CITATION_RETRIEVAL_OPENAI_TEMPERATURE`
  - `PAPER_CITATION_RETRIEVAL_OPENAI_TIMEOUT`

- 事实性/claim 解析（`paper_claim_resolution`）：
  - `PAPER_CLAIM_RESOLUTION_OPENAI_API_KEY`
  - `PAPER_CLAIM_RESOLUTION_OPENAI_MODEL`
  - `PAPER_CLAIM_RESOLUTION_OPENAI_BASE_URL`
  - `PAPER_CLAIM_RESOLUTION_OPENAI_TEMPERATURE`
  - `PAPER_CLAIM_RESOLUTION_OPENAI_TIMEOUT`

- LaTeX 渲染（当 `latex_renderer_backend=llm` 或 Pandoc 不可用时）（`paper_latex_renderer`）：
  - `PAPER_LATEX_RENDERER_OPENAI_API_KEY`
  - `PAPER_LATEX_RENDERER_OPENAI_MODEL`
  - `PAPER_LATEX_RENDERER_OPENAI_BASE_URL`
  - `PAPER_LATEX_RENDERER_OPENAI_TEMPERATURE`
  - `PAPER_LATEX_RENDERER_OPENAI_TIMEOUT`

- LaTeX 修复（`paper_latex_fix`）：
  - `PAPER_LATEX_FIX_OPENAI_API_KEY`
  - `PAPER_LATEX_FIX_OPENAI_MODEL`
  - `PAPER_LATEX_FIX_OPENAI_BASE_URL`
  - `PAPER_LATEX_FIX_OPENAI_TEMPERATURE`
  - `PAPER_LATEX_FIX_OPENAI_TIMEOUT`

- Reviewer：理论（`paper_reviewer_theory`）
  - `PAPER_REVIEWER_THEORY_OPENAI_API_KEY`
  - `PAPER_REVIEWER_THEORY_OPENAI_MODEL`
  - `PAPER_REVIEWER_THEORY_OPENAI_BASE_URL`
  - `PAPER_REVIEWER_THEORY_OPENAI_TEMPERATURE`
  - `PAPER_REVIEWER_THEORY_OPENAI_TIMEOUT`

- Reviewer：实验（`paper_reviewer_experiment`）
  - `PAPER_REVIEWER_EXPERIMENT_OPENAI_API_KEY`
  - `PAPER_REVIEWER_EXPERIMENT_OPENAI_MODEL`
  - `PAPER_REVIEWER_EXPERIMENT_OPENAI_BASE_URL`
  - `PAPER_REVIEWER_EXPERIMENT_OPENAI_TEMPERATURE`
  - `PAPER_REVIEWER_EXPERIMENT_OPENAI_TIMEOUT`

- Reviewer：新颖性（`paper_reviewer_novelty`）
  - `PAPER_REVIEWER_NOVELTY_OPENAI_API_KEY`
  - `PAPER_REVIEWER_NOVELTY_OPENAI_MODEL`
  - `PAPER_REVIEWER_NOVELTY_OPENAI_BASE_URL`
  - `PAPER_REVIEWER_NOVELTY_OPENAI_TEMPERATURE`
  - `PAPER_REVIEWER_NOVELTY_OPENAI_TIMEOUT`

- Reviewer：实用性（`paper_reviewer_practical`）
  - `PAPER_REVIEWER_PRACTICAL_OPENAI_API_KEY`
  - `PAPER_REVIEWER_PRACTICAL_OPENAI_MODEL`
  - `PAPER_REVIEWER_PRACTICAL_OPENAI_BASE_URL`
  - `PAPER_REVIEWER_PRACTICAL_OPENAI_TEMPERATURE`
  - `PAPER_REVIEWER_PRACTICAL_OPENAI_TIMEOUT`

Embedding（当 `skip_novelty_check: false` 且需要构建向量库时）：
- `PAPER_EMBEDDING_OPENAI_API_KEY`：Embedding API Key（可与聊天模型不同）。
- `PAPER_EMBEDDING_OPENAI_BASE_URL`：Embedding 的 base URL（OpenAI-compatible）。
- `PAPER_EMBEDDING_OPENAI_EMBEDDING_MODEL`：Embedding 模型名（默认 `text-embedding-3-small`）。
- `PAPER_EMBEDDING_OPENAI_TIMEOUT`：Embedding 请求超时秒数。

### 3.2 `config_paper.yaml`（工作流配置）

CLI 会在你不传 `--config` 时自动加载项目根目录的 `config_paper.yaml`。

该文件的结构是：

```yaml
paper_workflow:
  key: value
```

下面逐项说明 `config_paper.yaml` 里每个参数的含义。

---

## 4. 参数参考：`config_paper.yaml`

> 说明：这里的 `temperature`/`max_patterns` 等影响的是“从 KG 采样套路”的策略；LLM 的温度在 `.env` 的 `OPENAI_TEMPERATURE`。

### 4.1 基本与日志

- `seed`：随机种子；`null` 表示每次随机，填整数可复现。
- `log_llm_outputs`：是否把 LLM 原始输出写入日志（调试用，可能很长）。
- `log_llm_output_max_chars`：LLM 日志截断长度；0 表示不截断。
- `step_log_enable`：是否额外输出“步骤摘要日志文件”。
- `step_log_path`：步骤摘要日志路径；为空会自动生成到 `outputs/logs/`。

### 4.2 语言与提示词

- `paper_language`：`en`/`zh`，用于选择默认提示词目录（`prompts_paper_en/` 或 `prompts_paper/`）以及模板语言。
- `paper_prompt_dir`：显式指定提示词目录（覆盖 `paper_language` 自动选择）。
- `enforce_english`：是否强制英文输出（会做中文比例校验）。
- `max_cjk_ratio`：允许的中日韩字符比例上限（例如 0.01 表示最多约 1%）。

### 4.3 新颖性（查重）与套路选择

- `skip_novelty_check`：跳过 embedding 构建与 novelty 检索/判定（端到端跑通/离线测试常用）。
- `novelty_threshold`：新颖性阈值；相似度高于阈值会触发“降权+重新采样套路”。
- `force_novelty_pass`：强制新颖性通过（即使相似度高也继续），用于测试/避免循环。
- `retriever_top_k`：向量检索返回 top_k；也用作引用检索候选上限（`CitationRetrievalAgent.max_candidates`）。
- `penalty`：当 novelty 命中时，对导致相似的套路分数做降权的幅度（越大降权越明显）。
- `max_plan_retry`：新颖性不通过时，最多允许“降权重采样”的次数；超过后会给 warning 并继续写稿。
- `max_patterns`：从 KG 里选择的套路（patterns）数量上限。
- `temperature`：套路选择的采样温度（越大越发散，越小越偏向高分套路）。

### 4.4 写稿与章节生成

- `template_path`：章节模板文件路径（如 `paper_template_en.yaml`），定义了 section_id/目标/字数/依赖等。
- `enable_related_work`：是否启用 Related Work 章节（如果模板中有该章节，会影响是否写它）。
- `writing_guide_limit_chars`：给写作 agent 的“写作指南/套路提示”最大字符数（避免 prompt 过长）。
- `writing_guide_max_guides`：最多注入多少条 guide。
- `section_max_retries`：每个章节写作/修复的最大重试次数。
- `section_summary_max_chars`：章节摘要的最大长度（用于后续章节的上下文输入）。

### 4.5 评审与回退循环

- `format_only_mode`：格式优先模式；跳过 reviewers 与内容质量自检/回退（但 LaTeX 编译闸门仍可能执行）。
- `pass_threshold`：多 reviewer 聚合后的通过阈值（avg_score >= pass_threshold 视为通过）。
- `require_individual_threshold`：是否要求每个 reviewer 的单独评分也达标（见 reviewer 配置的 threshold）。
- `max_iterations`：评审-回退的最大轮数；到上限仍未通过则失败。
- `max_rewrite_retry`：当主问题为 CLARITY/SOUNDNESS/SIGNIFICANCE 时，最多允许触发“改稿回退”的次数。
- `max_global_retry`：当主问题为 NOVELTY_LOW 时，最多允许回退到“重新 plan_from_kg”的次数。

### 4.6 输出格式与 Markdown 质量闸门

- `output_format`：`markdown` 或 `latex`（当前你用的是 `latex`）。
- `markdown_writer_backend`：`markdown`（直接写 Markdown）或 `spec`（结构化 spec → 渲染为 Markdown/LaTeX）。
- `spec_cite_token`：spec 写作中引用占位 token（默认 `[[CITE]]`）。
- `spec_render_require_cite_ids`：spec 渲染时是否强制 cite marker 带 id（便于后续引用链路定位）。
- `strict_markdown_validation`：是否启用 Markdown 硬校验（质量优先模式常开）。
- `section_min_word_ratio`：章节最低字数比例（相对模板的 `target_words_min`）。
- `section_min_quality_max_retries`：章节质量不足时的额外重试次数。
- `coverage_enable`：是否启用“章节覆盖验收器”（检查 required_points 覆盖情况）。
- `coverage_validator_backend`：覆盖验收器后端：`llm`/`rule`/`off`。
- `coverage_validator_max_retries`：覆盖验收失败的最大重试次数。
- `coverage_validator_require_evidence`：是否要求给出证据（更严格，通常更慢）。
- `quality_fail_fast`：质量优先失败即中止（避免长时间循环）。

### 4.7 LaTeX 渲染、模板与导出（Overleaf）

- `latex_template_preset`：导出 main.tex 的预设模板：`article_ctex` / `colm2024_conference`。
- `latex_titlepage_style`：当 `latex_template_preset=colm2024_conference` 时的第一页样式：`qwen` / `agentalpha`。
- `latex_template_assets_dir`：COLM 模板资源目录（会被复制到导出工程根目录；需包含 `colm2024_conference.sty/.bst` 和 `logo/`）。
- `latex_title`：导出 `main.tex` 的 `\\title{}` 内容（空则用占位符）。
- `latex_author`：导出 `main.tex` 的 `\\author{}` 内容（空则用占位符）。
- `latex_hflink` / `latex_mslink` / `latex_ghlink`：首页三行链接表格中的 URL 覆盖。
- `latex_output_dir`：导出工程输出目录（默认 `./outputs/overleaf`）。

### 4.8 LaTeX 工具链（本地编译闸门）

- `latex_engine`：编译引擎（传给 latexmk），常用 `xelatex`。
- `latex_require_tools`：是否强制要求本机安装 `latexmk/chktex/lacheck`；否则会报错。
- `latex_use_chktex`：是否运行 chktex 静态检查。
- `latex_use_lacheck`：是否运行 lacheck 静态检查。
- `latex_section_compile`：是否对单章节做最小编译（用于定位章节级错误）。
- `latex_final_compile`：是否对最终 Overleaf 工程执行编译验证（latexmk）。
- `latex_compile_timeout_sec`：单次编译超时时间（秒）。
- `latex_max_fix_retries`：LaTeX 修复最大重试次数（章节/最终稿修复循环）。
- `latex_format_max_retries`：仅 LaTeX 编译失败时的“自动修复+重编译”最大轮数。
- `latex_force_todo_cite`：是否把引用 key 强制规范化为 `TODO_` 前缀（防止无效 bibkey 导致编译失败）。
- `latex_generation_mode`：`per_section`（逐章生成 LaTeX）/ `final_only`（最终统一转换/修复）。
- `latex_renderer_backend`：`pandoc`（推荐）/ `llm`（Pandoc 不可用时的备选）。
- `pandoc_require`：Pandoc 缺失时是否直接失败（否则可能回退到 LLM 渲染）。
- `pandoc_bin`：Pandoc 可执行文件名或路径（也可用环境变量 `PANDOC_BIN` 覆盖）。
- `pandoc_consistency_threshold`：Pandoc 渲染一致性阈值（内容丢失检测）。
- `pandoc_fail_on_content_loss`：当检测到内容丢失时是否直接失败。

### 4.9 引用链路（Citations pipeline）

- `citations_enable`：是否启用引用链路（抽取引用需求 → 检索候选 → 选择条目 → 写入 refs.bib）。
- `citations_require_verifiable`：是否要求引用可验证（更严格，可能更慢）。
- `citations_require_api_key`：是否要求引用提供方必须有 API Key（例如 Semantic Scholar）。
- `citations_provider`：引用检索提供方：`mock`（离线/演示）或 `semantic_scholar` 等。
- `citations_year_range`：引用年份范围过滤（字符串，例如 `"2018-2026"`）。
- `citations_max_rounds`：引用检索的最大轮次（避免无限循环）。

---

## 5. 运行与测试命令

### 5.1 端到端（推荐）

```bash
python -m api.paper_cli workflow "我想写一篇关于代码库检索与打包的论文" --config config_paper.yaml --snapshot-path knowledge_graph.json
```

要点：
- `--config` 可省略（会自动加载根目录 `config_paper.yaml`）。
- `--snapshot-path` 可省略：若根目录存在 `knowledge_graph.json` 会优先用它；否则默认用 `paper_kg_snapshot.json`（若不存在会自动构建 mock 快照）。

### 5.2 Web UI（可选）

```bash
uvicorn api.web_server:app --reload --port 8000
```

浏览器打开：`http://127.0.0.1:8000`

### 5.3 单元测试

```bash
python -m pytest -q
```

---

## 6. 输出目录说明

默认输出：
- 日志：`outputs/logs/`
- Overleaf 工程：`outputs/overleaf/<run_id>/`
  - `overleaf.zip`：可直接上传 Overleaf
  - `main.tex` / `sections/*.tex` / `refs.bib`：工程源码

---

## 7. 常见问题

### 7.1 `inconsolata.sty not found`

COLM 模板依赖该字体包；在 Ubuntu/WSL 下通常安装即可：

```bash
sudo apt-get install -y texlive-fonts-extra
```

### 7.2 引用链路报错 / API 限制

如果你不需要真实引用，可在 `config_paper.yaml` 里设置：
- `citations_provider: mock`
- 或直接 `citations_enable: false`

---

## 8. 相关文档

- `PAPER_IDEA_TO_OVERLEAF.md`：更详细的链路说明与排障建议
