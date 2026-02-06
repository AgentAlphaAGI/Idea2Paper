# 开发者指南：端到端重建 Idea2Paper（KG → 索引 → Pipeline）

本文件面向希望在本仓库里**从零重建全链路产物**的开发者，包括：

- 知识图谱产物（`Paper-KG-Pipeline/output/*`）
- 本地离线索引（novelty / recall）
- 可选的 Path2「canonical subdomain taxonomy」（二级领域标签规范化）
- 运行完整 pipeline（`idea2story_pipeline.py`）并检查日志/结果

目标是：**可执行、可复现、可排查**，且不涉及改动任何业务算法逻辑。

---

## 0) 仓库结构（重要目录与职责）

- 仓库根目录：
  - `.env` / `.env.example`：密钥 + 运行时配置（优先级仅次于 shell `export`）
  - `i2p_config.json` / `i2p_config.example.json`：非敏感配置（优先级低于 env）
  - `log/`：每次运行的结构化日志（`log/run_.../`）
  - `results/`：每次运行的结果打包（`results/run_.../`）
  - `frontend/`：可选的本地 Web UI
- `Paper-KG-Pipeline/`：
  - `src/idea2paper/`：**核心库**（config/infra/recall/novelty/pipeline 模块）
  - `scripts/idea2story_pipeline.py`：**主 CLI 入口**
  - `scripts/tools/`：离线工具脚本（构建索引、taxonomy 等）
  - `output/`：KG + 索引 + 最近一次 pipeline 输出（本仓库中已包含一份可用数据）
  - `docs/`：内部说明（可能滞后于代码；以代码 + 根目录 README 为准）

### 0.1 关键入口与“运行时真正依赖的产物”

**运行 pipeline 的入口（多数开发者用这个）：**
- `Paper-KG-Pipeline/scripts/idea2story_pipeline.py`

**运行时读取的 KG 核心产物：**
- `Paper-KG-Pipeline/output/nodes_idea.json`
- `Paper-KG-Pipeline/output/nodes_pattern.json`
- `Paper-KG-Pipeline/output/nodes_domain.json`
- `Paper-KG-Pipeline/output/nodes_paper.json`
- `Paper-KG-Pipeline/output/knowledge_graph_v2.gpickle`

**可选的 Pattern 增强产物（pipeline 会自动探测并使用）：**
- `Paper-KG-Pipeline/output/patterns_structured.json`
  - 若存在，`idea2story_pipeline.py` 会将其与召回结果合并，用于增强 Pattern 的 prompt 字段（例如更丰富的 skeleton 示例 / tricks）。
  - 若缺失，pipeline **仍可运行**，会退回只使用 `nodes_pattern.json`（质量可能有差异）。
- `Paper-KG-Pipeline/output/paper_to_pattern.json`（主要供 legacy 工具使用；运行时 pipeline 不强依赖）
- `Paper-KG-Pipeline/output/patterns_statistics.json` / `Paper-KG-Pipeline/output/patterns_guide.txt`（给人看的统计/指南；运行时不需要）

**离线索引产物（可选但强烈建议，能提速/稳定）：**
- `Paper-KG-Pipeline/output/novelty_index*/`
- `Paper-KG-Pipeline/output/recall_index*/`

---

## 1) 你要重建到什么程度（按需求选）

### Level A —「只跑 pipeline」（最快）
直接复用仓库里已有的 `Paper-KG-Pipeline/output/`（KG + 图谱 + 部分索引）。

你只需要：
- 安装 Python 依赖
- 配好 `.env` 里的 API Key

然后运行：
```bash
python Paper-KG-Pipeline/scripts/idea2story_pipeline.py "your idea"
```

### Level B —「只重建索引」（最推荐：切 embedding / 索引异常时）
重建：
- novelty index（`output/novelty_index__.../`）
- recall 离线索引（`output/recall_index__.../`）
- 可选 subdomain taxonomy（`output/subdomain_taxonomy.json`）

典型场景：你切换了 **embedding provider/url/model**，不想让旧索引污染效果。

### Level C —「重建 KG + 索引 + 跑 pipeline」（全量）
重建：
- nodes（`nodes_*.json`）
- graph（`knowledge_graph_v2.gpickle` / `edges.json`）
- 索引
- taxonomy（可选）
- 跑 pipeline

这要求你具备 ICLR_25 的原始数据文件（本仓库未必包含；见第 4 节）。

---

## 2) 环境准备

### 2.1 Python
建议：
- Python 3.10+（推荐 3.11）

### 2.1.1 跨平台注意事项（避免命令踩坑）
- Linux/macOS 虚拟环境激活：`source .venv/bin/activate`
- Windows PowerShell 激活：`.\.venv\Scripts\Activate.ps1`
- 如遇到脚本没有执行权限：统一用 `python path/to/script.py`（本文所有命令均可这样执行）

### 2.2 安装依赖
在仓库根目录：
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r Paper-KG-Pipeline/requirements.txt
```

如果你要跑前端（`frontend/`），也使用同一套 Python 环境即可。

---

## 3) 配置（env + i2p_config）

### 3.1 配置优先级（非常重要）
项目读取配置的优先级：

1) shell `export ...`
2) `.env`（仓库根目录）
3) `i2p_config.json`
4) 代码默认值

所以你可以用 `export` 临时覆盖任意配置而不改文件。

**布尔值注意（与项目实现一致）：**在 env 里，只有字符串 `"1"` 才算 true，其他都算 false。建议用 `1/0`，不要用 `true/false`。

### 3.2 创建本地配置文件
```bash
cp .env.example .env
cp i2p_config.example.json i2p_config.json
```

不要提交 `.env`。

### 3.3 必需密钥（最低要求）
最少需要：

- `SILICONFLOW_API_KEY`（当前代码里作为 LLM key，同时也是 embedding key 的 fallback）
  - LLM：story generation / critic / refinement / verifier 等
  - Embedding：recall rerank、离线索引构建、novelty、taxonomy 构建等

在 `.env`：
```bash
SILICONFLOW_API_KEY=YOUR_KEY
```

如果你使用单独的 embedding key：
```bash
EMBEDDING_API_KEY=YOUR_EMBEDDING_KEY
```

### 3.4 LLM endpoint 与模型
当前 LLM 传输层使用 **OpenAI Chat Completions 兼容**协议。

在 `.env`：
```bash
LLM_API_URL=https://api.siliconflow.cn/v1/chat/completions
LLM_MODEL=Pro/zai-org/GLM-4.7
```

如果你使用 OpenAI-compatible 平台（例如 DeepSeek 或其他网关），只要它支持 chat completions 协议，设置对应 `LLM_API_URL`（指向 chat completions）+ `LLM_MODEL` 即可。

### 3.5 Embedding endpoint 与模型
在 `.env`：
```bash
EMBEDDING_API_URL=https://api.siliconflow.cn/v1/embeddings
EMBEDDING_MODEL=Qwen/Qwen3-Embedding-8B
```

核心规则：**embedding provider/url/model 改了，就应该重建/复用对应的索引目录**（否则可能出现 mismatch 或效果漂移）。

为了避免覆盖旧索引，仓库支持自动 profile 目录：
```bash
I2P_INDEX_DIR_MODE=auto_profile
```
启用后 novelty/recall 索引会按 embedding 配置自动落到不同目录（避免“切模型就来回重建”）。

### 3.6 Embedding 维度（实用约束）
本仓库当前默认配置（以及部分 legacy 降级逻辑）基于 **4096 维 embedding**（与默认 `Qwen/Qwen3-Embedding-8B` 一致）。

实用规则（与根 README 的提示一致）：
- 使用输出 **4096 维向量**的 embedding 模型；
- 并确保以下两类阶段使用的 embedding 模型**一致**：
  - 离线索引构建（novelty/recall/taxonomy）
  - 运行时 pipeline

如果怀疑维度或模型不一致：
- 查看 `log/run_.../embedding_calls.jsonl`，确认 embedding 向量长度一致。

---

## 4) 重建知识图谱（Level C）

### 4.1 输入数据（ICLR_25）
ICLR 的 KG 构建脚本读取：
`Paper-KG-Pipeline/data/ICLR_25/`

- `assignments.jsonl`
- `cluster_library_sorted.jsonl`
- `iclr_patterns_full.jsonl`
-（可选）`paper_reviews_dataset_iclr_reviews_filtered.jsonl`

如果你没有这些文件，就无法完整重建 KG nodes/edges；此时建议直接复用仓库已有的 `Paper-KG-Pipeline/output/`。

### 4.1.1 上游数据抽取（可选 / legacy）
如果你希望从原始 PDF / review 数据开始重建 `Paper-KG-Pipeline/data/...`：

- `Paper-KG-Pipeline/scripts/tools/` 下存在一些 legacy 抽取脚本（例如 `extract_paper_review.py`）。
- 其中部分脚本使用更旧的环境变量命名（例如 `LLM_AUTH_TOKEN`），可能与当前 `.env.example` 不完全一致。

务实建议：
- 把 **ICLR_25 JSONL**（或仓库预置的 `Paper-KG-Pipeline/output/`）作为“可复现重建”的 source of truth。
- 如必须跑 legacy 抽取，建议在独立分支/独立 workspace 进行，并记录你产出的输入/输出文件版本。

### 4.2 重建 nodes（实体节点）
命令：
```bash
python Paper-KG-Pipeline/scripts/build_entity_v3.py
```

期望输出（写入 `Paper-KG-Pipeline/output/`）：
- `nodes_idea.json`
- `nodes_pattern.json`
- `nodes_domain.json`
- `nodes_paper.json`
- `nodes_review.json`（如果存在 review 数据）
- `knowledge_graph_stats.json`

说明：
- 该步骤可能会调用 LLM 生成 pattern 的 `llm_enhanced_summary`，因此需要可用的 LLM API。
- “同输入可复现”取决于 LLM 输出稳定性（温度/模型/平台都会影响）。

### 4.3 重建 edges + graph
命令：
```bash
python Paper-KG-Pipeline/scripts/build_edges.py
```

期望输出：
- `Paper-KG-Pipeline/output/edges.json`
- `Paper-KG-Pipeline/output/knowledge_graph_v2.gpickle`

### 4.4 KG 快速自检（建议）
```bash
python - <<'PY'
import json, pickle
from pathlib import Path
p = Path("Paper-KG-Pipeline/output")
for name in ["nodes_pattern.json","nodes_domain.json","nodes_idea.json","nodes_paper.json"]:
    data = json.loads((p/name).read_text(encoding="utf-8"))
    print(name, len(data))
G = pickle.loads((p/"knowledge_graph_v2.gpickle").read_bytes())
print("graph nodes:", G.number_of_nodes(), "edges:", G.number_of_edges())
PY
```

### 4.5 备用 KG 构建器（非 ICLR / 多会议 legacy 路径）
仓库里还有一套更旧的“多会议”构图流程（不是当前预置 ICLR 数据默认走的路径）：

- 节点构建：`Paper-KG-Pipeline/scripts/tools/build_entity.py`
  - 输入依赖：
    - `Paper-KG-Pipeline/data/<CONFERENCE>/*_paper_node.json`（或 `_all_paper_nodes.json`）
    - `Paper-KG-Pipeline/output/patterns_structured.json`
    - `Paper-KG-Pipeline/output/paper_to_pattern.json`
  - 输出：
    - `Paper-KG-Pipeline/output/nodes_idea.json`
    - `Paper-KG-Pipeline/output/nodes_pattern.json`
    - `Paper-KG-Pipeline/output/nodes_domain.json`
    - `Paper-KG-Pipeline/output/nodes_paper.json`

这条路线适合你没有 ICLR_25 JSONL，但有自己抽取的 conference 数据；它要求你已经：
1) 把 paper nodes 抽取到 `Paper-KG-Pipeline/data/<CONFERENCE>/...`
2) 把 pattern 聚类结果写到 `Paper-KG-Pipeline/output/patterns_structured.json` + `paper_to_pattern.json`

如果你没有这些输入，建议走 ICLR_25 路径（第 4.1 节）或直接用仓库已有的 `Paper-KG-Pipeline/output/`。

### 4.6（可选但推荐）重建 `patterns_structured.json`（让 Pattern prompt 更“厚”）
如果你想做“真正从零”的重建，并且你清空/覆盖了 `Paper-KG-Pipeline/output/`，你可能会丢失：
- `patterns_structured.json`
- `paper_to_pattern.json`
- `patterns_statistics.json`
- `patterns_guide.txt`

这些文件 **不是** `build_entity_v3.py` 生成的，但运行时 pipeline 会在检测到它们时（尤其是 `patterns_structured.json`）用于增强 Pattern 信息。

生成它们的脚本是：
```bash
python Paper-KG-Pipeline/scripts/tools/generate_patterns.py
```

注意：该脚本本身有明确前置条件（来自代码实现）：
- 需要存在带 `skeleton` + `tricks` 的 paper nodes，路径形如：
  - `Paper-KG-Pipeline/data/<CONFERENCE>/*_paper_node.json`
- 这些输入**默认不在本仓库中提供**；你需要自行生成（legacy extraction），或者保留仓库预置的 `output/patterns_structured.json`。
- 它使用了一组 legacy 的环境变量名：
  - `EMBED_API_URL` / `EMBED_MODEL`（不是 `EMBEDDING_API_URL` / `EMBEDDING_MODEL`）
  - `SILICONFLOW_API_KEY` 作为鉴权 token

如果你不想改脚本代码，可以用 env 做一次“桥接”：
```bash
export EMBED_API_URL="$EMBEDDING_API_URL"
export EMBED_MODEL="$EMBEDDING_MODEL"
python Paper-KG-Pipeline/scripts/tools/generate_patterns.py
```

如果你没有 `data/<CONFERENCE>/*_paper_node.json` 这些输入，建议**跳过此步骤**；pipeline 仍会用 `nodes_pattern.json` 正常跑通。

---

## 5) 重建索引（Level B/C）

### 5.1 Novelty index（本地查重/相似度）
构建：
```bash
python Paper-KG-Pipeline/scripts/tools/build_novelty_index.py
```

输出位置：
- 开启 `I2P_INDEX_DIR_MODE=auto_profile` 时：`Paper-KG-Pipeline/output/novelty_index__<PROFILE_ID>/...`
- manual 模式：`Paper-KG-Pipeline/output/novelty_index`

依赖：
- `Paper-KG-Pipeline/output/nodes_paper.json`
- `EMBEDDING_API_URL / EMBEDDING_MODEL / EMBEDDING_API_KEY`

### 5.2 Recall offline index（可选但强烈建议：提速）
如果 `I2P_RECALL_USE_OFFLINE_INDEX=1`，pipeline 会优先使用离线 recall index。

构建：
```bash
python Paper-KG-Pipeline/scripts/tools/build_recall_index.py
```

输出位置：
- 开启 `auto_profile` 时：`Paper-KG-Pipeline/output/recall_index__<PROFILE_ID>/...`

依赖：
- `Paper-KG-Pipeline/output/nodes_idea.json`
- `Paper-KG-Pipeline/output/nodes_paper.json`
- embedding 配置

### 5.3 Canonical subdomain taxonomy（可选：改善 Path2 的 subdomain 信号）
该 taxonomy 主要解决：
- subdomain 同义/近义重复
- “泛化标签”造成的超大桶（stoplist）
- 尾部极小桶的受约束合并（减少长尾）

离线构建一次：
```bash
python Paper-KG-Pipeline/scripts/tools/build_subdomain_taxonomy.py --output Paper-KG-Pipeline/output/subdomain_taxonomy.json
```

运行时启用：
```bash
I2P_SUBDOMAIN_TAXONOMY_ENABLE=1
I2P_SUBDOMAIN_TAXONOMY_PATH=Paper-KG-Pipeline/output/subdomain_taxonomy.json
I2P_SUBDOMAIN_TAXONOMY_STOPLIST_MODE=drop
```

如果开启了但文件缺失/格式不对，运行时应打印 warning 并安全回退到原始 subdomain。

### 5.4 是否必须手动构建？
- **novelty/recall index**：如果 `I2P_INDEX_AUTO_PREPARE=1` 且 `I2P_INDEX_ALLOW_BUILD=1`，pipeline 会在运行时尝试自动构建缺失索引。
- **subdomain taxonomy**：刻意设计为离线构建（质量优先、可复用、可对比 diff）。开启但缺失时运行时会回退，不会把流程打崩。

### 5.5 强制重建 / 清理状态（调试时常用）
你可能需要强制重建的原因：
- 切换了 embedding 配置
- 上一次构建中断导致半成品
- 想验证可复现性

推荐做法：

- novelty index：
  - 支持 `--force-rebuild` 与 `--resume`：
    ```bash
    python Paper-KG-Pipeline/scripts/tools/build_novelty_index.py --force-rebuild
    ```
- recall index：
  - 支持 `--force-rebuild`（以 `--help` 为准）：
    ```bash
    python Paper-KG-Pipeline/scripts/tools/build_recall_index.py --force-rebuild
    ```
- taxonomy：
  - 直接重跑（覆盖输出）：
    ```bash
    python Paper-KG-Pipeline/scripts/tools/build_subdomain_taxonomy.py --output Paper-KG-Pipeline/output/subdomain_taxonomy.json
    ```

不建议手工乱删目录；优先用脚本提供的 `--force-rebuild`。

---

## 6) 运行完整 pipeline（Idea → Story）

### 6.1 一键运行（推荐）
```bash
python Paper-KG-Pipeline/scripts/idea2story_pipeline.py "我们研究音频-文本大模型的文本主导偏见..."
```

### 6.2 运行时大概会做什么
- 从 `Paper-KG-Pipeline/output/` 加载 KG 数据
- 运行三路召回（Path1/Path2/Path3）
- Pattern selection
- Story generation
- Multi-agent critic
- Refinement（按需进入循环）
- Novelty + final verification（若开启）
- 写出：
  - `Paper-KG-Pipeline/output/final_story.json`
  - `Paper-KG-Pipeline/output/pipeline_result.json`
  - `log/run_.../`（若开启）
  - `results/run_.../`（若开启打包）

### 6.3 为什么你可能看到“三路召回”跑了不止一次
某些功能会触发额外的 recall：
- Idea Packaging（pattern-guided 双轮 recall）
- Novelty mode（遍历候选 pattern 做多轮尝试）

如果你想减少 recall 次数便于 debug，可先关掉 Idea Packaging：
```bash
I2P_IDEA_PACKAGING_ENABLE=0
```

---

## 7) 输出、日志与排查

### 7.1 最新一次输出（覆盖写）
pipeline 会把最新输出写到：
- `Paper-KG-Pipeline/output/final_story.json`
- `Paper-KG-Pipeline/output/pipeline_result.json`

### 7.2 每次运行的结构化日志（强烈建议开启）
当 `I2P_ENABLE_LOGGING=1`，每次运行会写：
- `log/run_<timestamp>_<pid>_<hash>/events.jsonl`
- `log/run_<...>/llm_calls.jsonl`
- `log/run_<...>/embedding_calls.jsonl`
- `log/run_<...>/meta.json`

### 7.3 每次运行的结果打包（便于复现与分享）
当 `I2P_RESULTS_ENABLE=1`，每次运行会创建：
- `results/run_<...>/final_story.json`
- `results/run_<...>/pipeline_result.json`
- `results/run_<...>/manifest.json`
- `results/run_<...>/run_log/`（复制日志目录；不会用软链接）

---

## 8) 性能与稳定性开关（实践默认值）

### 8.1 关闭 adaptive densify（更快 + 更少 JSON 出错）
adaptive densify 会触发第二轮 anchored critic，时延显著增加。

建议：
```bash
I2P_ANCHOR_DENSIFY_ENABLE=0
```

### 8.2 strict JSON vs “继续跑”
如果你反复遇到 “Critic JSON invalid after retries”，可以临时关闭 strict 模式：
```bash
I2P_CRITIC_STRICT_JSON=0
```
这主要用于开发/调试；质量跑建议保持 strict。

### 8.3 Index auto-prepare（自动预检/自动构建索引）
默认建议：
```bash
I2P_INDEX_AUTO_PREPARE=1
I2P_INDEX_ALLOW_BUILD=1
```

如果你希望索引缺失时直接失败（fail fast）：
```bash
I2P_INDEX_ALLOW_BUILD=0
```

### 8.4 常见失败原因速查（排障优先级）
1) **没配 key / key 错了**：LLM/Embedding 调用失败或走 simulated。
2) **URL 配错**：
   - LLM 通常应指向 Chat Completions 兼容 URL（常见以 `/v1/chat/completions` 结尾）
   - Embedding 通常应指向 embeddings URL（常见以 `/v1/embeddings` 结尾）
3) **embedding 配置变了但复用了旧索引**：可能出现索引 mismatch、召回变差；应重建新 profile 的索引。
4) **critic strict JSON 开启 + 平台输出不稳定**：导致 “Critic JSON invalid after retries”。
5) **adaptive densify 开启**：Phase 3 变慢（第二轮 anchored critic）。

---

## 9) 前端（可选，本地 Web UI）

启动：
```bash
python frontend/server/app.py --host 127.0.0.1 --port 8080
```

访问：
```text
http://127.0.0.1:8080/
```

前端会：
- 以子进程方式启动 pipeline
- 显示高层阶段（不是 raw logs）
- 允许下载日志 zip

---

## 10) 开发者验证清单（强烈建议）

### 10.1 语法检查（最低成本）
```bash
python -m py_compile \
  Paper-KG-Pipeline/src/idea2paper/config.py \
  Paper-KG-Pipeline/src/idea2paper/recall/recall_system.py \
  Paper-KG-Pipeline/scripts/idea2story_pipeline.py \
  Paper-KG-Pipeline/scripts/tools/build_novelty_index.py \
  Paper-KG-Pipeline/scripts/tools/build_recall_index.py \
  Paper-KG-Pipeline/scripts/tools/build_subdomain_taxonomy.py
```

### 10.2 离线-ish smoke（没有 key 也能验证“流程不会崩”）
如果你暂时没有 key，也可以验证流程 wiring：
- 关闭 strict JSON
- 关闭 novelty/verification
- 允许 LLM 走 simulated mode

示例：
```bash
I2P_CRITIC_STRICT_JSON=0 I2P_VERIFICATION_ENABLE=0 I2P_NOVELTY_ENABLE=0 \
python Paper-KG-Pipeline/scripts/idea2story_pipeline.py "test idea"
```

---

## 11) 代码入口（source of truth）
本仓库更新频繁；以 **代码 + 根目录 README** 为准。

当你在排查“为什么重建/运行不一致”时，最有用的入口文件：
- 配置加载与优先级：`Paper-KG-Pipeline/src/idea2paper/config.py`
- pipeline CLI 入口：`Paper-KG-Pipeline/scripts/idea2story_pipeline.py`
- 召回系统（Path1/Path2/Path3）：`Paper-KG-Pipeline/src/idea2paper/recall/recall_system.py`
- 结构化日志：`Paper-KG-Pipeline/src/idea2paper/infra/run_logger.py`
- results 打包：`Paper-KG-Pipeline/src/idea2paper/infra/result_bundler.py`
