# aisv2_job_standalone

这是对AI-Scientist-v2项目的独立、可移动实验任务模块的封装。
https://github.com/SakanaAI/AI-Scientist-v2

注意：模型能力的原因可能导致奇怪的toolcall schema错误
可以使用qwen-plus + qwen3-vl-plus测试一下链路
## 功能说明（仅实验管线）

它会做这些事：
- 从 ideas.json 里选择 `idea_idx`
- 生成 `idea.md` 与 `idea.json`
- 为本次运行准备 `bfts_config.yaml`
- 运行 BFTS 实验（stage 1-4）
- 写出 summaries：draft / baseline / research / ablation
- 运行 `aggregate_plots` 生成图片
- 写出 token tracker 文件

它**不会**做：
- 不做 writeup/review
- 不执行任何全局进程清理

## 安装

在本目录下执行：

```bash
pip install -r requirements.txt
```

硬件与环境要求与上游项目一致（GPU、CUDA、模型 API Key、以及 LaTeX/PDF 等系统工具）。

## 一键运行（使用配置文件）

1) 准备环境变量（模型 API Key 等）：把 `.env.example` 复制为 `.env` 并按需填写，或自行在 shell / Docker /
Kubernetes Secrets 中设置环境变量。

2) 从项目根目录一条命令运行（显式使用 config）：

```bash
(cd experiment-agent && python -m aisv2_job run --config ./job_config.yaml)
```

说明：
- 默认情况下（在本目录内运行），不传 `--config` 也会自动读取 `job_config.yaml`。
- 你通常只需要改两个文件：
  - `job_config.yaml`：选择 idea 文件、idea_idx、输出目录、以及要用的 `bfts_config.yaml`。
  - `bfts_config.yaml`：控制 stages/模型路由/迭代次数等。

## 运行（CLI）

```bash
python -m aisv2_job run \
  --load_ideas ./ai_scientist/ideas/i_cant_believe_its_not_better.json \
  --idea_idx 0 \
  --attempt_id 0 \
  --out_root ./runs \
  --model_agg_plots o3-mini-2025-01-31 \
  --plot_agg_reflections 5 \
  --plot_agg_max_figures 12 \
  --plot_agg_max_tokens 12000 \
  --keep_experiment_results
```

CLI 会在 stdout 打印 JSON manifest，并在
`./runs/experiments/<timestamp>_<IdeaName>_attempt_<id>/` 内写入 `run_manifest.json`。

## 配置说明

默认情况下，CLI 会读取本目录下的 `job_config.yaml`。你也可以用 `--config` 覆盖它。优先级顺序为：

CLI 参数 > job_config.yaml > 代码默认值

`job_config.yaml` 中的路径会以“本目录”为基准解析；CLI 里传入的路径按当前工作目录解析。

补充：`aggregate_plots`（生成 `auto_plot_aggregator.py` + `figures/`）相关的关键配置项：
- `job.plot_agg_max_tokens`：控制聚合画图阶段 LLM 的最大输出 token（太小会导致脚本被截断 -> SyntaxError）。
- `job.plot_agg_reflections` / `job.plot_agg_max_figures`：控制反思轮数与目标图数量。
- `job.keep_experiment_results`：保留 `experiment_results/`，方便你后续手动重跑聚合画图（避免“空图”）。

示例（只用 config，不传 CLI 参数）：

```bash
python -m aisv2_job run
```

示例（覆盖部分配置）：

```bash
python -m aisv2_job run --idea_idx 1 --out_root ./runs_alt
```

环境变量清单见 `.env.example`，按需复制到你的 shell / Docker env / Kubernetes secrets。

## OpenAI 兼容的模型路由（OpenAI-Compatible Model Routing）

所有模型调用都通过 OpenAI 兼容的 Chat Completions 接口完成。模型字符串支持：

- `<model_id>`：使用默认 provider（`OPENAI_API_KEY` / `OPENAI_BASE_URL`）
- `<provider>::<model_id>`：使用 provider 专属的 `OAIC_*` 环境变量
- `ollama/<model_id>`：等价于 `ollama::<model_id>`

Provider 环境变量：

- Default：`OPENAI_API_KEY`，`OPENAI_BASE_URL`（可选）
- Custom：`OAIC_<PROVIDER>_API_KEY`，`OAIC_<PROVIDER>_BASE_URL`
- Ollama：`OAIC_OLLAMA_BASE_URL`（默认 `http://localhost:11434/v1`）
- 可选开关：`OAIC_DISABLE_TOOLS`，`OAIC_DISABLE_VISION`，`OAIC_DEBUG`

兼容性降级（自动按顺序尝试）：

- 移除 `seed`、移除 `reasoning_effort`、在 `max_completion_tokens` 和 `max_tokens` 之间做兼容替换、
  移除 `response_format`、移除 `stop=None`
- tools -> functions -> 纯 JSON 响应（带 schema 约束）
- vision -> 纯文本（移除 image_url）

示例（单一 provider）：

```yaml
# bfts_config.yaml
agent:
  code:
    model: gpt-4o-mini-2024-07-18
  feedback:
    model: gpt-4o-2024-11-20
```

示例（不同 agent 使用不同 provider）：

```yaml
# bfts_config.yaml
agent:
  code:
    model: deepseek-ai/DeepSeek-V3
  feedback:
    model: meta-llama/llama-3.1-405b-instruct
```

## 输出目录结构（Output Contract）

运行目录（关键文件）：
- idea.md
- idea.json
- bfts_config.yaml
- logs/0-run/unified_tree_viz.html
- logs/0-run/draft_summary.json
- logs/0-run/baseline_summary.json
- logs/0-run/research_summary.json
- logs/0-run/ablation_summary.json
- figures/（来自 aggregate_plots）
- auto_plot_aggregator.py
- token_tracker.json
- token_tracker_interactions.json
- run_manifest.json

manifest 会包含所有产物的绝对路径；如果某些预期文件缺失，也会记录 warnings。你可以用 `--strict_artifacts`
让程序在 summaries 或 figures 缺失时直接失败。

## 只重跑最后的聚合画图（aggregate_plots）

当你想在不中断前面实验的情况下，只重跑最后一步聚合画图：

```bash
python -m ai_scientist.perform_plotting \
  --folder <idea_dir> \
  --model <your_llm_model> \
  --reflections 5 \
  --max_figures 12 \
  --max_tokens 12000
```

注意：
- `perform_plotting` 会优先使用 summaries 里记录的 `experiment_results/...` 相对路径；如果顶层 `experiment_results/` 不存在，会自动从 `logs/0-run/experiment_results` 重新创建（symlink 或 copy）。
- 会生成 `plot_agg_manifest.json`，里面包含可用的 `.npy` 绝对路径；聚合脚本应只从这里读取数据，避免相对路径基准错误。

## Python SDK

```python
from aisv2_job.sdk import run_job_subprocess

manifest = run_job_subprocess(
    load_ideas="./ai_scientist/ideas/i_cant_believe_its_not_better.json",
    idea_idx=0,
    out_root="./runs",
)
```

SDK 本质上只是调用 CLI 子进程，并返回解析后的 manifest。

## 安全提示

该系统会执行 LLM 生成的代码。建议在沙箱环境或容器中运行，并遵循许可证对 AI 使用披露的要求。

## 自检（不依赖 AI-Scientist-v2）

从仓库根目录执行：

```bash
cd experiment-agent
python -c "import ai_scientist, inspect; print(ai_scientist.__file__)"
python -m aisv2_job --help
cd ..
mv AI-Scientist-v2.bak AI-Scientist-v2
```

import path 应该指向 `experiment-agent/ai_scientist/__init__.py`。
