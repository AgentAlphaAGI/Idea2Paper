# Developer Guide: Rebuild Idea2Paper End-to-End (KG → Indexes → Pipeline)

This document is for developers who want to **rebuild everything from scratch** in this repo:

- Knowledge Graph artifacts (`Paper-KG-Pipeline/output/*`)
- Local offline indexes (novelty / recall)
- Optional Path2 “canonical subdomain taxonomy”
- Run the full pipeline (`idea2story_pipeline.py`) and inspect logs/results

It is written to be **actionable and reproducible**, without changing any business logic.

---

## 0) Repo map (what lives where)

- Repo root:
  - `.env` / `.env.example`: secrets + runtime config (highest priority after shell `export`)
  - `i2p_config.json` / `i2p_config.example.json`: non-secret config (lower priority than env)
  - `log/`: per-run structured logs (`log/run_.../`)
  - `results/`: per-run bundled artifacts (`results/run_.../`)
  - `frontend/`: optional local web UI
- `Paper-KG-Pipeline/`:
  - `src/idea2paper/`: **core library** (config, infra, recall, novelty, pipeline modules)
  - `scripts/idea2story_pipeline.py`: **main CLI entry**
  - `scripts/tools/`: offline tooling (build indexes, taxonomy, etc)
  - `output/`: all KG + indexes + latest pipeline outputs (tracked in git in this repo)
  - `docs/`: internal notes (may lag behind code; treat code + root README as source of truth)

### 0.1 Key entrypoints & “source of truth” artifacts

**Runtime pipeline entry (most users):**
- `Paper-KG-Pipeline/scripts/idea2story_pipeline.py`

**KG artifacts consumed by runtime:**
- `Paper-KG-Pipeline/output/nodes_idea.json`
- `Paper-KG-Pipeline/output/nodes_pattern.json`
- `Paper-KG-Pipeline/output/nodes_domain.json`
- `Paper-KG-Pipeline/output/nodes_paper.json`
- `Paper-KG-Pipeline/output/knowledge_graph_v2.gpickle`

**Optional pattern-enrichment artifacts (auto-detected by the pipeline):**
- `Paper-KG-Pipeline/output/patterns_structured.json`
  - When present, `idea2story_pipeline.py` will merge it into recalled patterns to enrich prompt fields (e.g., richer skeleton examples / tricks).
  - When missing, the pipeline **still runs** and falls back to `nodes_pattern.json` only (quality may differ).
- `Paper-KG-Pipeline/output/paper_to_pattern.json` (mainly used by legacy tools; not required by the runtime pipeline)
- `Paper-KG-Pipeline/output/patterns_statistics.json` / `Paper-KG-Pipeline/output/patterns_guide.txt` (human-facing; not required by runtime)

**Offline index artifacts (optional, but recommended):**
- `Paper-KG-Pipeline/output/novelty_index*/`
- `Paper-KG-Pipeline/output/recall_index*/`

---

## 1) What you can rebuild (choose your level)

### Level A — “Just run the pipeline” (fastest)
Use the prebuilt artifacts already in `Paper-KG-Pipeline/output/` (KG + graphs + some indexes).

You only need:
- Python deps installed
- API keys configured (`.env`)

Then run:
```bash
python Paper-KG-Pipeline/scripts/idea2story_pipeline.py "your idea"
```

### Level B — “Rebuild indexes only” (recommended for embedding/model changes)
Rebuild:
- Novelty index (`output/novelty_index__.../`)
- Recall offline index (`output/recall_index__.../`)
- Optional subdomain taxonomy (`output/subdomain_taxonomy.json`)

This is what you do when you change **embedding provider/url/model**.

### Level C — “Rebuild KG + indexes + pipeline” (full rebuild)
Rebuild:
- Nodes (`nodes_*.json`)
- Graph (`knowledge_graph_v2.gpickle` / `edges.json`)
- Indexes
- Optional taxonomy
- Run pipeline

This requires the **raw ICLR_25 data files** (not always committed in the repo; see Section 4).

---

## 2) Environment setup

### 2.1 Python
Recommended:
- Python 3.10+ (3.11 is a good default)

### 2.1.1 OS notes (so commands don’t surprise you)
- Linux/macOS activation: `source .venv/bin/activate`
- Windows PowerShell activation: `.\.venv\Scripts\Activate.ps1`
- If you see “permission denied” on scripts, run them via `python path/to/script.py` (as shown below).

### 2.2 Install dependencies
From repo root:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r Paper-KG-Pipeline/requirements.txt
```

If you also want the web UI, it uses the same Python environment.

---

## 3) Configuration (env + i2p_config)

### 3.1 Config priority (important)
The project reads configuration in this order:

1) Shell `export ...`
2) `.env` (repo root)
3) `i2p_config.json`
4) code defaults

So you can temporarily override anything using shell `export`.

**Boolean note (matches repo behavior):** in env vars, only the string `"1"` is treated as true; everything else is false.
So prefer `1/0` instead of `true/false`.

### 3.2 Create your local config files
```bash
cp .env.example .env
cp i2p_config.example.json i2p_config.json
```

Do not commit `.env`.

### 3.3 Required secrets
At minimum you need:

- `SILICONFLOW_API_KEY` (used as the LLM key + embedding key fallback in current code)
  - LLM: story generation, critic, refinement, verifier, etc.
  - Embedding: recall rerank, offline index build, novelty, etc.

In `.env`:
```bash
SILICONFLOW_API_KEY=YOUR_KEY
```

If you use a separate embedding key:
```bash
EMBEDDING_API_KEY=YOUR_EMBEDDING_KEY
```

### 3.4 LLM endpoint & model
Current LLM transport expects an **OpenAI Chat Completions compatible** endpoint.

In `.env`:
```bash
LLM_API_URL=https://api.siliconflow.cn/v1/chat/completions
LLM_MODEL=Pro/zai-org/GLM-4.7
```

If your provider is OpenAI-compatible (DeepSeek / other gateways), you can point `LLM_API_URL` to that gateway’s chat completions endpoint and set `LLM_MODEL` accordingly.

### 3.5 Embedding endpoint & model
In `.env`:
```bash
EMBEDDING_API_URL=https://api.siliconflow.cn/v1/embeddings
EMBEDDING_MODEL=Qwen/Qwen3-Embedding-8B
```

Key rule: **embedding model/provider/url changes require reusing/rebuilding the correct indexes**.

To make this painless, the repo supports auto profile dirs:
```bash
I2P_INDEX_DIR_MODE=auto_profile
```
This makes novelty/recall indexes live in separate directories per embedding config (so you don’t overwrite old ones).

### 3.6 Embedding dimension (practical requirement)
This repo’s current defaults (and some legacy fallbacks) assume **4096-dimensional embeddings** (same as the default `Qwen/Qwen3-Embedding-8B`).

Practical rule (matches the repo README guidance):
- Use an embedding model that outputs **4096-dim vectors**.
- Keep the embedding model **consistent** between:
  - offline index builds (novelty/recall/taxonomy)
  - runtime pipeline runs

If you suspect a mismatch:
- Inspect `log/run_.../embedding_calls.jsonl` and verify embedding vector lengths look consistent.

---

## 4) Knowledge Graph rebuild (Level C)

### 4.1 Inputs (ICLR_25 data files)
The ICLR KG builder uses these files under:
`Paper-KG-Pipeline/data/ICLR_25/`

- `assignments.jsonl`
- `cluster_library_sorted.jsonl`
- `iclr_patterns_full.jsonl`
- (optional) `paper_reviews_dataset_iclr_reviews_filtered.jsonl`

If you don’t have them, you can’t fully regenerate the KG nodes/edges; use the prebuilt `Paper-KG-Pipeline/output/` instead.

### 4.1.1 Upstream data extraction (optional / legacy)
If your goal is to rebuild `Paper-KG-Pipeline/data/...` from raw PDFs / reviews:

- There are legacy scripts under `Paper-KG-Pipeline/scripts/tools/` such as `extract_paper_review.py`.
- Some of these scripts use older environment variables (e.g. `LLM_AUTH_TOKEN`) and may not match the current `.env.example` conventions.

Practical recommendation:
- Treat the **ICLR_25 JSONL files** (or prebuilt `Paper-KG-Pipeline/output/`) as the “source of truth” for reproducible rebuilds.
- If you must run legacy extraction, do it in a separate branch/workspace and document the exact inputs/outputs you produced.

### 4.2 Rebuild nodes (entities)
Command:
```bash
python Paper-KG-Pipeline/scripts/build_entity_v3.py
```

Expected outputs (written to `Paper-KG-Pipeline/output/`):
- `nodes_idea.json`
- `nodes_pattern.json`
- `nodes_domain.json`
- `nodes_paper.json`
- `nodes_review.json` (if review data exists)
- `knowledge_graph_stats.json`

Notes:
- This step may call LLM to generate `llm_enhanced_summary` for patterns; it requires LLM API access.
- It should be deterministic given the same inputs and LLM outputs (LLM outputs are the non-deterministic part unless you use low temperature and stable model).

### 4.3 Rebuild edges + graph
Command:
```bash
python Paper-KG-Pipeline/scripts/build_edges.py
```

Expected outputs:
- `Paper-KG-Pipeline/output/edges.json`
- `Paper-KG-Pipeline/output/knowledge_graph_v2.gpickle`

### 4.4 Sanity check the KG
Run:
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

### 4.5 Alternative KG builder (non-ICLR / legacy multi-conference path)
This repo also contains an older “multi-conference” KG build pipeline (not the default used by the prebuilt ICLR artifacts):

- Node builder: `Paper-KG-Pipeline/scripts/tools/build_entity.py`
  - Inputs:
    - `Paper-KG-Pipeline/data/<CONFERENCE>/*_paper_node.json` (or `_all_paper_nodes.json`)
    - `Paper-KG-Pipeline/output/patterns_structured.json`
    - `Paper-KG-Pipeline/output/paper_to_pattern.json`
  - Outputs:
    - `Paper-KG-Pipeline/output/nodes_idea.json`
    - `Paper-KG-Pipeline/output/nodes_pattern.json`
    - `Paper-KG-Pipeline/output/nodes_domain.json`
    - `Paper-KG-Pipeline/output/nodes_paper.json`

This path is useful when you are not using the ICLR_25 JSONL dataset, but it assumes you already have:
1) extracted paper nodes into `Paper-KG-Pipeline/data/<CONFERENCE>/...`
2) a pattern clustering result in `Paper-KG-Pipeline/output/patterns_structured.json` + `paper_to_pattern.json`

If you don’t have those inputs, stick to the ICLR_25 path (Section 4.1) or use the prebuilt `Paper-KG-Pipeline/output/`.

### 4.6 (Optional but recommended) Rebuild `patterns_structured.json` for richer pattern prompts
If your goal is a **full “from scratch” rebuild** and you cleaned `Paper-KG-Pipeline/output/`, you may lose:
- `patterns_structured.json`
- `paper_to_pattern.json`
- `patterns_statistics.json`
- `patterns_guide.txt`

These are **not produced** by `build_entity_v3.py`, but the runtime pipeline will **use them when present** (especially `patterns_structured.json`).

The script that generates them is:
```bash
python Paper-KG-Pipeline/scripts/tools/generate_patterns.py
```

Important constraints (from the script itself):
- It expects paper nodes with `skeleton` + `tricks` under:
  - `Paper-KG-Pipeline/data/<CONFERENCE>/*_paper_node.json`
- These inputs are **not shipped** in this repo by default. You must generate them yourself (legacy extraction) or keep the prebuilt `output/patterns_structured.json`.
- It uses some legacy env var names:
  - `EMBED_API_URL` / `EMBED_MODEL` (not `EMBEDDING_API_URL` / `EMBEDDING_MODEL`)
  - `SILICONFLOW_API_KEY` as the auth token

If you want to run it without editing the script, you can bridge env vars like:
```bash
export EMBED_API_URL="$EMBEDDING_API_URL"
export EMBED_MODEL="$EMBEDDING_MODEL"
python Paper-KG-Pipeline/scripts/tools/generate_patterns.py
```

If you don’t have the required `data/<CONFERENCE>/*_paper_node.json` inputs, **skip this step**; the pipeline will still run with `nodes_pattern.json` only.

---

## 5) Index rebuild (Level B/C)

### 5.1 Novelty index (local similarity against papers)
Build:
```bash
python Paper-KG-Pipeline/scripts/tools/build_novelty_index.py
```

Where it writes:
- `Paper-KG-Pipeline/output/novelty_index__<PROFILE_ID>/...` when `I2P_INDEX_DIR_MODE=auto_profile`
- or `Paper-KG-Pipeline/output/novelty_index` in manual mode

What it depends on:
- `Paper-KG-Pipeline/output/nodes_paper.json`
- `EMBEDDING_API_URL / EMBEDDING_MODEL / EMBEDDING_API_KEY`

### 5.2 Recall offline index (optional but recommended for speed)
If `I2P_RECALL_USE_OFFLINE_INDEX=1`, the pipeline will prefer the offline recall index.

Build:
```bash
python Paper-KG-Pipeline/scripts/tools/build_recall_index.py
```

Where it writes:
- `Paper-KG-Pipeline/output/recall_index__<PROFILE_ID>/...` when `auto_profile` is enabled

What it depends on:
- `Paper-KG-Pipeline/output/nodes_idea.json`
- `Paper-KG-Pipeline/output/nodes_paper.json`
- embedding config

### 5.3 Canonical subdomain taxonomy (optional; improves Path2 signal)
This reduces:
- duplicate/synonymous raw subdomains
- “giant bucket” generic tags (stoplist)
- extreme small buckets via constrained merge

Build once (offline):
```bash
python Paper-KG-Pipeline/scripts/tools/build_subdomain_taxonomy.py --output Paper-KG-Pipeline/output/subdomain_taxonomy.json
```

Enable (runtime):
```bash
I2P_SUBDOMAIN_TAXONOMY_ENABLE=1
I2P_SUBDOMAIN_TAXONOMY_PATH=Paper-KG-Pipeline/output/subdomain_taxonomy.json
I2P_SUBDOMAIN_TAXONOMY_STOPLIST_MODE=drop
```

If enabled but the file is missing/invalid, the system should warn and safely fall back to raw subdomains.

### 5.4 “Do I need to build these manually?”
- **Novelty/recall indexes**: if `I2P_INDEX_AUTO_PREPARE=1` and `I2P_INDEX_ALLOW_BUILD=1`, the pipeline will try to auto-build missing indexes at runtime.
- **Subdomain taxonomy**: intentionally an offline build step (quality-first, deterministic, reusable). If enabled but missing, runtime falls back to raw subdomains.

### 5.5 Force rebuild / clean state (when debugging)
Sometimes you want to force a rebuild because:
- you changed embedding settings
- a previous build was interrupted
- you want to validate determinism

Recommended options:

- Novelty index:
  - build script supports `--force-rebuild` and `--resume`:
    ```bash
    python Paper-KG-Pipeline/scripts/tools/build_novelty_index.py --force-rebuild
    ```
- Recall index:
  - build script supports `--force-rebuild` (see `--help`):
    ```bash
    python Paper-KG-Pipeline/scripts/tools/build_recall_index.py --force-rebuild
    ```
- Subdomain taxonomy:
  - just re-run builder with a chosen `--output` path (it overwrites):
    ```bash
    python Paper-KG-Pipeline/scripts/tools/build_subdomain_taxonomy.py --output Paper-KG-Pipeline/output/subdomain_taxonomy.json
    ```

Avoid manually deleting files unless you know what you’re removing; prefer the scripts’ `--force-rebuild` flags.

---

## 6) Run the full pipeline (Idea → Story)

### 6.1 One-command run (recommended)
```bash
python Paper-KG-Pipeline/scripts/idea2story_pipeline.py "我们研究音频-文本大模型的文本主导偏见..."
```

### 6.2 What happens during the run
- Loads data from `Paper-KG-Pipeline/output/`
- Runs three-path recall (Path1/Path2/Path3)
- Pattern selection
- Story generation
- Multi-agent critic
- Refinement loop (if needed)
- Novelty + final verification (if enabled)
- Writes:
  - `Paper-KG-Pipeline/output/final_story.json`
  - `Paper-KG-Pipeline/output/pipeline_result.json`
  - `log/run_.../` (if enabled)
  - `results/run_.../` bundle (if enabled)

### 6.3 Why you might see “three-path recall” multiple times
Some features can trigger additional recalls:
- Idea Packaging (pattern-guided double recall)
- Novelty mode (iterating across candidate patterns)

If you want fewer recalls for debugging, disable Idea Packaging:
```bash
I2P_IDEA_PACKAGING_ENABLE=0
```

---

## 7) Outputs, logs, and how to inspect them

### 7.1 Latest output snapshot
The pipeline always writes the latest outputs to:
- `Paper-KG-Pipeline/output/final_story.json`
- `Paper-KG-Pipeline/output/pipeline_result.json`

### 7.2 Per-run logs (recommended)
If `I2P_ENABLE_LOGGING=1`, each run writes:
- `log/run_<timestamp>_<pid>_<hash>/events.jsonl`
- `log/run_<...>/llm_calls.jsonl`
- `log/run_<...>/embedding_calls.jsonl`
- `log/run_<...>/meta.json`

### 7.3 Per-run bundled artifacts
If `I2P_RESULTS_ENABLE=1`, each run creates:
- `results/run_<...>/final_story.json`
- `results/run_<...>/pipeline_result.json`
- `results/run_<...>/manifest.json`
- `results/run_<...>/run_log/` (copied log folder; no symlinks)

---

## 8) Performance / stability knobs (practical defaults)

### 8.1 Disable adaptive densify (faster + fewer JSON failures)
Adaptive densify triggers a second anchored critic round and increases latency.

Recommended:
```bash
I2P_ANCHOR_DENSIFY_ENABLE=0
```

### 8.2 Strict JSON vs. “keep going”
If you get repeated “Critic JSON invalid after retries” errors, you can temporarily disable strict mode:
```bash
I2P_CRITIC_STRICT_JSON=0
```
This is mainly for development/debugging. For quality runs, strict mode is safer.

### 8.3 Index auto-prepare
By default, the pipeline will preflight and build missing indexes if allowed:
```bash
I2P_INDEX_AUTO_PREPARE=1
I2P_INDEX_ALLOW_BUILD=1
```

If you want “fail fast” when indexes are missing:
```bash
I2P_INDEX_ALLOW_BUILD=0
```

### 8.4 Typical “why did my run fail?” checklist
1) **No API key / wrong key**: LLM/Embedding calls fail or run simulated.
2) **Wrong endpoint URL**:
   - LLM should typically point to a Chat Completions compatible URL (often ends with `/v1/chat/completions`)
   - Embedding should point to an embeddings URL (often ends with `/v1/embeddings`)
3) **Embedding config changed but indexes reused**: expect index mismatch warnings or degraded recall; rebuild indexes for the new embedding profile.
4) **Critic strict JSON enabled + provider output unstable**: “Critic JSON invalid after retries”.
5) **Adaptive densify enabled**: Phase 3 becomes much slower (second anchored critic round).

---

## 9) Frontend (optional local UI)

Start:
```bash
python frontend/server/app.py --host 127.0.0.1 --port 8080
```

Open:
```text
http://127.0.0.1:8080/
```

The frontend:
- starts the pipeline as a subprocess
- shows high-level stages
- allows downloading logs as a zip

---

## 10) Developer validation checklist (recommended)

### 10.1 Basic syntax checks
```bash
python -m py_compile \
  Paper-KG-Pipeline/src/idea2paper/config.py \
  Paper-KG-Pipeline/src/idea2paper/recall/recall_system.py \
  Paper-KG-Pipeline/scripts/idea2story_pipeline.py \
  Paper-KG-Pipeline/scripts/tools/build_novelty_index.py \
  Paper-KG-Pipeline/scripts/tools/build_recall_index.py \
  Paper-KG-Pipeline/scripts/tools/build_subdomain_taxonomy.py
```

### 10.2 Smoke test (offline-ish)
If you don’t have keys, you can still validate the pipeline wiring by:
- disabling strict JSON
- disabling novelty/verification
- letting LLM run in simulated mode

Example:
```bash
I2P_CRITIC_STRICT_JSON=0 I2P_VERIFICATION_ENABLE=0 I2P_NOVELTY_ENABLE=0 \
python Paper-KG-Pipeline/scripts/idea2story_pipeline.py "test idea"
```

---

## 11) Code pointers (source of truth)
This repo evolves quickly; treat **code + root README** as the source of truth.

Useful entrypoints to read when debugging a rebuild:
- Config loading & priorities: `Paper-KG-Pipeline/src/idea2paper/config.py`
- Pipeline CLI entry: `Paper-KG-Pipeline/scripts/idea2story_pipeline.py`
- Recall system (Path1/Path2/Path3): `Paper-KG-Pipeline/src/idea2paper/recall/recall_system.py`
- Run logs: `Paper-KG-Pipeline/src/idea2paper/infra/run_logger.py`
- Results bundling: `Paper-KG-Pipeline/src/idea2paper/infra/result_bundler.py`
