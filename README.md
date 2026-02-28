<p align="center">
  <img src="assets/images/logo2.png" alt="logo" width="650">
</p>

<div align="center"> 

[![PyPI - Python Version](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()
[![arXiv - Idea2Story](https://img.shields.io/badge/arXiv-2601.20833-b31b1b.svg)](https://arxiv.org/abs/2601.20833)
[![Stars](https://img.shields.io/github/stars/AgentAlphaAGI/Idea2Paper?style=social)](https://github.com/AgentAlphaAGI/Idea2Paper/stargazers)

[English](README.md) | [中文](README-zh_CN.md)

</div>


<details>
  <summary><h2>📌 Table of Contents</h2></summary>

  <br/>

  - [📄 Idea2Paper](#-idea2paper)
  - [💬 User Community](#-user-community)
  - [✨ Key Features](#-key-features)
  - [📦 Outputs](#-outputs)
  - [🚀 Getting Started](#-getting-started)
  - [🌐 Frontend (Local Web UI)](#-frontend-local-web-ui)
  - [🔨 Knowledge Graph Builder](#-knowledge-graph-builder)
  - [🤖 Anchored Multi-Agent Review](#-anchored-multi-agent-review)
  - [📚 Files & Docs](#-files--docs)
  - [🤝 Contributing & License](#-contributing--license)
  - [🙏 Credits](#-credits)
  - [👥 Contributors](#-contributors)
  - [📑 Citation (Idea2Story)](#-citation-idea2story)

</details>

---

## 📄 Idea2Paper

Idea2Paper is an end-to-end research agent framework that aims to systematically define and analyze the major stages of the contemporary research process, along with the core challenges inherent to each stage. Rather than treating paper writing as a monolithic generation problem, Idea2Paper explicitly decomposes scientific research into structured phases and identifies critical bottlenecks that hinder the transformation of raw ideas into coherent, submission-ready academic narratives. Through this analysis, Idea2Paper highlights that one of the most fundamental yet underexplored challenges lies in research paradigm generation—the process of converting an underspecified research idea into a logically consistent, academically grounded research story. Existing systems often struggle to produce stable and reusable research paradigms, especially when reasoning is performed entirely at runtime and under limited contextual grounding.

To address these challenges in a principled and engineering-oriented manner, Idea2Paper adopts a modular system design. Instead of immediately building a fully end-to-end writing system, the project prioritizes the construction of targeted engineering submodules that tackle specific bottlenecks in the research pipeline. As the first and core engineering submodule, Idea2Story is introduced to directly address the problem of research paradigm generation. Idea2Story focuses on transforming underspecified research ideas into complete, coherent, and submission-ready scientific narrative skeletons. By providing a structured research story as an intermediate representation, Idea2Story establishes a stable foundation for downstream stages such as method development, experiment design, and paper writing.
 
> **Idea2Paper** : https://www.researchgate.net/publication/400280248_Idea2Paper_What_Should_an_End-to-End_Research_Agent_Really_Do

### Idea2Story (Core Submodule of Idea2Paper)

Idea2Story introduces a pre-computation–driven framework that shifts literature understanding
from runtime reasoning to offline knowledge graph construction, enabling more efficient and
reliable autonomous scientific discovery.

> **Idea2Story** : https://arxiv.org/abs/2601.20833

### 🧠 Core Philosophy
- **Knowledge-Driven**: Uses ICLR data to build a comprehensive knowledge graph.
- **Auditable Review**: Implements an anchored multi-agent review system for objective feedback.
- **Automated Refinement**: Includes RAG deduplication and intelligent revision to enhance novelty.

<div align="center">
<img src="https://arxiv.org/html/2601.20833v1/x1.png" alt="Idea2Paper Architecture" width="800"/>
<br/>
<em>Idea2Story pipeline architecture (a core module within Idea2Paper)</em>
</div>

## 💬 User Community

| WeChat Group                                                                                | Discord Channel |
|---------------------------------------------------------------------------------------------| --- |
| <p align="center"> <img src="./assets/images/idea2paper_code.jpg" width="200" /><br/>  </p> | (https://discord.gg/XfAQYRZ4kk) |


## ✨ Key Features

- **🕸️ Knowledge Graph**: Built from ICLR data with Idea/Pattern/Domain/Paper nodes.
- **🎣 Advanced Retrieval**: Three-path retrieval (Idea/Domain/Paper) with two-stage ranking (Jaccard + Embedding).
- **📝 Idea2Story Generation**: From pattern selection to story generation, anchored review, and smart correction.
- **🤖 Anchored Multi-Agent Review**: Uses real review statistics as anchors for relative comparisons, producing deterministic and auditable 1-10 scores.
- **📊 Comprehensive Logging**: Per-run structured logs for full reproducibility and auditing.

## 📦 Outputs

- 📄 `Paper-KG-Pipeline/output/final_story.json`: Final structured Story (title/abstract/problem/method/contribs/experiments).
- 🔍 `Paper-KG-Pipeline/output/pipeline_result.json`: Full pipeline trace (reviews, corrections, audits).
- 📂 `log/run_.../`: Structured logs for every run.

## 🚀 Getting Started

### Prerequisites
- Python 3.10+

### Installation

```bash
pip install -r Paper-KG-Pipeline/requirements.txt
```
> **Note:** The embedding model is configurable via `EMBEDDING_MODEL` / `EMBEDDING_API_URL` (env or `i2p_config.json`). If you switch models, rebuild novelty/recall indexes or use model-specific index directories to avoid mismatch.  
> **Constraint:** the embedding dimension must match your index; if you switch models, rebuild indexes or use model-specific index dirs.  
> **Recommended (auto_profile):** set `I2P_INDEX_DIR_MODE=auto_profile` to auto-map each embedding model to its own index dirs: `Paper-KG-Pipeline/output/novelty_index__{model}` and `.../recall_index__{model}`.  
> Explicit `I2P_NOVELTY_INDEX_DIR` / `I2P_RECALL_INDEX_DIR` (env or `i2p_config.json`) override auto_profile.  
> **Tip (speed/stability):** set `I2P_ANCHOR_DENSIFY_ENABLE=0` to skip Adaptive Densify; otherwise Phase 3 Critic can be much slower and may fail due to strict JSON validation.  
> **Tip (debug):** if you repeatedly hit Critic JSON errors, set `I2P_CRITIC_STRICT_JSON=0` (or `critic.strict_json=false`) to disable strict mode and allow fallback.  
> **Tip (LLM temperature):** per-stage temperatures are configurable via `I2P_LLM_TEMPERATURE_*` or `llm.temperature.*`; defaults preserve current behavior. Critic is usually low temp for stability, while story generation can be moderate.  
> **Tip (Idea Packaging):** optional quality boost via pattern-guided idea packaging + double recall (default off). Enable with `I2P_IDEA_PACKAGING_ENABLE=1` or `idea.packaging_enable=true`.  
> **Tip (Subdomain taxonomy):** optional quality boost for Path2 to reduce duplicated/long-tail subdomains. When enabled, the pipeline auto-detects and (if `I2P_INDEX_ALLOW_BUILD=1`) auto-builds `subdomain_taxonomy.json` under `recall_index_dir` (recommended: leave `I2P_SUBDOMAIN_TAXONOMY_PATH` empty). First build uses batched embeddings; you can also build manually via `Paper-KG-Pipeline/scripts/tools/build_subdomain_taxonomy.py`.  
> **Supported (no code changes):** OpenAI-compatible Embeddings APIs (`/v1/embeddings`) that accept `input` as a string or a list.  
> **Not supported yet:** DashScope “native” embeddings endpoint (`/api/v1/services/embeddings/...`) requires an adapter.

### Dataset

👉 **[DATA](https://huggingface.co/datasets/AgentAlphaAGI/Paper-Review-Dataset/tree/main)** <br>

If you need to use the prebuilt local index, please place the two folders in `paper-embedding` from Hugging Face into `paper-KG-Pipeline/output`, <br>
```text
paper-KG-Pipeline/
└── output/
    ├── recall_index__{model}/
    └── novelty_index__{model}/
```
and make sure the embedding model matches the index you downloaded, otherwise errors may occur.

> **Migration note (auto_profile naming change):** if you previously used provider/urlhash-based dirs, you can either (A) rename the old folders to `recall_index__{model}` / `novelty_index__{model}`, or (B) keep old folder names and set `I2P_RECALL_INDEX_DIR` / `I2P_NOVELTY_INDEX_DIR` explicitly to those paths.


### Configuration

1. Copy `.env.example` to `.env` and fill in `LLM_API_KEY` (and optionally `LLM_PROVIDER`, `LLM_BASE_URL`).
2. (Optional) Copy `i2p_config.example.json` to `i2p_config.json` to tweak settings.

### Usage

```bash
python Paper-KG-Pipeline/scripts/idea2story_pipeline.py "your research idea"
```

## 🌐 Frontend (Local Web UI)

> **Status:** The frontend is currently unstable. We recommend running the pipeline from the terminal for now. We will improve the frontend in future updates.

Run a minimal local UI to launch the pipeline and view **only** high-level stage + final results (no raw logs on screen).

### Start

```bash
python frontend/server/app.py --host 127.0.0.1 --port 8080
```

Open in your browser:

```text
http://127.0.0.1:8080/
```

### What you can do in the UI
- Run the same pipeline entrypoint (`idea2story_pipeline.py`) from a web page.
- Configure `LLM_API_KEY`, `LLM_PROVIDER`, `LLM_BASE_URL`/`LLM_API_URL`, `LLM_MODEL` for the current run (not persisted by the server).
- Toggle Novelty / Verification.
- Download the current run logs as a zip.

For more details, see `frontend/README.md`.

### 🔨 Knowledge Graph Builder

The Web UI includes a **KG Builder** page that lets you build a custom knowledge graph from your own paper dataset, instead of relying on the prebuilt ICLR graph.

#### Dataset Format

Prepare a JSONL file where each line is a JSON object with the following fields:

| Field | Required | Description |
|-------|----------|-------------|
| `id` or `paper_id` | Yes | Unique paper identifier |
| `title` | Yes | Paper title |
| `abstract` | Yes | Paper abstract |
| `keywords` | No | List of keywords |
| `year` | No | Publication year |

Place the file under `Paper-KG-Pipeline/data/`. A sample dataset (`paper_reviews_dataset_iclr_sample_100.jsonl`) is included for testing.

#### How to Use

1. Start the server: `python frontend/server/app.py --host 127.0.0.1 --port 8080`
2. Open `http://127.0.0.1:8080/` and navigate to the **KG Builder** tab.
3. Select your dataset from the dropdown.
4. Configure LLM settings (API Key, Model, API URL).
5. Click **Start Build**.

The pipeline runs 4 steps sequentially:

| Step | Script | Description |
|------|--------|-------------|
| 1 | `extract_patterns_ICLR_en_local.py` | Extract writing patterns from papers via LLM |
| 2 | `generate_clusters.py` | Cluster patterns using SBERT + HDBSCAN |
| 3 | `build_entity_v3.py` | Build Idea / Pattern / Domain / Paper nodes |
| 4 | `build_edges.py` | Build edges and export `.gpickle` graph |

Progress and logs are displayed in real-time on the page. Once completed, the output files are written to `Paper-KG-Pipeline/output/` and can be used directly by the `idea2story_pipeline.py`.

### Output

```text
output/
├── final_story.json # Final generated paper story
├── pipeline_result.json # Full pipeline results
└── log.json # Detailed logs
```
Check `final_story.json` for the result and `pipeline_result.json` for the full process.


## 🤖 Anchored Multi‑Agent Review

Instead of arbitrary scores, this project uses **anchored comparisons**. We select anchor papers with known scores, ask LLMs to compare your target against these anchors (better/tie/worse), and then deterministically fit a final numeric score. This ensures the review process is auditable and grounded in real-world data.

## 📚 Files & Docs



- **Core Code**: `Paper-KG-Pipeline/src/idea2paper/`
- **Documentation**:

| No. | Document | Content | Target Audience |
| ----- |--------------------------| ---------------- | ------- |
| **0** | [Project Overview](Paper-KG-Pipeline/docs/00_PROJECT_OVERVIEW.md) | Overall architecture, core modules, parameter configuration, execution workflow | Everyone |
| **1** | [Knowledge Graph Construction](docs/01_KG_CONSTRUCTION.md) | Data sources, node/edge definitions, LLM enhancement, how to run | Developers |
| **2** | [Retrieval System](docs/02_RECALL_SYSTEM.md) | Three-way retrieval strategies, similarity computation, performance optimization | Developers |
| **3** | [Idea2Story Pipeline](docs/03_IDEA2STORY_PIPELINE.md) | Pattern selection, Idea fusion, story reflection, critic review | Developers |

- **Review Details**: [MULTIAGENT_REVIEW.md](MULTIAGENT_REVIEW.md)

## 🤝 Contributing & License

We welcome PRs and Issues! Please follow the contribution guidelines.
Licensed under the **MIT License**.

## 🙏 Credits

- **Data Source**: ICLR (see KG construction docs)
- **Inspiration**: Auditable, anchor-centered review processes.
- **Community Support**: [agentAlpha Community](https://agentalpha.top)

## 👥 Contributors

<a href="https://github.com/AgentAlphaAGI/Idea2Paper/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=AgentAlphaAGI/Idea2Paper" />
</a>

## 📑 Citation (Idea2Story)

If you find **Idea2Story** useful, please cite:

```bibtex
@misc{xu2026idea2storyautomatedpipelinetransforming,
  title={Idea2Story: An Automated Pipeline for Transforming Research Concepts into Complete Scientific Narratives},
  author={Tengyue Xu and Zhuoyang Qian and Gaoge Liu and Li Ling and Zhentao Zhang and Biao Wu and Shuo Zhang and Ke Lu and Wei Shi and Ziqi Wang and Zheng Feng and Yan Luo and Shu Xu and Yongjin Chen and Zhibo Feng and Zhuo Chen and Bruce Yuan and Harry Wang and Kris Chen},
  year={2026},
  eprint={2601.20833},
  archivePrefix={arXiv},
  primaryClass={cs.CE},
  url={https://arxiv.org/abs/2601.20833}
}
```


```bibtex
@article{xu2026idea2paper,
  title={Idea2Paper: What Should an End-to-End Research Agent Really Do?},
  author={Xu, Tengyue and Qian, Zhuoyang and Liu, Gaoge and Zhang, Zhentao and Ling, Li and Wu, Biao and Zhang, Shuo and Lu, Ke and Shi, Wei and Wang, Ziqi and others},
  year={2026}
}

```


---

## 📈 Star History

<a href="https://star-history.com/#AgentAlphaAGI/Idea2Paper&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=AgentAlphaAGI/Idea2Paper&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=AgentAlphaAGI/Idea2Paper&type=Date" />
   <img alt="Star History Chart"
     src="https://api.star-history.com/svg?repos=AgentAlphaAGI/Idea2Paper&type=Date&v=20260130" />
 </picture>
</a>


---
