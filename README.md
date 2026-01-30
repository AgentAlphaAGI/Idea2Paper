# Idea2Paper

An end-to-end intelligent system that automatically transforms research ideas into publication-quality paper narratives using knowledge graphs built from ICLR 2025 papers.

## Overview

Idea2Paper leverages a knowledge graph constructed from 8,285 ICLR 2025 papers to generate well-structured academic paper frameworks following top-conference standards. Given a research idea, the system outputs a complete paper story structure including problem statement, methodology, contributions, and experiment design.

## Key Features

- **Knowledge Graph Foundation**: 16,791 nodes (Ideas, Patterns, Domains, Papers) and 444,872 edges built from ICLR 2025 papers
- **Three-Path Recall System**: Complementary retrieval via idea similarity, domain generalization, and paper similarity (27 seconds, 13x speedup)
- **LLM-Enhanced Patterns**: 124 writing patterns with both exemplars and inductive abstractions
- **Anchored Multi-Agent Review**: Objective scoring using real ICLR review statistics as ground truth
- **Adaptive Refinement**: Novelty mode, score rollback, and intelligent pattern injection
- **RAG Verification**: Collision detection against recent top-conference papers

## Architecture

```
User Idea
    |
    v
[Phase 1] Knowledge Graph (offline, one-time)
    |-- 8,284 Idea Nodes
    |-- 124 Pattern Nodes (LLM-enhanced)
    |-- 98 Domain Nodes
    |-- 8,285 Paper Nodes
    v
[Phase 2] Three-Path Recall (27 sec)
    |-- Path 1: Idea similarity -> Pattern (40%)
    |-- Path 2: Domain generalization -> Pattern (20%)
    |-- Path 3: Paper similarity -> Pattern (40%)
    v
[Phase 3] Story Generation & Refinement (3-10 min)
    |-- Pattern Classification (Stability/Novelty/Cross-domain)
    |-- Idea Fusion (conceptual integration)
    |-- Story Reflection (quality assessment)
    |-- Multi-Agent Critic Review (3 reviewers)
    |-- Iterative Refinement
    v
[Phase 4] RAG Verification (~30 sec)
    |-- Collision detection
    |-- Pivot strategy if needed
    v
Final Story Output
```

## Installation

### Requirements

- Python 3.10+
- NetworkX 2.8+
- NumPy 1.21+
- scikit-learn 1.0+
- Requests 2.28+

### Setup

```bash
# Clone the repository
git clone https://github.com/your-repo/idea2paper.git
cd idea2paper

# Install dependencies
pip install -r Paper-KG-Pipeline/requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your SiliconFlow API key
```

## Configuration

The system uses hierarchical configuration (priority: env vars > .env > i2p_config.json > defaults):

```bash
# Required in .env
SILICONFLOW_API_KEY=your_api_key
```

Optional settings in `i2p_config.json`:
- `llm_model`: LLM model for generation (default: Qwen3-14B)
- `embedding_model`: Embedding model (default: Qwen3-Embedding-4B)
- `max_refinement_rounds`: Maximum refinement iterations

## Usage

### Build Knowledge Graph (One-time)

```bash
cd Paper-KG-Pipeline

# Build nodes (~15 min)
python scripts/build_entity_v3.py

# Build edges (~3 min)
python scripts/build_edges.py
```

### Generate Paper Story

```bash
python scripts/idea2story_pipeline.py --idea "Your research idea here"
```

### Example

```bash
python scripts/idea2story_pipeline.py --idea "LLM-assisted domain data extraction with self-verification"
```

Output files are saved to `Paper-KG-Pipeline/output/`:
- `final_story.json`: Generated paper story
- `pipeline_result.json`: Complete pipeline execution result

## Project Structure

```
Idea2Paper/
├── Paper-KG-Pipeline/
│   ├── data/ICLR_25/           # Source data (ICLR 2025 papers)
│   ├── output/                  # Generated artifacts
│   │   ├── nodes_*.json        # Graph nodes
│   │   ├── edges.json          # Graph edges
│   │   └── knowledge_graph_v2.gpickle
│   ├── src/idea2paper/         # Core library
│   │   ├── config.py           # Configuration management
│   │   ├── infra/              # Infrastructure (LLM, logging)
│   │   ├── recall/             # Three-path recall system
│   │   ├── review/             # Multi-agent critic
│   │   └── pipeline/           # Generation pipeline
│   ├── scripts/                # Entry points
│   │   ├── build_entity_v3.py  # Node construction
│   │   ├── build_edges.py      # Edge construction
│   │   └── idea2story_pipeline.py  # Main pipeline
│   └── docs/                   # Documentation
└── log/                        # Run logs (auto-generated)
```

## Multi-Agent Review System

The review system uses three specialized reviewers:
- **Methodology Reviewer**: Evaluates technical soundness
- **Novelty Reviewer**: Assesses innovation and uniqueness
- **Storyteller Reviewer**: Judges narrative quality and clarity

Key features:
- Ground truth anchoring with real ICLR review statistics
- Deterministic scoring via weighted least squares fitting
- Pattern-specific quantile thresholds (not fixed scores)

## Performance

| Metric | Value |
|--------|-------|
| Knowledge Graph Nodes | 16,791 |
| Knowledge Graph Edges | 444,872 |
| Recall Speed | 27 seconds |
| Typical Pipeline Runtime | 5-7 minutes |
| Pattern Coverage | 124/124 (100%) |
| Idea Coverage | 8,284/8,285 (100%) |

## Logging

Each run creates a timestamped log directory in `log/run_YYYYMMDD_HHMMSS_<pid>_<rand>/`:
- `meta.json`: Run metadata
- `events.jsonl`: Key pipeline events
- `llm_calls.jsonl`: LLM input/output records
- `embedding_calls.jsonl`: Embedding call details

## Documentation

Detailed documentation is available in `Paper-KG-Pipeline/docs/`:
- `00_PROJECT_OVERVIEW.md`: Architecture overview
- `01_KG_CONSTRUCTION.md`: Knowledge graph building
- `02_RECALL_SYSTEM.md`: Recall strategy details
- `03_IDEA2STORY_PIPELINE.md`: Generation pipeline

## License

[Add your license here]

## Citation

If you use Idea2Paper in your research, please cite:

```bibtex
@software{idea2paper,
  title={Idea2Paper: Transforming Research Ideas into Paper Narratives},
  year={2025}
}
```
