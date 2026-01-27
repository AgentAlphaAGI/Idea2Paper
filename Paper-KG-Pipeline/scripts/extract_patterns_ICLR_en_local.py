# extract_patterns_100.py
import os, json, time
from typing import Any, Dict, List
from datasets import load_dataset
from tqdm import tqdm
from pathlib import Path

from openai import OpenAI

# ===== HF dataset =====
DATASET_NAME = "AgentAlphaAGI/Paper-Review-Dataset"
SPLIT = "train"
N = 10  # 本次运行抽取多少新记录

# ===== LLM Model =====
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")  # 你可改成 gpt-4.1 / gpt-4o 等

# 获取项目根目录 (知识图谱Pipeline)
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

# ===== Local input file (downloaded JSONL) =====
INPUT_PATH = os.getenv(
    "INPUT_JSONL_PATH",
    str(PROJECT_ROOT / "data" / "ICLR_merged_cleaned_huggingface.jsonl")
)

# ===== Output file =====
# 输出路径
OUTPUT_DIR = PROJECT_ROOT / "output"
OUT_PATH = OUTPUT_DIR / "iclr_patterns_full.jsonl"


# ====== English Prompt ======
PROMPT_TEMPLATE = r"""
【Role】
You are a “Top-Conference Research Pattern Extractor”.
Your task is NOT to summarize the paper, but to extract the reusable research narration framework that turns ordinary techniques into a convincing top-tier contribution through framing, conceptual packaging, and evidence design.

【Task Objective】
Given a paper’s title, keywords, and abstract, extract the paper’s Research Pattern core framework:
⟨Base Problem, Solution Pattern, Story⟩,
and identify the paper’s core idea.

This extraction is used to learn how top papers package methods into high-impact research narratives. The Story field is the most important.

【Key Definitions】
1) Research Pattern = ⟨Base Problem, Solution Pattern, Story⟩
- Base Problem:
  The concrete, actionable pain point in a specific scenario that the paper targets.
  Must be specific and grounded. Avoid vague phrasing such as “improves performance” without context.
- Solution Pattern:
  The core technical route that solves the Base Problem.
  Explicitly describe key components, pipeline structure, and the mechanism of improvement.
- Story (MOST IMPORTANT):
  The conceptual packaging and narrative framing that makes the work look novel, forward-looking, and impactful.
  Focus on narrative devices such as reframing the problem, introducing a new lens, elevating engineering issues into research questions, and highlighting long-term implications.

2) idea (Paper-level, OPTIONAL)
- A concise 1–2 sentence description of the paper’s key innovation or central insight.
- This should capture what is fundamentally new or different at a high level, without technical detail.
- If the core idea cannot be clearly stated, omit this field.

3) Taxonomy Fields (for retrieval)
- domain (string):
  One primary top-level field (Level-1). Must be a broad discipline label.
  Examples: "Machine Learning", "Computer Vision", "Natural Language Processing", "Systems", "Security & Privacy", "Fairness & Accountability".
- sub_domains (array of strings):
  Level-2 tags under the chosen domain. 2–5 items preferred.
  These should be specific and retrieval-friendly.
  Rules:
  - sub_domains must be consistent with the chosen domain
  - avoid repeating the domain name itself

4) application (string)
Concrete deployable scenarios implied by the paper.

【Hard Output Constraints】
- Output STRICT JSON only. No extra text, no Markdown, no comments.
- All values must be in English, concise, academic.
- paper_id must be: {paper_info['paper_id']}
- paper_title must be the input title; if missing use "N/A"
- idea is OPTIONAL
- domain must be a non-empty string
- sub_domains must be a non-empty array (at least 1 item)
- research_patterns must be a non-empty array (at least 1 object)
- No field inside research_patterns is allowed to be empty.

【Output Schema】
{
  "paper_id": "{paper_info['paper_id']}",
  "paper_title": "paper title",
  "idea": "concise 1–2 sentence description of the paper’s key innovation or core insight",
  "domain": "Level-1 domain",
  "sub_domains": ["Level-2 tag 1", "Level-2 tag 2"],
  "research_patterns": [
    {
      "base_problem": "concrete pain point in a specific scenario",
      "solution_pattern": "core technical route (components + workflow + mechanism)",
      "story": "conceptual packaging that makes the work look novel and high-impact",
      "application": "deployable scenarios"
    }
  ]
}

========================
【Few-shot Example 1】

【Input】
- paper_title: Research on Self-Evolution of Intelligent Agents Based on Reflect+Memory
- keywords: Intelligent Agent, Self-Evolution, Memory Mechanism
- abstract: Existing agents fail to retain experience after task execution, repeatedly making the same mistakes in similar tasks. This work introduces a Reflect module and a long-term Memory module to store, summarize, and retrieve experience for improved task execution over time.

【Output】
{
  "paper_id": "ARR_2022_106",
  "paper_title": "Research on Self-Evolution of Intelligent Agents Based on Reflect+Memory",
  "idea": "Enable intelligent agents to continuously improve by accumulating, summarizing, and reusing task experience through a reflection-plus-memory architecture",
  "domain": "Artificial Intelligence",
  "sub_domains": ["Agentic Systems", "Memory-Augmented Models", "Reflection", "Experience Retrieval"],
  "research_patterns": [
    {
      "base_problem": "Task-executing agents do not accumulate reusable experience, causing repeated failures and stagnant capability when facing recurring task families",
      "solution_pattern": "Augment the agent architecture with a reflection module that converts trajectories into distilled lessons, store them in long-term memory, and retrieve relevant lessons during inference to guide future decisions",
      "story": "Reframe agents from one-shot executors into a self-evolving paradigm where accumulated experience becomes a scalable capability multiplier, enabling sustained improvement and long-horizon autonomy",
      "application": "Customer-support automation with iterative playbook refinement, autonomous operations incident handling, robotics skill transfer across tasks, long-horizon decision-making assistants"
    }
  ]
}

========================
【Few-shot Example 2】

【Input】
- paper_title: Quantifying and Mitigating the Impact of Label Errors on Model Disparity Metrics
- keywords: Fairness, Label Noise, Influence Function, Disparity Metrics
- abstract: Existing fairness evaluation methods assume reliable labels, but real-world data often contains label errors that disproportionately affect different groups. This work analyzes label noise impact on disparity metrics and proposes an influence-function-based method to identify and correct high-impact label errors.

【Output】
{
  "paper_id": "ICLR_2023_089",
  "paper_title": "Quantifying and Mitigating the Impact of Label Errors on Model Disparity Metrics",
  "idea": "Diagnose and mitigate fairness failures by analyzing how individual label errors influence group-level disparity metrics",
  "domain": "Fairness & Accountability",
  "sub_domains": ["Label Noise", "Influence Functions", "Disparity Metrics", "Model Auditing"],
  "research_patterns": [
    {
      "base_problem": "Fairness evaluation becomes unreliable in the presence of label noise, systematically distorting disparity metrics and disproportionately harming minority groups",
      "solution_pattern": "Extend influence functions from loss-based analysis to group disparity metrics in order to quantify the effect of individual label perturbations and prioritize high-impact label corrections",
      "story": "Reframe fairness from a model optimization problem into an auditing and reliability problem, introducing a principled framework for diagnosing and correcting data-induced fairness failures",
      "application": "Fairness auditing in high-stakes decision systems, robustness evaluation under noisy labels, automated data quality inspection pipelines"
    }
  ]
}

========================
【Now Process This Paper】
- paper_title: {paper_info['paper_title']}
- keywords: {paper_info['keywords']}
- abstract: {paper_info['abstract']}

Return STRICT JSON only.


"""


def build_paper_info(row: Dict[str, Any]) -> Dict[str, Any]:
    # HF 数据集字段：title/authors/abstract/pdf_url/source_url/id/related_notes/year/conference/content/content_meta
    # keywords：数据集中没有单独列，这里先留空数组（后续你可加 keyphrase 模块自动生成）
    return {
        "paper_id": row.get("id", "") or "",
        "paper_title": row.get("title", "") or "无",
        "keywords": [],  # <- 先空
        "abstract": (row.get("abstract", "") or "").strip(),
        "source_url": row.get("source_url", ""),
        "pdf_url": row.get("pdf_url", ""),
        "year": str(row.get("year", "")),
        "conference": row.get("conference", ""),
    }


def render_prompt(paper_info: Dict[str, Any]) -> str:
    # 这里用最简单的字符串替换；如果你担心 prompt 里有花括号冲突，可用更稳健的模板引擎
    prompt = PROMPT_TEMPLATE
    prompt = prompt.replace("{paper_info['paper_id']}", paper_info["paper_id"])
    prompt = prompt.replace("{paper_info['paper_title']}", paper_info["paper_title"])
    prompt = prompt.replace("{paper_info['keywords']}", ", ".join(paper_info["keywords"]) if paper_info["keywords"] else "无")
    prompt = prompt.replace("{paper_info['abstract']}", paper_info["abstract"])
    return prompt


def call_llm(client: OpenAI, prompt: str) -> dict:
    resp = client.responses.create(
        model=MODEL,
        input=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    text = resp.output_text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # 兜底：抽取 JSON 主体
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            return json.loads(text[start:end+1])
        raise


# ===== Local input file (downloaded JSONL) =====
INPUT_PATH = os.getenv(
    "INPUT_JSONL_PATH",
    str(PROJECT_ROOT / "data" / "ICLR_merged_cleaned_huggingface.jsonl")
)

def iter_jsonl(path: str):
    """Yield dict per line from a local JSONL file."""
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as e:
                # If a line is corrupted, skip it but keep a traceable warning
                print(f"[WARN] bad json at line {line_no}: {e}")
                continue


def load_done_ids(out_path: Path) -> set:
    """Resume key: treat both success and error records as done."""
    done = set()
    if not out_path.exists():
        return done
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            pid = obj.get("paper_id")
            if pid:
                done.add(pid)
    return done


def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY env var")

    client = OpenAI(api_key=api_key)

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Resume: read already processed ids
    done_ids = load_done_ids(OUT_PATH)
    print(f"[resume] already processed: {len(done_ids)}")
    print(f"[input] local jsonl: {INPUT_PATH}")

    newly_written = 0
    skipped = 0
    seen = 0

    # Append mode for resume
    with open(OUT_PATH, "a", encoding="utf-8") as f:
        for row in tqdm(iter_jsonl(INPUT_PATH), desc="Extracting patterns (local+resume)"):
            seen += 1
            paper_info = build_paper_info(row)
            paper_id = paper_info.get("paper_id", "")

            if not paper_id:
                skipped += 1
                continue

            if paper_id in done_ids:
                skipped += 1
                continue

            # Stop after writing N new records (N is "new extractions" count)
            if newly_written >= N:
                break

            prompt = render_prompt(paper_info)

            last_err = None
            ok = False

            for attempt in range(3):
                try:
                    obj = call_llm(client, prompt)
                    f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    f.flush()

                    done_ids.add(paper_id)
                    newly_written += 1
                    ok = True
                    break
                except Exception as e:
                    last_err = e
                    time.sleep(2.0 * (attempt + 1))

            if not ok:
                err = {
                    "paper_id": paper_id,
                    "paper_title": paper_info.get("paper_title", "N/A"),
                    "error": str(last_err),
                }
                f.write(json.dumps(err, ensure_ascii=False) + "\n")
                f.flush()

                # Mark as done to avoid infinite loop on the same paper
                done_ids.add(paper_id)
                newly_written += 1

    print(f"[done] scanned={seen}, newly_written={newly_written}, skipped={skipped}, out={OUT_PATH}")

if __name__ == "__main__":
    main()

