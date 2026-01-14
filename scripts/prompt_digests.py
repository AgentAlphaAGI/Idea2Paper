#!/usr/bin/env python3
import argparse
import hashlib
import json
import string
from pathlib import Path
from typing import Any

from ai_scientist.prompts.loader import load_prompt, prompt_root
from ai_scientist.treesearch.backend.utils import compile_prompt_to_md


SAMPLE_VALUES: dict[str, Any] = {
    "abstract": "ABSTRACT",
    "aggregator_code": "AGGREGATOR_CODE",
    "aggregator_out": "AGGREGATOR_OUT",
    "best_count": 0,
    "caption": "CAPTION",
    "citations": "CITATIONS",
    "code_section": "",
    "combined_summaries_str": "COMBINED_SUMMARIES",
    "convergence_status": "CONVERGED",
    "current_round": 1,
    "dataset_name": "DATASET",
    "datasets_tested": ["synthetic_sine"],
    "figure_count": 0,
    "figure_list": "FIGURE_LIST",
    "good_nodes": 1,
    "idea_text": "IDEA_TEXT",
    "issues_json": "[]",
    "latex_writeup": "LATEX_CONTENT",
    "main_stage_goal": "MAIN_STAGE_GOAL",
    "max_figures": 12,
    "max_reflections": 3,
    "metrics_json": "{}",
    "min_datasets": 2,
    "notes": "NOTES",
    "page_limit": 8,
    "paper_abstract": "PAPER_ABSTRACT",
    "paper_text": "PAPER_TEXT",
    "paper_title": "PAPER_TITLE",
    "plot_descriptions": "PLOT_DESCRIPTIONS",
    "plot_list": "FIG1, FIG2",
    "prev_overall_plan": "PREV_PLAN",
    "prev_summary": "PREV_SUMMARY",
    "recent_changes_json": "[]",
    "reflection_page_info": "REFLECTION_PAGE_INFO",
    "report": "REPORT_TEXT",
    "report_input": "REPORT_INPUT",
    "review_count": 2,
    "review_index": 1,
    "review_json": "{}",
    "review_text": "REVIEW_TEXT",
    "reviewer_count": 2,
    "stage_description": "STAGE_DESCRIPTION",
    "stage_goals": "STAGE_GOALS",
    "stage_name": "STAGE_NAME",
    "stage_name_label": "STAGE_NAME_LABEL",
    "stage_number": 1,
    "stage_progress": "STAGE_PROGRESS",
    "stage_name_value": "STAGE_NAME_VALUE",
    "stage_summary": "STAGE_SUMMARY",
    "stage_title": "STAGE_TITLE",
    "substage_goal": "SUBSTAGE_GOAL",
    "summaries": "SUMMARIES",
    "task_desc": {"Title": "TITLE", "Abstract": "ABSTRACT"},
    "title": "TITLE",
    "total_nodes": 5,
    "total_rounds": 3,
    "unused_figs": [],
    "invalid_figs": [],
    "used_figs": [],
    "vlm_feedback": "VLM_FEEDBACK",
    "plot_descriptions_str": "PLOT_DESCRIPTIONS",
    "workshop_description": "WORKSHOP_DESCRIPTION",
    "num_reflections": 3,
    "num_reviews_ensemble": 2,
    "exec_time_minutes": "1.00",
    "name": "NAME",
    "description": "DESCRIPTION",
    "short_hypothesis": "SHORT_HYPOTHESIS",
    "tool_descriptions": "TOOL_DESCRIPTIONS",
    "tool_names_str": "TOOL_NAMES",
    "Idea": "IDEA",
    "papers": "PAPERS",
    "report_input_str": "REPORT_INPUT",
    "summary": "SUMMARY",
    "plot_list_str": "PLOT_LIST",
    "plot_names": "PLOT_NAMES",
    "plot_paths": "PLOT_PATHS",
    "plot_paths_str": "PLOT_PATHS",
    "plot_paths_list": [],
    "plot_analyses": "PLOT_ANALYSES",
    "plot_analyses_str": "PLOT_ANALYSES",
    "plot_analyses_list": [],
    "plot_paths_json": "PLOT_PATHS_JSON",
    "plot_analyses_json": "PLOT_ANALYSES_JSON",
    "plot_analyses_summary": "PLOT_ANALYSES_SUMMARY",
    "node_infos": "NODE_INFOS",
    "node_summaries": "NODE_SUMMARIES",
    "stage_name_str": "STAGE_NAME",
    "current_summary": "CURRENT_SUMMARY",
    "overall_plan": "OVERALL_PLAN",
    "current_plan": "CURRENT_PLAN",
    "summary_output": "SUMMARY_OUTPUT",
    "analysis": "ANALYSIS",
    "execution_output": "EXEC_OUTPUT",
    "plot_descriptions_list": "PLOT_DESCRIPTIONS_LIST",
    "plot_selection": "PLOT_SELECTION",
    "review_img_cap_ref": "REVIEW_IMG_CAP_REF",
    "analysis_duplicate_figs": "ANALYSIS_DUPLICATE_FIGS",
    "review_img_selection": "REVIEW_IMG_SELECTION",
    "check_output": "CHECK_OUTPUT",
    "best_metric": "BEST_METRIC",
    "convergence": "CONVERGENCE",
    "total_attempts": 5,
    "good_attempts": 2
}


class SafeDict(dict):
    def __missing__(self, key: str) -> Any:
        return f"<{key}>"


def render_with_samples(text: str) -> str:
    formatter = string.Formatter()
    values: dict[str, Any] = {}
    for _, field_name, _, _ in formatter.parse(text):
        if not field_name:
            continue
        if field_name not in values:
            values[field_name] = SAMPLE_VALUES.get(field_name, f"<{field_name}>")
    return text.format_map(SafeDict(values))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compute_digests() -> dict[str, str]:
    registry = json.loads((prompt_root() / "registry.json").read_text(encoding="utf-8"))
    digests: dict[str, str] = {}
    for prompt_id, entry in registry.items():
        prompt_type = entry.get("type")
        prompt_value = load_prompt(prompt_id)
        if prompt_type == "template":
            rendered = render_with_samples(prompt_value)
            digests[prompt_id] = sha256_text(rendered)
        elif prompt_type == "text":
            digests[prompt_id] = sha256_text(prompt_value)
        elif prompt_type == "json":
            compiled = compile_prompt_to_md(prompt_value)
            digests[prompt_id] = sha256_text(compiled)
        elif prompt_type == "py":
            serialized = json.dumps(prompt_value, sort_keys=True)
            digests[prompt_id] = sha256_text(serialized)
        else:
            digests[prompt_id] = sha256_text(str(prompt_value))
    return digests


def write_digests(path: Path) -> None:
    digests = compute_digests()
    path.write_text(json.dumps(digests, indent=2), encoding="utf-8")
    print(f"Wrote digests to {path}")


def diff_digests(before_path: Path, after_path: Path) -> int:
    before = json.loads(before_path.read_text(encoding="utf-8"))
    after = json.loads(after_path.read_text(encoding="utf-8"))
    all_keys = sorted(set(before.keys()) | set(after.keys()))
    mismatches = []
    for key in all_keys:
        if before.get(key) != after.get(key):
            mismatches.append(key)
    if mismatches:
        print("Digest mismatches:")
        for key in mismatches:
            print(f"- {key}")
        return 1
    print("All digests match.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("before")
    sub.add_parser("after")
    sub.add_parser("diff")
    args = parser.parse_args()

    before_path = prompt_root() / "_digests_before.json"
    after_path = prompt_root() / "_digests_after.json"

    if args.command == "before":
        write_digests(before_path)
        return 0
    if args.command == "after":
        write_digests(after_path)
        return 0
    if args.command == "diff":
        if not before_path.exists() or not after_path.exists():
            print("Missing digest files; run 'before' and 'after' first.")
            return 1
        return diff_digests(before_path, after_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
