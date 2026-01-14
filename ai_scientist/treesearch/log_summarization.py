import json
import os
import sys

from .journal import Node, Journal

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, parent_dir)
from ai_scientist.llm import get_response_from_llm, extract_json_between_markers
from ai_scientist.treesearch.backend import get_ai_client
from ai_scientist.prompts.loader import load_prompt, render_prompt


report_summarizer_sys_msg = load_prompt(
    "log_summarization.report_summarizer_sys_msg"
)
output_format_control = load_prompt("log_summarization.output_format_control")
report_summarizer_prompt_template = load_prompt(
    "log_summarization.report_summarizer_prompt"
)
stage_aggregate_prompt_template = load_prompt(
    "log_summarization.stage_aggregate_prompt"
)


def get_nodes_infos(nodes):
    node_infos = ""
    for n in nodes:
        node_info = f"Node ID: {n.id}\n"
        node_info += (
            f"Plan: {n.overall_plan}\n"
            if hasattr(n, "overall_plan")
            else "Plan: Not available\n"
        )
        node_info += (
            f"Analysis: {n.analysis}\n"
            if hasattr(n, "analysis")
            else "Analysis: Not available\n"
        )
        node_info += (
            f"Numerical Results: {n.metric}\n"
            if hasattr(n, "metric")
            else "Numerical Results: Not available\n"
        )
        node_info += "Plot Analyses:\n"
        if hasattr(n, "plot_analyses") and n.plot_analyses:
            for plot in n.plot_analyses:
                node_info += f"- Plot Path: {plot.get('plot_path', 'Not available')}, Description: {plot.get('analysis', 'Not available')}\n"
        else:
            node_info += "No plot analyses available\n"
        node_infos += node_info + "\n"
    return node_infos


def get_summarizer_prompt(journal, stage_name):
    good_leaf_nodes = [n for n in journal.good_nodes if n.is_leaf]
    if not good_leaf_nodes:
        print("NO GOOD LEAF NODES!!!")
        good_leaf_nodes = [n for n in journal.good_nodes]
    node_infos = get_nodes_infos(good_leaf_nodes)
    prompt = render_prompt(
        "log_summarization.report_summarizer_prompt",
        node_infos=node_infos,
        stage_name=stage_name,
    )
    return report_summarizer_sys_msg, prompt + output_format_control


def get_stage_summary(journal, stage_name, model, client, max_tokens: int | None = None):
    sys_msg, prompt = get_summarizer_prompt(journal, stage_name)
    response = get_response_from_llm(prompt, client, model, sys_msg, max_tokens=max_tokens)
    summary_json = extract_json_between_markers(response[0])
    return summary_json


def get_node_log(node):
    node_dict = node.to_dict()
    # Only include keys that are relevant for logging/analysis
    keys_to_include = [
        "overall_plan",
        "analysis",
        "metric",
        "code",
        "plot_code",
        "plot_plan",
        "plot_analyses",
        "plot_paths",
        "vlm_feedback_summary",
        "exp_results_dir",
        "ablation_name",
    ]
    ret = {
        key: node_dict[key]
        for key in keys_to_include
        if key in node_dict and node_dict[key] is not None
    }
    if "exp_results_dir" in ret:
        original_dir_path = ret["exp_results_dir"]
        # Remove leading path segments before "experiment_results"
        idx = original_dir_path.find("experiment_results")
        short_dir_path = original_dir_path
        if idx != -1:
            short_dir_path = original_dir_path[idx:]

        ret["exp_results_dir"] = short_dir_path

        if os.path.isdir(original_dir_path):
            npy_files = [f for f in os.listdir(original_dir_path) if f.endswith(".npy")]
            # Prepend the shortened path to each .npy filename
            ret["exp_results_npy_files"] = [
                os.path.join(short_dir_path, f) for f in npy_files
            ]
        else:
            ret["exp_results_npy_files"] = []
    return ret


def update_summary(
    prev_summary,
    cur_stage_name,
    cur_journal,
    cur_summary,
    model,
    client,
    max_retry=5,
    max_tokens: int | None = None,
):
    good_leaf_nodes = [n for n in cur_journal.good_nodes if n.is_leaf]
    node_infos = get_nodes_infos(good_leaf_nodes)
    prompt = render_prompt(
        "log_summarization.stage_aggregate_prompt",
        prev_summary=prev_summary,
        stage_name=cur_stage_name,
        current_summary=cur_summary,
    )
    try:
        response = get_response_from_llm(
            prompt,
            client,
            model,
            load_prompt("log_summarization.stage_aggregate_system_message"),
            max_tokens=max_tokens,
        )
        summary_json = extract_json_between_markers(response[0])
        assert summary_json
    except Exception as e:
        if max_retry > 0:
            print(f"Error occurred: {e}. Retrying... ({max_retry} attempts left)")
            return update_summary(
                prev_summary,
                cur_stage_name,
                cur_journal,
                cur_summary,
                model,
                client,
                max_retry - 1,
                max_tokens=max_tokens,
            )
        else:
            print(f"Failed to update summary after multiple attempts. Error: {e}")
            raise
    return summary_json


overall_plan_summarizer_prompt = load_prompt(
    "log_summarization.overall_plan_summarizer_prompt"
)


def annotate_history(journal, cfg=None):
    for node in journal.nodes:
        if node.parent:
            max_retries = 3
            retry_count = 0
            while retry_count < max_retries:
                try:
                    if cfg.agent.get("summary", None) is not None:
                        model = cfg.agent.summary.model
                        max_tokens = cfg.agent.summary.get("max_tokens", None)
                    else:
                        model = "gpt-4o-2024-08-06"
                        max_tokens = None
                    client = get_ai_client(model)
                    response = get_response_from_llm(
                        overall_plan_summarizer_prompt.format(
                            prev_overall_plan=node.parent.overall_plan,
                            current_plan=node.plan,
                        ),
                        client,
                        model,
                        report_summarizer_sys_msg,
                        max_tokens=max_tokens,
                    )
                    node.overall_plan = extract_json_between_markers(response[0])[
                        "overall_plan"
                    ]
                    break
                except Exception as e:
                    retry_count += 1
                    if retry_count == max_retries:
                        print(f"Failed after {max_retries} attempts. Error: {e}")
                        raise
                    print(
                        f"Error occurred: {e}. Retrying... ({max_retries - retry_count} attempts left)"
                    )
        else:
            node.overall_plan = node.plan


def overall_summarize(journals, cfg=None):
    from concurrent.futures import ThreadPoolExecutor

    def process_stage(idx, stage_tuple):
        stage_name, journal = stage_tuple
        annotate_history(journal, cfg=cfg)
        if idx in [1, 2]:
            best_node = journal.get_best_node(cfg=cfg)
            # get multi-seed results and aggregater node
            child_nodes = best_node.children
            multi_seed_nodes = [
                n for n in child_nodes if n.is_seed_node and not n.is_seed_agg_node
            ]
            agg_node = None
            for n in child_nodes:
                if n.is_seed_node and n.is_seed_agg_node:
                    agg_node = n
                    break
            if agg_node is None:
                # skip agg node
                return {
                    "best node": get_node_log(best_node),
                    "best node with different seeds": [
                        get_node_log(n) for n in multi_seed_nodes
                    ],
                }
            else:
                return {
                    "best node": get_node_log(best_node),
                    "best node with different seeds": [
                        get_node_log(n) for n in multi_seed_nodes
                    ],
                    "aggregated results of nodes with different seeds": get_node_log(
                        agg_node
                    ),
                }
        elif idx == 3:
            good_leaf_nodes = [
                n for n in journal.good_nodes if n.is_leaf and n.ablation_name
            ]
            return [get_node_log(n) for n in good_leaf_nodes]
        elif idx == 0:
            if cfg.agent.get("summary", None) is not None:
                model = cfg.agent.summary.get("model", "")
                max_tokens = cfg.agent.summary.get("max_tokens", None)
            else:
                model = "gpt-4o-2024-08-06"
                max_tokens = None
            client = get_ai_client(model)
            summary_json = get_stage_summary(
                journal, stage_name, model, client, max_tokens=max_tokens
            )
            return summary_json

    from tqdm import tqdm

    with ThreadPoolExecutor() as executor:
        results = list(
            tqdm(
                executor.map(process_stage, range(len(list(journals))), journals),
                desc="Processing stages",
                total=len(list(journals)),
            )
        )
        draft_summary, baseline_summary, research_summary, ablation_summary = results

    return draft_summary, baseline_summary, research_summary, ablation_summary


if __name__ == "__main__":
    # Test
    example_path = "logs/247-run"

    def load_stage_folders(base_path):
        """
        Load the folders that start with 'stage_' followed by a number.

        Args:
            base_path (str): The base directory path where stage folders are located.

        Returns:
            list: A sorted list of stage folder paths.
        """
        stage_folders = []
        for folder_name in os.listdir(base_path):
            if folder_name.startswith("stage_"):
                stage_folders.append(os.path.join(base_path, folder_name))
        return sorted(stage_folders, key=lambda x: int(x.split("_")[1]))

    def reconstruct_journal(journal_data):
        # Create a mapping of node IDs to Node instances
        id_to_node = {}
        for node_data in journal_data["nodes"]:
            # Remove unused or invalid keys if needed
            if "actionable_insights_from_plots" in node_data:
                del node_data["actionable_insights_from_plots"]
            node = Node.from_dict(node_data)
            id_to_node[node.id] = node

        # Set up parent-child relationships using node2parent
        for node_id, parent_id in journal_data["node2parent"].items():
            child_node = id_to_node[node_id]
            parent_node = id_to_node[parent_id]
            child_node.parent = parent_node
            parent_node.children.add(child_node)

        # Create a Journal and add all nodes
        journal = Journal()
        journal.nodes.extend(id_to_node.values())

        return journal

    # Example usage
    stage_folders = load_stage_folders(example_path)
    journals = []
    for index, folder in enumerate(stage_folders, start=1):
        print(f"Stage {index}: {folder}")
        stage_name = os.path.basename(folder)
        journal_path = os.path.join(folder, "journal.json")
        if os.path.exists(journal_path):
            with open(journal_path, "r") as file:
                journal_data = json.load(file)
                print(f"Loaded journal.json for Stage {index}")
        else:
            print(f"No journal.json found for Stage {index}")
        journal = reconstruct_journal(journal_data)
        journals.append((stage_name, journal))

    # Convert manager journals to list of (stage_name, journal) tuples
    (
        draft_summary,
        baseline_summary,
        research_summary,
        ablation_summary,
    ) = overall_summarize(journals)
    log_dir = "logs/247-run"
    draft_summary_path = log_dir + "/draft_summary.json"
    baseline_summary_path = log_dir + "/baseline_summary.json"
    research_summary_path = log_dir + "/research_summary.json"
    ablation_summary_path = log_dir + "/ablation_summary.json"

    with open(draft_summary_path, "w") as draft_file:
        json.dump(draft_summary, draft_file, indent=2)

    with open(baseline_summary_path, "w") as baseline_file:
        json.dump(baseline_summary, baseline_file, indent=2)

    with open(research_summary_path, "w") as research_file:
        json.dump(research_summary, research_file, indent=2)

    with open(ablation_summary_path, "w") as ablation_file:
        json.dump(ablation_summary, ablation_file, indent=2)

    print(f"Summary reports written to files:")
    print(f"- Draft summary: {draft_summary_path}")
    print(f"- Baseline summary: {baseline_summary_path}")
    print(f"- Research summary: {research_summary_path}")
    print(f"- Ablation summary: {ablation_summary_path}")
