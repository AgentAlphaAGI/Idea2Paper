import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from rich import print

from ai_scientist.llm import create_client, get_response_from_llm
from ai_scientist.utils.token_tracker import token_tracker
from ai_scientist.perform_icbinb_writeup import (
    load_idea_text,
    load_exp_summaries,
    filter_experiment_summaries,
)
from ai_scientist.prompts.loader import render_prompt

DEFAULT_MAX_FIGURES = 12


def build_aggregator_system_message(max_figures: int) -> str:
    return render_prompt(
        "perform_plotting.aggregator_system",
        max_figures=max_figures,
    )


def build_aggregator_prompt(combined_summaries_str, idea_text):
    return render_prompt(
        "perform_plotting.aggregator_user",
        idea_text=idea_text,
        combined_summaries_str=combined_summaries_str,
    )


def extract_code_snippet(text: str) -> str:
    """
    Look for a Python code block in triple backticks in the LLM response.
    Return only that code. If no code block is found, return an empty string.
    """
    pattern = r"```(?:python)?(.*?)```"
    matches = re.findall(pattern, text, flags=re.DOTALL)
    return matches[0].strip() if matches else ""


def extract_code_blocks(text: str) -> list[str]:
    pattern = r"```(?:python)?(.*?)```"
    return [match.strip() for match in re.findall(pattern, text, flags=re.DOTALL)]


def select_compilable_code(text: str) -> tuple[str, str]:
    blocks = extract_code_blocks(text)
    if not blocks:
        return "", "No code block found."
    last_error = ""
    for block in blocks:
        try:
            compile(block, "<auto_plot_aggregator>", "exec")
            return block, ""
        except Exception:
            last_error = traceback.format_exc()
    return "", last_error


def scrub_plot_paths(value):
    if isinstance(value, dict):
        return {
            key: scrub_plot_paths(val)
            for key, val in value.items()
            if key not in {"plot_paths", "plot_path"}
        }
    if isinstance(value, list):
        return [scrub_plot_paths(item) for item in value]
    return value


def collect_exp_results_npy_files(value, output):
    if isinstance(value, dict):
        for key, val in value.items():
            if key == "exp_results_npy_files":
                if isinstance(val, list):
                    output.extend(val)
                elif isinstance(val, str):
                    output.append(val)
            else:
                collect_exp_results_npy_files(val, output)
        return
    if isinstance(value, list):
        for item in value:
            collect_exp_results_npy_files(item, output)


def normalize_npy_path(raw_path: str, base_folder: str, idea_dir_name: str) -> str | None:
    if not isinstance(raw_path, str):
        return None
    candidate = raw_path.strip()
    if not candidate:
        return None
    if os.path.isabs(candidate):
        normalized = candidate
    else:
        normalized = candidate.replace("\\", "/")
        redundant_prefix = f"runs/experiments/{idea_dir_name}/"
        if normalized.startswith(redundant_prefix):
            normalized = normalized[len(redundant_prefix) :]
        elif normalized.startswith(f"./{redundant_prefix}"):
            normalized = normalized[len(f"./{redundant_prefix}") :]
        normalized = os.path.abspath(os.path.join(base_folder, normalized))
    if normalized.endswith(".npy") and os.path.exists(normalized):
        return normalized
    return None


def scan_npy_files(base_folder: str) -> list[str]:
    search_root = Path(base_folder) / "logs" / "0-run" / "experiment_results"
    if not search_root.is_dir():
        return []
    return sorted({str(path.resolve()) for path in search_root.rglob("experiment_data.npy")})


def write_plot_agg_manifest(base_folder: str, filtered_summaries) -> tuple[str, list[str]]:
    idea_dir_name = os.path.basename(os.path.abspath(base_folder))
    manifest_path = os.path.join(base_folder, "plot_agg_manifest.json")
    raw_paths: list[str] = []
    collect_exp_results_npy_files(filtered_summaries, raw_paths)

    normalized_paths = []
    for raw_path in raw_paths:
        normalized = normalize_npy_path(raw_path, base_folder, idea_dir_name)
        if normalized:
            normalized_paths.append(normalized)
    npy_files = sorted(set(normalized_paths))
    if not npy_files:
        npy_files = scan_npy_files(base_folder)

    manifest = {"base_dir": os.path.abspath(base_folder), "npy_files": npy_files}
    try:
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=True)
    except Exception as exc:
        print(f"[yellow]Warning:[/yellow] Failed to write plot_agg_manifest.json: {exc}")
    if not npy_files:
        print("[yellow]Warning:[/yellow] No .npy files found for plot aggregation.")
    return manifest_path, npy_files


def count_figure_files(figures_dir: str) -> int:
    if not os.path.isdir(figures_dir):
        return 0
    allowed_ext = {".png", ".pdf", ".svg", ".jpg", ".jpeg"}
    return len(
        [
            f
            for f in os.listdir(figures_dir)
            if os.path.isfile(os.path.join(figures_dir, f))
            and os.path.splitext(f)[1].lower() in allowed_ext
        ]
    )


def run_aggregator_script(
    aggregator_code, aggregator_script_path, base_folder, script_name
):
    if not aggregator_code.strip():
        print("No aggregator code was provided. Skipping aggregator script run.")
        return ""
    with open(aggregator_script_path, "w") as f:
        f.write(aggregator_code)

    print(
        f"Aggregator script written to '{aggregator_script_path}'. Attempting to run it..."
    )

    aggregator_out = ""
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            cwd=base_folder,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        aggregator_out = result.stdout + "\n" + result.stderr
        print("Aggregator script ran successfully.")
    except subprocess.CalledProcessError as e:
        aggregator_out = (e.stdout or "") + "\n" + (e.stderr or "")
        print("Error: aggregator script returned a non-zero exit code.")
        print(e)
    except Exception as e:
        aggregator_out = str(e)
        print("Error while running aggregator script.")
        print(e)

    return aggregator_out


def aggregate_plots(
    base_folder: str,
    model: str = "o1-2024-12-17",
    n_reflections: int = 5,
    max_figures: int = DEFAULT_MAX_FIGURES,
    max_tokens: int | None = None,
) -> None:
    filename = "auto_plot_aggregator.py"
    aggregator_script_path = os.path.join(base_folder, filename)
    figures_dir = os.path.join(base_folder, "figures")
    best_figures_dir = os.path.join(base_folder, "_figures_best")

    # Many summaries store .npy paths relative to "experiment_results/...". If the
    # top-level folder was cleaned up, recreate it from logs for reruns.
    experiment_results_dir = os.path.join(base_folder, "experiment_results")
    if os.path.islink(experiment_results_dir) and not os.path.exists(experiment_results_dir):
        os.unlink(experiment_results_dir)
    if not os.path.isdir(experiment_results_dir):
        logs_results_dir = os.path.join(
            base_folder, "logs", "0-run", "experiment_results"
        )
        if os.path.isdir(logs_results_dir):
            try:
                os.symlink(logs_results_dir, experiment_results_dir)
            except OSError:
                shutil.copytree(
                    logs_results_dir, experiment_results_dir, dirs_exist_ok=True
                )

    # Clean up previous files
    if os.path.exists(aggregator_script_path):
        os.remove(aggregator_script_path)
    if os.path.exists(figures_dir):
        shutil.rmtree(figures_dir)
        print(f"Cleaned up previous figures directory")
    if os.path.exists(best_figures_dir):
        shutil.rmtree(best_figures_dir)

    idea_text = load_idea_text(base_folder)
    exp_summaries = load_exp_summaries(base_folder)
    filtered_summaries_for_plot_agg = filter_experiment_summaries(
        exp_summaries, step_name="plot_aggregation"
    )
    write_plot_agg_manifest(base_folder, filtered_summaries_for_plot_agg)
    sanitized_summaries_for_plot_agg = scrub_plot_paths(
        filtered_summaries_for_plot_agg
    )
    # Convert them to one big JSON string for context
    combined_summaries_str = json.dumps(sanitized_summaries_for_plot_agg, indent=2)

    # Build aggregator prompt
    aggregator_prompt = build_aggregator_prompt(combined_summaries_str, idea_text)

    # Call LLM
    client, model_name = create_client(model)
    response, msg_history = None, []
    system_message = build_aggregator_system_message(max_figures)
    try:
        response, msg_history = get_response_from_llm(
            prompt=aggregator_prompt,
            client=client,
            model=model_name,
            system_message=system_message,
            print_debug=False,
            msg_history=msg_history,
            max_tokens=max_tokens,
        )
    except Exception:
        traceback.print_exc()
        print("Failed to get aggregator script from LLM.")
        return

    aggregator_code, compile_error = select_compilable_code(response)
    if not aggregator_code.strip():
        print(
            "No Python code block was found in LLM response. Full response:\n", response
        )
        if compile_error:
            print(compile_error)
        return

    best_script = ""
    best_count = -1
    best_out = ""
    last_candidate_count = 0

    def save_best_figures() -> None:
        try:
            if os.path.exists(best_figures_dir):
                shutil.rmtree(best_figures_dir)
            if os.path.isdir(figures_dir):
                shutil.copytree(figures_dir, best_figures_dir)
        except Exception as exc:
            print(f"[yellow]Warning:[/yellow] Failed to save best figures: {exc}")

    def restore_best_figures() -> None:
        if best_count < 0:
            return
        try:
            if os.path.exists(figures_dir):
                shutil.rmtree(figures_dir)
            if os.path.isdir(best_figures_dir):
                shutil.copytree(best_figures_dir, figures_dir)
            if best_script:
                with open(aggregator_script_path, "w") as f:
                    f.write(best_script)
        except Exception as exc:
            print(f"[yellow]Warning:[/yellow] Failed to restore best figures: {exc}")

    def run_candidate(code: str, label: str) -> tuple[str, int, bool]:
        nonlocal best_script, best_count, best_out, last_candidate_count
        try:
            compile(code, "<auto_plot_aggregator>", "exec")
        except Exception:
            compile_trace = traceback.format_exc()
            last_candidate_count = 0
            return f"Compile error in {label}:\n{compile_trace}", 0, False

        if os.path.exists(figures_dir):
            shutil.rmtree(figures_dir)
        os.makedirs(figures_dir, exist_ok=True)

        candidate_out = run_aggregator_script(
            code, aggregator_script_path, base_folder, filename
        )
        candidate_count = count_figure_files(figures_dir)
        last_candidate_count = candidate_count

        if candidate_count >= best_count:
            best_script = code
            best_count = candidate_count
            best_out = candidate_out
            save_best_figures()
            return candidate_out, candidate_count, True

        candidate_out = (
            f"{candidate_out}\n\n[Rejected] Candidate produced {candidate_count} "
            f"figure(s), which is below best {best_count}. Restored best figures."
        )
        restore_best_figures()
        return candidate_out, candidate_count, False

    # First run of aggregator script
    aggregator_out, last_candidate_count, _ = run_candidate(aggregator_code, "initial")

    # Multiple reflection loops
    for i in range(n_reflections):
        figure_count = last_candidate_count
        print(f"[{i + 1} / {n_reflections}]: Number of figures: {figure_count}")
        # Reflection prompt with reminder for common checks and early exit
        reflection_prompt = render_prompt(
            "perform_plotting.aggregator_reflection_prompt",
            figure_count=figure_count,
            aggregator_out=aggregator_out,
            best_count=best_count,
            max_figures=max_figures,
        )

        print("[green]Reflection prompt:[/green] ", reflection_prompt)
        try:
            reflection_response, msg_history = get_response_from_llm(
                prompt=reflection_prompt,
                client=client,
                model=model_name,
                system_message=system_message,
                print_debug=False,
                msg_history=msg_history,
                max_tokens=max_tokens,
            )

        except Exception:
            traceback.print_exc()
            print("Failed to get reflection from LLM.")
            return

        # Early-exit check
        if best_count >= 0 and "I am done" in reflection_response:
            print("LLM indicated it is done with reflections. Exiting reflection loop.")
            break

        aggregator_new_code, compile_error = select_compilable_code(reflection_response)
        if not aggregator_new_code.strip():
            if "I am done" in reflection_response:
                print("LLM indicated it is done with reflections. Exiting reflection loop.")
                break
            print(
                f"No new aggregator script was provided. Reflection step {i+1} complete."
            )
            if compile_error:
                print(compile_error)
            continue

        # If new code is provided and differs, run again
        if (
            aggregator_new_code.strip()
            and aggregator_new_code.strip() != aggregator_code.strip()
        ):
            aggregator_code = aggregator_new_code
            aggregator_out, last_candidate_count, _ = run_candidate(
                aggregator_code, f"reflection {i + 1}"
            )
        else:
            print(
                f"No new aggregator script was provided or it was identical. Reflection step {i+1} complete."
            )

    if best_script:
        try:
            with open(aggregator_script_path, "w") as f:
                f.write(best_script)
        except Exception as exc:
            print(f"[yellow]Warning:[/yellow] Failed to write best script: {exc}")
    restore_best_figures()
    if os.path.exists(best_figures_dir):
        shutil.rmtree(best_figures_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Generate and execute a final plot aggregation script with LLM assistance."
    )
    parser.add_argument(
        "--folder",
        required=True,
        help="Path to the experiment folder with summary JSON files.",
    )
    parser.add_argument(
        "--model",
        default="o1-2024-12-17",
        help="LLM model to use (default: o1-2024-12-17).",
    )
    parser.add_argument(
        "--reflections",
        type=int,
        default=5,
        help="Number of reflection steps to attempt (default: 5).",
    )
    parser.add_argument(
        "--max_figures",
        type=int,
        default=DEFAULT_MAX_FIGURES,
        help=f"Maximum number of figures to target (default: {DEFAULT_MAX_FIGURES}).",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=None,
        help="Max output tokens for the plot-aggregation LLM (default: model/provider default).",
    )
    args = parser.parse_args()
    aggregate_plots(
        base_folder=args.folder,
        model=args.model,
        n_reflections=args.reflections,
        max_figures=args.max_figures,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()
