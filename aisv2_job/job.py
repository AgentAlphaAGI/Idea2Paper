import json
import os
import shutil
import sys
import traceback
from datetime import datetime
from pathlib import Path

from . import manifest


def _get_root_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_vendor_imports() -> Path:
    root = _get_root_dir()
    root_str = str(root)
    if sys.path[0] != root_str:
        if root_str in sys.path:
            sys.path.remove(root_str)
        sys.path.insert(0, root_str)
    os.environ["AI_SCIENTIST_ROOT"] = root_str
    return root


def _read_text(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _save_token_tracker(idea_dir: Path) -> None:
    from ai_scientist.utils.token_tracker import token_tracker

    with open(idea_dir / "token_tracker.json", "w", encoding="utf-8") as f:
        json.dump(token_tracker.get_summary(), f)
    with open(
        idea_dir / "token_tracker_interactions.json", "w", encoding="utf-8"
    ) as f:
        json.dump(token_tracker.get_interactions(), f)


def _resolve_path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _build_inputs(args, root_dir: Path) -> dict:
    return {
        "load_ideas": str(_resolve_path(args.load_ideas)),
        "idea_idx": int(args.idea_idx),
        "attempt_id": int(args.attempt_id),
        "load_code": bool(args.load_code),
        "add_dataset_ref": bool(args.add_dataset_ref),
        "bfts_config": str(_resolve_path(args.bfts_config)),
        "model_agg_plots": str(args.model_agg_plots),
        "plot_agg_reflections": getattr(args, "plot_agg_reflections", None),
        "plot_agg_max_figures": getattr(args, "plot_agg_max_figures", None),
        "plot_agg_max_tokens": getattr(args, "plot_agg_max_tokens", None),
        "keep_experiment_results": getattr(args, "keep_experiment_results", None),
        "out_root": str(_resolve_path(args.out_root)),
        "run_id": args.run_id,
        "strict_artifacts": bool(args.strict_artifacts),
        "ai_scientist_root": str(root_dir),
    }


def _check_strict_artifacts(paths_dict: dict) -> list[str]:
    missing = []
    summaries = paths_dict.get("summaries", {})
    for key in ("draft", "baseline", "research", "ablation"):
        if not summaries.get(key):
            missing.append(f"summary:{key}")

    figures_dir = paths_dict.get("figures_dir")
    if not figures_dir:
        missing.append("figures_dir")
    else:
        fig_dir_path = Path(figures_dir)
        if not fig_dir_path.exists() or not any(fig_dir_path.iterdir()):
            missing.append("figures_dir_empty")

    return missing


def run_experiments_only(args) -> tuple[dict, int]:
    root_dir = _ensure_vendor_imports()

    inputs = _build_inputs(args, root_dir)
    warnings = []
    status = "success"
    error = None
    exit_code = 0

    idea_dir = None
    out_root = _resolve_path(args.out_root)

    try:
        load_ideas_path = _resolve_path(args.load_ideas)
        bfts_config_path = _resolve_path(args.bfts_config)

        with open(load_ideas_path, "r", encoding="utf-8") as f:
            ideas = json.load(f)

        if args.idea_idx < 0 or args.idea_idx >= len(ideas):
            raise IndexError(
                f"idea_idx {args.idea_idx} out of range for {len(ideas)} ideas"
            )

        idea = ideas[args.idea_idx]

        if args.run_id:
            date_token = args.run_id
        else:
            date_token = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        idea_dir = (
            out_root
            / "experiments"
            / f"{date_token}_{idea['Name']}_attempt_{args.attempt_id}"
        )
        idea_dir.mkdir(parents=True, exist_ok=True)

        idea_path_md = idea_dir / "idea.md"

        code = None
        code_path = None
        if args.load_code:
            code_path = str(load_ideas_path).rsplit(".", 1)[0] + ".py"
            if Path(code_path).exists():
                code = _read_text(Path(code_path))
            else:
                warnings.append(f"Code file not found: {code_path}")
        else:
            code_path = None

        from ai_scientist.treesearch.bfts_utils import idea_to_markdown

        idea_to_markdown(ideas[args.idea_idx], str(idea_path_md), code_path)

        dataset_ref_code = None
        if args.add_dataset_ref:
            dataset_ref_path = Path.cwd() / "hf_dataset_reference.py"
            if dataset_ref_path.exists():
                dataset_ref_code = _read_text(dataset_ref_path)
            else:
                warnings.append(
                    f"Dataset reference file not found: {dataset_ref_path}"
                )
                dataset_ref_code = None

        if dataset_ref_code is not None and code is not None:
            added_code = dataset_ref_code + "\n" + code
        elif dataset_ref_code is not None and code is None:
            added_code = dataset_ref_code
        elif dataset_ref_code is None and code is not None:
            added_code = code
        else:
            added_code = None

        if added_code is not None:
            ideas[args.idea_idx]["Code"] = added_code

        idea_path_json = idea_dir / "idea.json"
        with open(idea_path_json, "w", encoding="utf-8") as f:
            json.dump(ideas[args.idea_idx], f, indent=4)

        from ai_scientist.treesearch.bfts_utils import edit_bfts_config_file

        idea_config_path = edit_bfts_config_file(
            str(bfts_config_path),
            str(idea_dir),
            str(idea_path_json),
        )

        from ai_scientist.treesearch.perform_experiments_bfts_with_agentmanager import (
            perform_experiments_bfts,
        )

        perform_experiments_bfts(idea_config_path)

        experiment_results_dir = idea_dir / "logs" / "0-run" / "experiment_results"
        if experiment_results_dir.exists():
            staging_dir = idea_dir / "experiment_results"
            if staging_dir.exists() or staging_dir.is_symlink():
                if staging_dir.is_symlink() or staging_dir.is_file():
                    staging_dir.unlink()
                else:
                    shutil.rmtree(staging_dir)
            try:
                staging_dir.symlink_to(
                    experiment_results_dir, target_is_directory=True
                )
            except OSError:
                shutil.copytree(
                    experiment_results_dir,
                    staging_dir,
                    dirs_exist_ok=True,
                )

        from ai_scientist.perform_plotting import aggregate_plots

        plot_agg_max_tokens = getattr(args, "plot_agg_max_tokens", None)
        if isinstance(plot_agg_max_tokens, int) and plot_agg_max_tokens <= 0:
            plot_agg_max_tokens = None

        aggregate_plots(
            base_folder=str(idea_dir),
            model=args.model_agg_plots,
            n_reflections=int(getattr(args, "plot_agg_reflections", 5)),
            max_figures=int(getattr(args, "plot_agg_max_figures", 12)),
            max_tokens=plot_agg_max_tokens,
        )

        if not bool(getattr(args, "keep_experiment_results", False)):
            staging_dir = idea_dir / "experiment_results"
            if staging_dir.exists() or staging_dir.is_symlink():
                if staging_dir.is_symlink() or staging_dir.is_file():
                    staging_dir.unlink()
                else:
                    shutil.rmtree(staging_dir)

        _save_token_tracker(idea_dir)

    except Exception as exc:
        status = "failed"
        error = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        exit_code = 1

    manifest_data = manifest.build_manifest(
        inputs=inputs,
        idea_dir=idea_dir,
        status=status,
        warnings=warnings,
        error=error,
    )

    if status == "success":
        missing = _check_strict_artifacts(manifest_data.get("paths", {}))
        if missing:
            warning_msg = "Missing artifacts: " + ", ".join(missing)
            manifest_data["warnings"].append(warning_msg)
            if args.strict_artifacts:
                manifest_data["status"] = "failed"
                manifest_data["error"] = {
                    "type": "MissingArtifacts",
                    "message": warning_msg,
                    "traceback": "",
                }
                exit_code = 2

    manifest_path = None
    if idea_dir is not None:
        manifest_path = idea_dir / "run_manifest.json"
    else:
        manifest_path = out_root / "run_manifest.json"
        manifest_data["warnings"].append(
            "idea_dir was not created; manifest written to out_root"
        )

    manifest.write_manifest(manifest_data, manifest_path)

    return manifest_data, exit_code
