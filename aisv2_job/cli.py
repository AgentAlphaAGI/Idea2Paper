import argparse
import json
import sys
from pathlib import Path

from . import job
from . import config as job_config


PATH_KEYS = {"load_ideas", "bfts_config", "out_root"}
INT_KEYS = {
    "idea_idx",
    "attempt_id",
    "plot_agg_reflections",
    "plot_agg_max_figures",
    "plot_agg_max_tokens",
}


def _get_root_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def _prepare_vendor_imports() -> Path:
    root = _get_root_dir()
    root_str = str(root)
    if sys.path[0] != root_str:
        if root_str in sys.path:
            sys.path.remove(root_str)
        sys.path.insert(0, root_str)
    return root


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aisv2_job")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run experiments only")
    run_parser.add_argument("--config", type=str, default=None)
    run_parser.add_argument("--load_ideas", type=str, default=None)
    run_parser.add_argument("--idea_idx", type=int, default=None)
    run_parser.add_argument("--attempt_id", type=int, default=None)
    run_parser.add_argument("--load_code", action="store_true", default=None)
    run_parser.add_argument("--add_dataset_ref", action="store_true", default=None)
    run_parser.add_argument("--bfts_config", type=str, default=None)
    run_parser.add_argument("--model_agg_plots", type=str, default=None)
    run_parser.add_argument("--plot_agg_reflections", type=int, default=None)
    run_parser.add_argument("--plot_agg_max_figures", type=int, default=None)
    run_parser.add_argument("--plot_agg_max_tokens", type=int, default=None)
    run_parser.add_argument(
        "--keep_experiment_results", action="store_true", default=None
    )
    run_parser.add_argument("--out_root", type=str, default=None)
    run_parser.add_argument("--run_id", type=str, default=None)
    run_parser.add_argument("--strict_artifacts", action="store_true", default=None)

    return parser


def _resolve_config_path(value: str, root_dir: Path) -> str:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = root_dir / path
    return str(path.resolve())


def _coerce_int(value, name: str):
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be an integer")


def _merge_settings(args, config_data: dict, root_dir: Path) -> dict:
    defaults = {
        "load_ideas": None,
        "idea_idx": None,
        "attempt_id": 0,
        "load_code": False,
        "add_dataset_ref": False,
        "bfts_config": str(root_dir / "bfts_config.yaml"),
        "model_agg_plots": "o3-mini-2025-01-31",
        "plot_agg_reflections": 5,
        "plot_agg_max_figures": 12,
        "plot_agg_max_tokens": None,
        "keep_experiment_results": False,
        "out_root": str(Path.cwd()),
        "run_id": None,
        "strict_artifacts": False,
    }

    merged = defaults.copy()
    config_job = config_data.get("job", {}) if isinstance(config_data, dict) else {}

    if isinstance(config_job, dict):
        for key, value in config_job.items():
            if value is None:
                continue
            if key in PATH_KEYS:
                merged[key] = _resolve_config_path(value, root_dir)
            else:
                merged[key] = value

    for key in defaults:
        cli_value = getattr(args, key, None)
        if cli_value is not None:
            merged[key] = cli_value

    for key in INT_KEYS:
        merged[key] = _coerce_int(merged.get(key), key)

    return merged


def main(argv=None) -> int:
    root_dir = _prepare_vendor_imports()

    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        config_path = None
        if args.config:
            config_path = Path(args.config).expanduser()
        try:
            config_data = job_config.load_job_config(config_path, root_dir)
        except Exception as exc:
            print(f"Failed to load config: {exc}", file=sys.stderr)
            return 2

        try:
            merged = _merge_settings(args, config_data, root_dir)
        except Exception as exc:
            print(f"Invalid configuration: {exc}", file=sys.stderr)
            return 2

        if merged.get("load_ideas") is None or merged.get("idea_idx") is None:
            print(
                "load_ideas and idea_idx are required (use CLI or job_config.yaml).",
                file=sys.stderr,
            )
            return 2

        run_args = argparse.Namespace(**merged)
        manifest, exit_code = job.run_experiments_only(run_args)
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return exit_code

    parser.print_help()
    return 1
