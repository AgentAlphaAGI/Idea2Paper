import json
from pathlib import Path
from typing import Any


def _path_or_none(path: Path | None) -> str | None:
    if path is None:
        return None
    if path.exists():
        return str(path)
    return None


def _list_files(path: Path | None) -> list[str]:
    if path is None or not path.exists():
        return []
    files = [p for p in path.iterdir() if p.is_file()]
    return [str(p.resolve()) for p in sorted(files)]


def collect_paths(idea_dir: Path) -> dict:
    idea_dir = idea_dir.resolve()
    logs_run_dir = idea_dir / "logs" / "0-run"

    paths = {
        "idea_dir": str(idea_dir),
        "idea_md": _path_or_none(idea_dir / "idea.md"),
        "idea_json": _path_or_none(idea_dir / "idea.json"),
        "bfts_config_run": _path_or_none(idea_dir / "bfts_config.yaml"),
        "logs_run_dir": str(logs_run_dir),
        "unified_tree_viz_html": _path_or_none(logs_run_dir / "unified_tree_viz.html"),
        "summaries": {
            "draft": _path_or_none(logs_run_dir / "draft_summary.json"),
            "baseline": _path_or_none(logs_run_dir / "baseline_summary.json"),
            "research": _path_or_none(logs_run_dir / "research_summary.json"),
            "ablation": _path_or_none(logs_run_dir / "ablation_summary.json"),
        },
        "figures_dir": _path_or_none(idea_dir / "figures"),
        "auto_plot_aggregator_py": _path_or_none(
            idea_dir / "auto_plot_aggregator.py"
        ),
        "token_tracker": _path_or_none(idea_dir / "token_tracker.json"),
        "token_tracker_interactions": _path_or_none(
            idea_dir / "token_tracker_interactions.json"
        ),
    }

    return paths


def build_manifest(
    inputs: dict,
    idea_dir: Path | None,
    status: str,
    warnings: list[str],
    error: dict | None,
) -> dict[str, Any]:
    paths = {}
    figure_files: list[str] = []
    if idea_dir is not None:
        paths = collect_paths(idea_dir)
        figures_dir = Path(paths["figures_dir"]) if paths.get("figures_dir") else None
        figure_files = _list_files(figures_dir)

    return {
        "schema_version": "1.0",
        "status": status,
        "inputs": inputs,
        "paths": paths,
        "artifacts": {"figure_files": figure_files},
        "warnings": warnings,
        "error": error,
    }


def write_manifest(manifest_data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2, sort_keys=True)
