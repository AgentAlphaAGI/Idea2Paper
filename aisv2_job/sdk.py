import json
import subprocess
import sys
from typing import Any


def run_job_subprocess(
    *,
    load_ideas: str,
    idea_idx: int,
    attempt_id: int = 0,
    load_code: bool = False,
    add_dataset_ref: bool = False,
    bfts_config: str | None = None,
    model_agg_plots: str = "o3-mini-2025-01-31",
    out_root: str | None = None,
    run_id: str | None = None,
    strict_artifacts: bool = False,
    cwd: str | None = None,
    python_executable: str | None = None,
) -> dict[str, Any]:
    exe = python_executable or sys.executable
    cmd = [exe, "-m", "aisv2_job", "run", "--load_ideas", load_ideas, "--idea_idx", str(idea_idx)]

    cmd.extend(["--attempt_id", str(attempt_id)])

    if load_code:
        cmd.append("--load_code")
    if add_dataset_ref:
        cmd.append("--add_dataset_ref")
    if bfts_config:
        cmd.extend(["--bfts_config", bfts_config])
    if model_agg_plots:
        cmd.extend(["--model_agg_plots", model_agg_plots])
    if out_root:
        cmd.extend(["--out_root", out_root])
    if run_id:
        cmd.extend(["--run_id", run_id])
    if strict_artifacts:
        cmd.append("--strict_artifacts")

    result = subprocess.run(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    stdout = result.stdout.strip()
    stderr = result.stderr.strip()

    if stdout:
        try:
            return json.loads(stdout)
        except json.JSONDecodeError:
            pass

    raise RuntimeError(
        "Failed to parse manifest JSON from stdout. "
        f"exit_code={result.returncode}, stdout={stdout!r}, stderr={stderr!r}"
    )
