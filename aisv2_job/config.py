from pathlib import Path
from typing import Any

import yaml


def load_job_config(config_path: Path | None, root_dir: Path) -> dict[str, Any]:
    if config_path is None:
        config_path = root_dir / "job_config.yaml"
        if not config_path.exists():
            return {"job": {}, "env": {}}
    else:
        config_path = Path(config_path).expanduser()
        if not config_path.exists():
            return {"job": {}, "env": {}}

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        data = {}

    job_data = data.get("job") or {}
    env_data = data.get("env") or {}

    if not isinstance(job_data, dict):
        job_data = {}
    if not isinstance(env_data, dict):
        env_data = {}

    return {"job": job_data, "env": env_data}
