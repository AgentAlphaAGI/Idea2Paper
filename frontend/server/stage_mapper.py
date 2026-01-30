from __future__ import annotations

import json
from collections import deque
from datetime import datetime
import time
from pathlib import Path
from typing import Optional

STAGE_ORDER = [
    ("Recall", 0.15),
    ("Pattern Selection", 0.30),
    ("Story Generation", 0.45),
    ("Critic Review", 0.60),
    ("Refinement", 0.70),
    ("Novelty Check", 0.80),
    ("Verification", 0.88),
    ("Bundling", 0.95),
    ("Done", 1.0),
    ("Failed", 1.0),
]

EVENT_TO_STAGE = [
    ("run_error", "Failed"),
    ("run_end", "Done"),
    ("results_bundled", "Bundling"),
    ("verification_", "Verification"),
    ("verification_from_novelty", "Verification"),
    ("verification_skipped", "Verification"),
    ("novelty_", "Novelty Check"),
    ("novelty_check_done", "Novelty Check"),
    ("novelty_pivot_triggered", "Novelty Check"),
    ("critic_", "Critic Review"),
    ("critic_result", "Critic Review"),
    ("review_", "Critic Review"),
    ("iteration", "Refinement"),
    ("pattern_selected", "Pattern Selection"),
    ("recall_", "Recall"),
    ("recall_end", "Recall"),
    ("recall_start", "Recall"),
]

DETAILS = {
    "Recall": "Retrieving relevant ideas/patterns",
    "Pattern Selection": "Scoring and selecting patterns",
    "Story Generation": "Generating structured story",
    "Critic Review": "Multi-agent review in progress",
    "Refinement": "Applying refinement loop",
    "Novelty Check": "Checking novelty / similarity",
    "Verification": "Final collision verification",
    "Bundling": "Bundling results",
    "Done": "Run completed",
    "Failed": "Run failed",
}

def _extract_event_type(ev: dict) -> Optional[str]:
    """
    Supports both formats:
    1) Envelope: {"type":"event","data":{"event_type":"recall_start","payload":{...}}}
    2) Flat: {"event_type":"recall_start", ...}
    """
    if not isinstance(ev, dict):
        return None
    data = ev.get("data")
    if isinstance(data, dict):
        ev_type = data.get("event_type") or data.get("type") or data.get("event")
        if ev_type:
            return ev_type
        # fallback to outer only if it carries a real event_type (avoid outer type="event")
        return ev.get("event_type") or ev.get("event")
    return ev.get("event_type") or ev.get("type") or ev.get("event")

def _extract_event_ts(ev: dict) -> Optional[str]:
    if not isinstance(ev, dict):
        return None
    # logs use outer ts
    ts = ev.get("ts")
    if ts:
        return ts
    data = ev.get("data")
    if isinstance(data, dict):
        return data.get("ts")
    return None

def _parse_ts_to_epoch(ts: Optional[str]) -> Optional[float]:
    if not ts:
        return None
    try:
        # handles ISO with timezone like 2026-01-30T18:19:33.078521+00:00
        return datetime.fromisoformat(ts).timestamp()
    except Exception:
        return None


def _progress_for(stage: str) -> float:
    for name, prog in STAGE_ORDER:
        if name == stage:
            return prog
    return 0.05


def _read_last_events(path: Path, max_lines: int = 200):
    if not path.exists():
        return []
    dq = deque(maxlen=max_lines)
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                dq.append(line)
    except Exception:
        return []
    events = []
    for line in dq:
        try:
            events.append(json.loads(line))
        except Exception:
            continue
    return events


def infer_stage(events_path: Path, process_status: str) -> dict:
    events = _read_last_events(events_path)
    stage = "Initializing"
    if process_status == "failed":
        stage = "Failed"
    elif process_status == "done":
        stage = "Done"

    latest_event_type = None
    latest_event_ts = None
    # look for latest matching event
    if events:
        for ev in reversed(events):
            ev_type = _extract_event_type(ev)
            if not ev_type:
                continue
            latest_event_type = ev_type
            latest_event_ts = _extract_event_ts(ev)
            for key, name in EVENT_TO_STAGE:
                if key.endswith("_"):
                    if ev_type.startswith(key):
                        stage = name
                        break
                elif ev_type == key:
                    stage = name
                    break
            else:
                continue
            break

    idle_seconds = None
    ts_epoch = _parse_ts_to_epoch(latest_event_ts)
    if ts_epoch is not None:
        idle_seconds = max(0.0, time.time() - ts_epoch)

    if stage == "Initializing" and process_status in ("starting", "running"):
        detail = "Starting pipeline..."
        return {
            "name": stage,
            "progress": 0.05,
            "detail": detail,
            "substage_event_type": latest_event_type,
            "last_event_ts": latest_event_ts,
            "idle_seconds": idle_seconds,
        }

    detail = DETAILS.get(stage, "Running")
    if latest_event_type:
        detail = f"{detail} · Latest event: {latest_event_type}"
    return {
        "name": stage,
        "progress": _progress_for(stage),
        "detail": detail,
        "substage_event_type": latest_event_type,
        "last_event_ts": latest_event_ts,
        "idle_seconds": idle_seconds,
    }
