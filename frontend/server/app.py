from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import uuid
from datetime import datetime
from http import HTTPStatus
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse

from stage_mapper import infer_stage
from log_zipper import make_zip
from run_registry import RunRegistry

REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_SCRIPT = REPO_ROOT / "Paper-KG-Pipeline" / "scripts" / "idea2story_pipeline.py"
LOG_ROOT = REPO_ROOT / "log"
RESULTS_ROOT = REPO_ROOT / "results"
OUTPUT_ROOT = REPO_ROOT / "Paper-KG-Pipeline" / "output"
WEB_ROOT = REPO_ROOT / "frontend" / "web"
TMP_ROOT = REPO_ROOT / "frontend" / "server" / ".tmp"

registry = RunRegistry(REPO_ROOT)

def _file_mtime(path: Path):
    try:
        if path.exists():
            return path.stat().st_mtime
    except Exception:
        return None
    return None

def _activity_snapshot(log_dir: Path, active_window_sec: int = 120) -> dict:
    now = time.time()
    llm_path = log_dir / "llm_calls.jsonl"
    emb_path = log_dir / "embedding_calls.jsonl"
    llm_ts = _file_mtime(llm_path)
    emb_ts = _file_mtime(emb_path)
    llm_active = llm_ts is not None and (now - llm_ts) <= active_window_sec
    emb_active = emb_ts is not None and (now - emb_ts) <= active_window_sec
    return {
        "llm_active": llm_active,
        "embedding_active": emb_active,
        "last_llm_ts": llm_ts,
        "last_embedding_ts": emb_ts,
    }


def _now_iso():
    return datetime.utcnow().isoformat() + "Z"


def _json_response(handler: BaseHTTPRequestHandler, data: dict, status: int = 200):
    body = json.dumps(data, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _read_json(handler: BaseHTTPRequestHandler):
    length = int(handler.headers.get("Content-Length", "0"))
    raw = handler.rfile.read(length) if length else b"{}"
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return None


def _safe_env_meta(env: dict) -> dict:
    redacted = {k: v for k, v in env.items() if k != "SILICONFLOW_API_KEY"}
    if "SILICONFLOW_API_KEY" in env:
        redacted["SILICONFLOW_API_KEY"] = "***redacted***"
    return redacted


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Quiet default HTTP logs
        return

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/api/health":
            return _json_response(self, {"ok": True})

        if path.startswith("/api/runs/"):
            parts = path.strip("/").split("/")
            # /api/runs/<ui_run_id>[/result|logs.zip]
            if len(parts) < 3:
                return _json_response(self, {"ok": False, "error": "invalid runs path"}, status=400)
            ui_run_id = parts[2]
            if len(parts) >= 4 and parts[3] == "result":
                return self._handle_result(ui_run_id)
            if len(parts) >= 4 and parts[3] == "logs.zip":
                return self._handle_logs(ui_run_id)
            return self._handle_status(ui_run_id)

        # serve static
        if path == "/":
            path = "/index.html"
        file_path = (WEB_ROOT / path.lstrip("/")).resolve()
        if not str(file_path).startswith(str(WEB_ROOT.resolve())):
            return _json_response(self, {"ok": False, "error": "invalid path"}, status=400)
        if not file_path.exists() or file_path.is_dir():
            self.send_response(404)
            self.end_headers()
            return
        content_type = "text/plain"
        if file_path.suffix == ".html":
            content_type = "text/html; charset=utf-8"
        elif file_path.suffix == ".css":
            content_type = "text/css; charset=utf-8"
        elif file_path.suffix == ".js":
            content_type = "application/javascript; charset=utf-8"
        elif file_path.suffix in (".png", ".jpg", ".jpeg", ".svg", ".ico"):
            content_type = "image/" + file_path.suffix.lstrip(".")
        data = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/runs":
            return self._handle_run()
        return _json_response(self, {"ok": False, "error": "not found"}, status=404)

    def _handle_run(self):
        payload = _read_json(self)
        if not payload or "idea" not in payload:
            return _json_response(self, {"ok": False, "error": "invalid payload"}, status=400)

        idea = payload.get("idea", "").strip()
        llm = payload.get("llm", {}) or {}
        toggles = payload.get("toggles", {}) or {}

        ui_run_id = f"ui_{int(time.time())}_{uuid.uuid4().hex[:6]}"

        env = os.environ.copy()
        api_key = llm.get("api_key")
        if api_key:
            env["SILICONFLOW_API_KEY"] = api_key
        if llm.get("api_url"):
            env["LLM_API_URL"] = llm.get("api_url")
        if llm.get("model"):
            env["LLM_MODEL"] = llm.get("model")

        if "novelty" in toggles:
            env["I2P_NOVELTY_ENABLE"] = "1" if toggles.get("novelty") else "0"
        if "verification" in toggles:
            env["I2P_VERIFICATION_ENABLE"] = "1" if toggles.get("verification") else "0"

        # ensure logging/results for UI
        env["I2P_ENABLE_LOGGING"] = "1"
        env["I2P_RESULTS_ENABLE"] = "1"

        cmd = ["python", str(PIPELINE_SCRIPT), idea]
        try:
            popen = subprocess.Popen(
                cmd,
                cwd=str(REPO_ROOT),
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            return _json_response(self, {"ok": False, "error": str(e)}, status=500)

        info = registry.create(ui_run_id, popen, _safe_env_meta(env))
        return _json_response(self, {"ok": True, "pid": info.pid, "ui_run_id": ui_run_id, "run_id": None})

    def _handle_status(self, ui_run_id: str):
        info = registry.get(ui_run_id)
        if not info:
            return _json_response(self, {"ok": False, "error": "run not found"}, status=404)

        # resolve run_id
        if not info.run_id:
            rid = registry.resolve_run_id(info.pid)
            if rid:
                registry.update_run_id(info, rid)

        registry.refresh_status(info)
        log_dir = (LOG_ROOT / info.run_id) if info.run_id else None
        events_path = (log_dir / "events.jsonl") if log_dir and log_dir.exists() else None
        stage = infer_stage(events_path, info.status) if events_path else {"name": "Initializing", "progress": 0.05, "detail": "Waiting for log directory..."}
        if log_dir and log_dir.exists():
            stage["activity"] = _activity_snapshot(log_dir)
        else:
            stage["activity"] = {
                "llm_active": False,
                "embedding_active": False,
                "last_llm_ts": None,
                "last_embedding_ts": None,
            }

        results_dir = (RESULTS_ROOT / info.run_id) if info.run_id else None
        if results_dir and not results_dir.exists():
            results_dir = None

        return _json_response(self, {
            "ok": True,
            "ui_run_id": info.ui_run_id,
            "pid": info.pid,
            "run_id": info.run_id,
            "status": info.status,
            "stage": stage,
            "started_at": info.started_at,
            "updated_at": info.updated_at,
            "exit_code": info.exit_code,
            "paths": {
                "log_dir": str(log_dir) if log_dir and log_dir.exists() else None,
                "results_dir": str(results_dir) if results_dir else None,
            }
        })

    def _handle_result(self, ui_run_id: str):
        info = registry.get(ui_run_id)
        if not info:
            return _json_response(self, {"ok": False, "error": "run not found"}, status=404)

        run_id = info.run_id
        result_path = None
        final_path = None
        if run_id:
            rp = RESULTS_ROOT / run_id / "pipeline_result.json"
            fp = RESULTS_ROOT / run_id / "final_story.json"
            if rp.exists():
                result_path = rp
            if fp.exists():
                final_path = fp

        if result_path is None:
            # fallback to output
            rp = OUTPUT_ROOT / "pipeline_result.json"
            if rp.exists():
                result_path = rp
        if final_path is None:
            fp = OUTPUT_ROOT / "final_story.json"
            if fp.exists():
                final_path = fp

        pipeline_result = None
        final_story = None
        if result_path:
            try:
                pipeline_result = json.loads(result_path.read_text("utf-8"))
            except Exception:
                pipeline_result = None
        if final_path:
            try:
                final_story = json.loads(final_path.read_text("utf-8"))
            except Exception:
                final_story = None

        summary = {
            "success": None,
            "avg_score": None,
            "verification": {"collision_detected": None, "max_similarity": None},
            "novelty": {"risk_level": None, "max_similarity": None},
        }
        if pipeline_result:
            summary["success"] = pipeline_result.get("success")
            review_summary = pipeline_result.get("review_summary") or {}
            summary["avg_score"] = review_summary.get("final_score")
            verification_summary = pipeline_result.get("verification_summary") or {}
            summary["verification"] = {
                "collision_detected": verification_summary.get("collision_detected"),
                "max_similarity": verification_summary.get("max_similarity"),
            }
            novelty_report = pipeline_result.get("novelty_report") or {}
            summary["novelty"] = {
                "risk_level": novelty_report.get("risk_level"),
                "max_similarity": novelty_report.get("max_similarity"),
            }

        return _json_response(self, {
            "ok": True,
            "run_id": run_id,
            "final_story": final_story,
            "pipeline_result": pipeline_result,
            "summary": summary,
        })

    def _handle_logs(self, ui_run_id: str):
        info = registry.get(ui_run_id)
        if not info or not info.run_id:
            return _json_response(self, {"ok": False, "error": "run_id not ready"}, status=404)

        log_dir = LOG_ROOT / info.run_id
        if not log_dir.exists():
            return _json_response(self, {"ok": False, "error": "log dir missing"}, status=404)

        TMP_ROOT.mkdir(parents=True, exist_ok=True)
        zip_path = TMP_ROOT / f"{info.run_id}.zip"
        try:
            make_zip(log_dir, zip_path)
            data = zip_path.read_bytes()
        except Exception as e:
            return _json_response(self, {"ok": False, "error": str(e)}, status=500)

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/zip")
        self.send_header("Content-Disposition", f"attachment; filename={info.run_id}_logs.zip")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Frontend server running: http://{args.host}:{args.port}/")
    server.serve_forever()


if __name__ == "__main__":
    main()
