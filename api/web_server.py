# 中文注释: 本文件提供一个可选的 Web UI 服务端（FastAPI + SSE），用于把论文链路每个节点/Agent 的结构化输出实时推送到前端。
from __future__ import annotations

import json
import random
import secrets
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from api.paper_cli import _build_vector_store
from api.paper_cli import _default_snapshot_path
from api.paper_cli import _format_final_paper_reply
from api.paper_cli import build_mock_kg
from core.config import load_config
from core.embeddings import create_embedding_model
from core.llm import create_chat_model, load_env
from core.logging import setup_logging
from paper_kg.graph_networkx import NetworkXGraphStore
from paper_kg.embedding_mock import MockEmbeddingModel
from paper_kg.input_parser import PaperInputParser
from paper_kg.latex_toolchain import LocalLatexToolchain
from paper_kg.orchestrator import PaperDraftWriter
from paper_kg.paper_template import load_paper_template
from paper_kg.vectorstore_memory import InMemoryVectorStore
from workflow.paper_controller import PaperFeedbackLoopController


class CreatePaperJobRequest(BaseModel):
    """
    功能：创建论文工作流 Job 的请求体。
    参数：
    - text：用户输入的一句自然语言 idea。
    - snapshot_path：知识图谱快照路径（默认小规模 paper_kg_snapshot.json）。
    - config_path：YAML 配置路径；为空时将自动尝试加载项目根目录的 config_paper.yaml。
    - use_llm：是否启用真实 LLM（默认 False，便于离线体验）。
    - use_mock_embedding：是否使用 Mock embedding（默认 False，离线测试用）。
    - seed：可选随机种子，便于复现同一次运行。
    返回：Pydantic 模型实例。
    说明：前端会把这些字段直接 JSON 提交到后端。
    """

    text: str = Field(..., description="用户输入的一句自然语言论文想法")
    snapshot_path: Optional[str] = Field(None, description="图谱快照路径（为空则自动选择）")
    config_path: Optional[str] = Field(None, description="YAML 配置路径（可选）")
    use_llm: bool = Field(False, description="是否使用真实 LLM")
    use_mock_embedding: bool = Field(False, description="是否使用 Mock embedding")
    seed: Optional[int] = Field(None, description="随机种子（可选）")


# 中文注释：简单的进程内 Job 存储（演示用途）。生产环境应替换为 Redis/DB 或队列系统。
JOBS: Dict[str, CreatePaperJobRequest] = {}


def _utc_now_iso() -> str:
    """
    功能：生成 UTC ISO 时间字符串。
    参数：无。
    返回：形如 2025-01-01T00:00:00Z 的字符串。
    流程：datetime.now(timezone.utc) → isoformat → Z 标准化。
    说明：用于 SSE 事件的 ts 字段，便于前端按时间展示。
    """
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sse_data(payload: Dict[str, Any]) -> str:
    """
    功能：把 JSON payload 打包为 SSE 的 data 行。
    参数：payload（可 JSON 序列化 dict）。
    返回：符合 SSE 协议的数据块字符串。
    流程：json.dumps → 拼接 data: ... \\n\\n。
    说明：
    - EventSource 默认监听 message 事件，这里只输出 data 字段即可。
    - ensure_ascii=False 保证中文可读。
    """
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _to_jsonable(obj: Any) -> Any:
    """
    功能：将复杂对象递归转换为 JSON 可序列化结构。
    参数：obj（任意对象）。
    返回：可被 json.dumps 序列化的对象（dict/list/str/number/...）。
    流程：
    - Pydantic 模型：model_dump
    - dict/list/tuple/set：递归转换
    - 其他未知类型：转为 str
    说明：用于把 LangGraph 节点输出（包含 Pydantic）安全地发送到前端。
    """
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:  # noqa: BLE001
            return str(obj)
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, set):
        return [_to_jsonable(v) for v in obj]
    return str(obj)


def _summarize_step(node: str, update: Dict[str, Any], state: Dict[str, Any]) -> str:
    """
    功能：为前端生成“每一步的摘要文案”，避免用户只看到大段 JSON。
    参数：
    - node：当前节点名（parse_input/plan_from_kg/...）。
    - update：该节点本次返回的增量更新 dict。
    - state：合并更新后的全局 state（可用于补齐字段）。
    返回：简短的中文摘要字符串。
    流程：按 node 分类提取关键字段 → 拼接摘要。
    说明：摘要用于对话区气泡与右侧卡片标题。
    """
    if node == "parse_input":
        return f"识别领域={state.get('domain','')}；核心想法={state.get('core_idea','')}"

    if node == "plan_from_kg":
        patterns = state.get("selected_patterns") or []
        tricks = state.get("selected_tricks") or []
        pattern_names = []
        for p in patterns:
            name = (
                getattr(p, "pattern_name", None)
                or (p.get("pattern_name") if isinstance(p, dict) else None)
                or getattr(p, "name", None)
                or (p.get("name") if isinstance(p, dict) else None)
            )
            if name:
                pattern_names.append(str(name))
        trick_names = []
        for t in tricks:
            name = getattr(t, "name", None) or (t.get("name") if isinstance(t, dict) else None)
            if name:
                trick_names.append(str(name))
        return f"选中套路={ '、'.join(pattern_names) }；技巧={ '、'.join(trick_names) }"

    if node == "retrieve_similar":
        count = len(state.get("novelty_candidates") or [])
        return f"查重检索返回 {count} 条相似历史配置"

    if node == "novelty_check":
        report = state.get("novelty_report")
        if report and hasattr(report, "is_novel"):
            return f"新颖性判定：is_novel={getattr(report,'is_novel',False)}；max_similarity={getattr(report,'max_similarity',0.0):.3f}"
        return "新颖性判定完成"

    if node == "build_blueprint":
        plan = state.get("paper_plan")
        sections = plan.sections if hasattr(plan, "sections") else plan.get("sections") if isinstance(plan, dict) else []
        return f"生成论文蓝图完成（sections={len(sections)}）"

    if node.startswith("write_"):
        section_id = node.replace("write_", "")
        sections = state.get("paper_sections") or []
        for section in sections:
            sid = getattr(section, "section_id", None) or (section.get("section_id") if isinstance(section, dict) else None)
            if sid == section_id:
                word_count = getattr(section, "word_count", None) or (section.get("word_count") if isinstance(section, dict) else None)
                return f"章节写作完成：{section_id}（words={word_count}）"
        return f"章节写作完成：{section_id}"

    if node == "assemble_paper":
        content = state.get("paper_latex") or state.get("paper_markdown") or ""
        return f"论文拼接完成（长度={len(content)}）"

    if node == "export_overleaf":
        project_dir = state.get("overleaf_project_dir") or ""
        zip_path = state.get("overleaf_zip_path") or ""
        return f"Overleaf 工程导出完成：dir={project_dir} zip={zip_path}"

    if node == "compile_overleaf":
        report = state.get("latex_compile_report") or {}
        ok = report.get("ok", False)
        return f"LaTeX 编译闸门完成：ok={ok}"

    if node == "write_draft":
        attempt = int(state.get("draft_attempts") or 0)
        return f"生成/改写草稿完成（draft_attempts={attempt}）"

    return f"完成节点：{node}"


# 中文注释：FastAPI 应用实例。
app = FastAPI(title="ad-agent web")

# 中文注释：项目根目录与静态资源目录定位。
BASE_DIR = Path(__file__).resolve().parents[1]
WEB_DIR = BASE_DIR / "webui"

# 中文注释：挂载静态目录，用于提供 styles.css/app.js 等文件。
if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


@app.get("/healthz")
def healthz() -> Dict[str, str]:
    """
    功能：健康检查接口。
    参数：无。
    返回：简单 JSON。
    流程：直接返回 ok。
    说明：便于容器/反向代理探活。
    """
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    """
    功能：返回前端首页。
    参数：无。
    返回：HTMLResponse。
    流程：读取 webui/index.html → 返回。
    说明：前端资源（CSS/JS）由 /static 提供。
    """
    index_path = WEB_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=500, detail="webui/index.html not found")
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


@app.post("/api/paper/jobs")
def create_paper_job(req: CreatePaperJobRequest) -> Dict[str, str]:
    """
    功能：创建一个论文工作流 Job。
    参数：req（CreatePaperJobRequest）。
    返回：{job_id: "..."}。
    流程：校验输入 → 生成 job_id → 写入内存 JOBS。
    说明：后续通过 SSE 接口消费该 job 并输出事件流。
    """
    text = str(req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text 不能为空")
    job_id = uuid.uuid4().hex
    JOBS[job_id] = req
    return {"job_id": job_id}


@app.get("/api/paper/jobs/{job_id}/events")
def paper_job_events(job_id: str) -> StreamingResponse:
    """
    功能：以 SSE 形式实时推送论文工作流每个节点的结构化输出。
    参数：job_id。
    返回：StreamingResponse（text/event-stream）。
    流程：
    1) 读取 JOBS[job_id]；
    2) 初始化 controller 与 state；
    3) 使用 controller.graph.stream(...) 逐节点拿到 update；
    4) 每步推送 step 事件；
    5) 结束推送 final 或 error，并清理 JOB。
    说明：这是“前端实时显示每个 Agent 输出”的关键接口。
    """
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="job_id 不存在或已完成")

    req = JOBS[job_id]

    def event_stream() -> Generator[str, None, None]:
        """
        功能：SSE 生成器。
        参数：无（闭包读取 req/job_id）。
        返回：逐段 SSE 文本。
        流程：运行 LangGraph stream → yield step/final/error。
        说明：生成器结束即视为 SSE 连接关闭。
        """
        seq = 0
        try:
            setup_logging()
            load_env()

            # 中文注释：自动加载默认配置文件（与 paper_cli.py 行为一致）。
            config_path = req.config_path
            if not config_path:
                default_cfg = BASE_DIR / "config_paper.yaml"
                if default_cfg.exists():
                    config_path = str(default_cfg)

            config = load_config(config_path)
            if req.seed is not None:
                config.paper_workflow.seed = int(req.seed)
            if config.paper_workflow.seed is None:
                config.paper_workflow.seed = secrets.randbits(32)

            snapshot_path = req.snapshot_path or _default_snapshot_path()
            snapshot = Path(snapshot_path)
            if not snapshot.exists():
                default_snapshot = _default_snapshot_path()
                if snapshot_path == default_snapshot and snapshot.name == "paper_kg_snapshot.json":
                    build_mock_kg(str(snapshot), raw_path=None)
                else:
                    raise FileNotFoundError(f"Snapshot not found: {snapshot}")

            graph_store = NetworkXGraphStore()
            graph_store.load_dict(json.loads(snapshot.read_text(encoding="utf-8")))

            if bool(config.paper_workflow.skip_novelty_check):
                logger.info("PAPER_SKIP_NOVELTY: skip embedding/vector_store build")
                embedding_model = MockEmbeddingModel()
                vector_store = InMemoryVectorStore()
            else:
                if req.use_mock_embedding:
                    embedding_model = MockEmbeddingModel()
                else:
                    embedding_model = create_embedding_model("paper_embedding", required=True)
                    if embedding_model is None:
                        raise ValueError("Embedding model is required, please set API key in .env")
                vector_store = _build_vector_store(graph_store, embedding_model)

            # 中文注释：按 agent 前缀创建模型；use_llm=False 时全部传 None，走规则/模板兜底。
            parse_llm = create_chat_model("paper_parse") if req.use_llm else None
            story_llm = create_chat_model("paper_story") if req.use_llm else None
            section_llm = create_chat_model("paper_section_writer") if req.use_llm else None

            input_parser = PaperInputParser(llm=parse_llm)
            draft_writer = PaperDraftWriter(
                llm=story_llm,
                guide_limit=int(config.paper_workflow.writing_guide_limit_chars),
                max_guides=int(config.paper_workflow.writing_guide_max_guides),
            )
            paper_template = load_paper_template(config.paper_workflow.template_path)

            # 中文注释：LaTeX 模式下初始化工具链（用于编译闸门）。
            latex_toolchain = None
            if str(config.paper_workflow.output_format) == "latex":
                latex_toolchain = LocalLatexToolchain(
                    require_tools=bool(config.paper_workflow.latex_require_tools),
                )

            controller = PaperFeedbackLoopController(
                config=config.paper_workflow,
                graph_store=graph_store,
                vector_store=vector_store,
                embedding_model=embedding_model,
                rng=random.Random(int(config.paper_workflow.seed)),
                input_parser=input_parser,
                draft_writer=draft_writer,
                section_llm=section_llm,
                paper_template=paper_template,
                latex_toolchain=latex_toolchain,
            )

            state = controller.init_state(str(req.text))

            recursion_limit = controller.estimate_recursion_limit()

            # 中文注释：先发一条注释，促使浏览器尽快建立 SSE 连接。
            yield ": connected\n\n"

            for chunk in controller.graph.stream(state, config={"recursion_limit": recursion_limit}):
                if not isinstance(chunk, dict):
                    continue
                for node, update in chunk.items():
                    if not isinstance(update, dict):
                        continue
                    state.update(update)  # type: ignore[arg-type]
                    seq += 1
                    payload = {
                        "type": "step",
                        "seq": seq,
                        "node": str(node),
                        "ts": _utc_now_iso(),
                        "data": _to_jsonable(update),
                        "summary": _summarize_step(str(node), update, state),
                    }
                    yield _sse_data(payload)

            report = controller.build_report_from_state(state)
            report_dict = report.to_dict()
            final_reply = _format_final_paper_reply(report_dict)
            yield _sse_data(
                {
                    "type": "final",
                    "ts": _utc_now_iso(),
                    "report": report_dict,
                    "final_reply": final_reply,
                }
            )
        except Exception as exc:  # noqa: BLE001
            yield _sse_data(
                {
                    "type": "error",
                    "ts": _utc_now_iso(),
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
        finally:
            # 中文注释：无论成功/失败都清理 job，避免内存泄露。
            JOBS.pop(job_id, None)

    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=headers)
