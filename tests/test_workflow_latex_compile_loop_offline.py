# 中文注释: 测试全量编译失败触发 LaTeX 格式重试闭环。

import json
import random
from pathlib import Path

from api.paper_cli import _build_vector_store, build_mock_kg
from core.config import PaperWorkflowConfig
from paper_kg.embedding_mock import MockEmbeddingModel
from paper_kg.graph_networkx import NetworkXGraphStore
from paper_kg.input_parser import PaperInputParser
from paper_kg.latex_toolchain import CompileResult
from paper_kg.orchestrator import PaperDraftWriter
from workflow.paper_controller import PaperFeedbackLoopController


class _FakeLatexToolchain:
    def __init__(self) -> None:
        self.calls = 0

    def compile_project(self, project_dir: Path, engine: str, timeout_sec: int) -> CompileResult:
        self.calls += 1
        if self.calls == 1:
            return CompileResult(
                ok=False,
                stdout="",
                stderr="Undefined control sequence",
                log_excerpt="Undefined control sequence",
                errors=[],
                cmd=["latexmk"],
            )
        return CompileResult(
            ok=True,
            stdout="",
            stderr="",
            log_excerpt="",
            errors=[],
            cmd=["latexmk"],
        )


def _build_controller(tmp_path) -> PaperFeedbackLoopController:
    snapshot_path = tmp_path / "paper_kg_snapshot.json"
    build_mock_kg(str(snapshot_path), raw_path=None)

    graph = NetworkXGraphStore()
    graph.load_dict(json.loads(snapshot_path.read_text(encoding="utf-8")))

    embedding_model = MockEmbeddingModel()
    vector_store = _build_vector_store(graph, embedding_model)

    config = PaperWorkflowConfig(
        seed=123,
        output_format="latex",
        latex_require_tools=True,
        latex_format_max_retries=2,
    )

    return PaperFeedbackLoopController(
        config=config,
        graph_store=graph,
        vector_store=vector_store,
        embedding_model=embedding_model,
        rng=random.Random(123),
        input_parser=PaperInputParser(llm=None),
        draft_writer=PaperDraftWriter(llm=None),
        section_llm=None,
        latex_toolchain=_FakeLatexToolchain(),
    )


def test_compile_overleaf_triggers_render_latex_retry(tmp_path) -> None:
    controller = _build_controller(tmp_path)
    state = controller._init_state("test")  # noqa: SLF001
    state["overleaf_project_dir"] = str(tmp_path)

    updates = controller._compile_overleaf_node(state)  # noqa: SLF001
    assert updates.get("next_step") == "render_latex"
    state.update(updates)

    updates = controller._compile_overleaf_node(state)  # noqa: SLF001
    report = updates.get("latex_compile_report", {})
    assert report.get("ok") is True
    assert updates.get("next_step") != "render_latex"
