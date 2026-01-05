# 中文注释: 本文件测试 Paper KG CLI 关键路径。
import json

from api.paper_cli import build_mock_kg, _build_vector_store
from paper_kg.graph_networkx import NetworkXGraphStore
from paper_kg.orchestrator import StoryOrchestrator
from paper_kg.embedding_mock import MockEmbeddingModel


def test_cli_build_and_generate(tmp_path) -> None:
    """
    功能：验证 mock KG 构建与故事生成流程。
    参数：tmp_path。
    返回：无。
    流程：生成快照 → 加载 → 生成故事。
    说明：不依赖真实网络。
    """
    snapshot_path = tmp_path / "paper_kg_snapshot.json"
    build_mock_kg(str(snapshot_path), raw_path=None)

    graph = NetworkXGraphStore()
    graph.load_dict(json.loads(snapshot_path.read_text(encoding="utf-8")))

    embedding_model = MockEmbeddingModel()
    vector_store = _build_vector_store(graph, embedding_model)
    orchestrator = StoryOrchestrator(
        graph_store=graph,
        vector_store=vector_store,
        embedding_model=embedding_model,
    )

    draft = orchestrator.build_story(
        idea_text="用 Diffusion 做推理",
        domain_name="NLP",
        max_patterns=2,
        temperature=0.8,
        novelty_threshold=0.85,
    )

    assert draft.patterns
    assert "问题定义" in draft.final_story
