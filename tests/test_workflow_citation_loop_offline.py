# 中文注释: 测试引用检索/降级闭环（离线）。

import json
import random

from api.paper_cli import _build_vector_store, build_mock_kg
from core.config import PaperWorkflowConfig
from paper_kg.citations.models import RetrievedCitation
from paper_kg.embedding_mock import MockEmbeddingModel
from paper_kg.graph_networkx import NetworkXGraphStore
from paper_kg.input_parser import PaperInputParser
from paper_kg.models import PaperPlan, PaperSectionOutput, PaperSectionPlan
from paper_kg.orchestrator import PaperDraftWriter
from workflow.paper_controller import PaperFeedbackLoopController


class _NoopReviewerGroup:
    def review(self, _draft):  # noqa: ANN001
        return []


class _FakeCitationProvider:
    def search_and_resolve(self, need):  # noqa: ANN001
        return RetrievedCitation(candidate_id=need.candidate_id, status="NOT_FOUND")


def _fake_claim_resolver(sections, _candidates, unresolved):  # noqa: ANN001
    updated = []
    for section_id, markdown in sections:
        for cid in unresolved:
            markdown = markdown.replace(f"<!-- CITE_CANDIDATE:{cid} -->", "")
        updated.append((section_id, markdown))
    return updated


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
        latex_generation_mode="final_only",
        citations_enable=True,
        citations_max_rounds=2,
        format_only_mode=True,
        latex_require_tools=False,
        latex_final_compile=False,
    )

    return PaperFeedbackLoopController(
        config=config,
        graph_store=graph,
        vector_store=vector_store,
        embedding_model=embedding_model,
        rng=random.Random(123),
        input_parser=PaperInputParser(llm=None),
        draft_writer=PaperDraftWriter(llm=None),
        reviewer_group=_NoopReviewerGroup(),
        section_llm=None,
        citation_provider=_FakeCitationProvider(),
        claim_resolver=_fake_claim_resolver,
    )


def test_citation_loop_resolves_unresolved(tmp_path) -> None:
    controller = _build_controller(tmp_path)
    state = controller._init_state("test")  # noqa: SLF001

    plan = PaperPlan(
        template_name="test",
        sections=[PaperSectionPlan(section_id="introduction", title="引言")],
    )
    section = PaperSectionOutput(
        section_id="introduction",
        title="引言",
        content_markdown="# 引言\n\n已有研究表明该方法有效。<!-- CITE_CANDIDATE -->",
    )

    state["paper_plan"] = plan
    state["paper_sections"] = [section]

    state.update(controller._extract_cite_candidates_node(state))  # noqa: SLF001
    assert state["citation_candidates"]

    state.update(controller._citation_need_review_node(state))  # noqa: SLF001
    state.update(controller._citation_retrieval_node(state))  # noqa: SLF001
    assert state["unresolved_citations"]

    state.update(controller._claim_resolution_node(state))  # noqa: SLF001
    assert state["citation_rounds"] == 1
    assert state["next_step"] == "extract_cite_candidates"

    state.update(controller._extract_cite_candidates_node(state))  # noqa: SLF001
    assert not state["citation_candidates"]
