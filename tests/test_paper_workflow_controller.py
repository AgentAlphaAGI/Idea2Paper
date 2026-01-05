# 中文注释: 本文件测试论文 LangGraph 工作流的通过路径与两类回退路径（离线可跑）。

import json
import random
from typing import List

from api.paper_cli import _build_vector_store, build_mock_kg
from core.config import PaperWorkflowConfig
from paper_kg.graph_networkx import NetworkXGraphStore
from paper_kg.input_parser import PaperInputParser
from paper_kg.llm_schemas import PaperReviewDefect, PaperReviewScores
from paper_kg.orchestrator import PaperDraftWriter
from paper_kg.paper_template import load_paper_template
from paper_kg.paper_reviewers import PaperReviewResult
from paper_kg.embedding_mock import MockEmbeddingModel
from workflow.paper_controller import PaperFeedbackLoopController


def _make_review(
    reviewer_id: str,
    persona: str,
    overall: float,
    issue_type: str,
    defects: List[PaperReviewDefect] | None = None,
) -> PaperReviewResult:
    """
    功能：构造单条 PaperReviewResult（用于测试，不调用真实 LLM）。
    参数：reviewer_id、persona、overall、issue_type、defects。
    返回：PaperReviewResult。
    流程：填充 scores 与必要字段 → 返回。
    说明：scores 这里使用固定值，重点验证 controller 回退分支。
    """
    scores = PaperReviewScores(novelty=7.0, soundness=7.0, clarity=7.0, significance=7.0)
    return PaperReviewResult(
        reviewer_id=reviewer_id,
        persona=persona,
        threshold=7.0,
        overall_score=overall,
        scores=scores,
        strengths=["优点示例"],
        weaknesses=["缺点示例"],
        defects=defects or [],
        decision="borderline",
        issue_type=issue_type,
    )


class _StaticReviewerGroup:
    """
    功能：测试用 reviewer group，根据调用次数返回预设结果序列。
    参数：sequence（每次 review 返回的 reviews 列表）。
    返回：具备 review(draft) 方法的对象。
    流程：按 call_index 返回对应 reviews。
    说明：用于离线触发 PASS/回退路径。
    """

    def __init__(self, sequence: List[List[PaperReviewResult]]) -> None:
        self.sequence = sequence
        self.call_index = 0

    def review(self, _draft):  # noqa: ANN001
        idx = min(self.call_index, len(self.sequence) - 1)
        self.call_index += 1
        return self.sequence[idx]


def _build_controller(tmp_path, reviewer_group) -> PaperFeedbackLoopController:
    """
    功能：构造测试用 PaperFeedbackLoopController。
    参数：tmp_path、reviewer_group。
    返回：PaperFeedbackLoopController。
    流程：构建 mock KG → vector store → 初始化 controller（无 LLM）。
    说明：所有组件离线可跑。
    """
    snapshot_path = tmp_path / "paper_kg_snapshot.json"
    build_mock_kg(str(snapshot_path), raw_path=None)

    graph = NetworkXGraphStore()
    graph.load_dict(json.loads(snapshot_path.read_text(encoding="utf-8")))

    embedding_model = MockEmbeddingModel()
    vector_store = _build_vector_store(graph, embedding_model)

    config = PaperWorkflowConfig(
        seed=123,
        novelty_threshold=0.85,
        pass_threshold=7.0,
        max_iterations=5,
        max_rewrite_retry=2,
        max_global_retry=2,
        max_plan_retry=2,
        max_patterns=2,
        temperature=0.8,
        retriever_top_k=5,
        penalty=0.3,
        require_individual_threshold=False,
    )

    return PaperFeedbackLoopController(
        config=config,
        graph_store=graph,
        vector_store=vector_store,
        embedding_model=embedding_model,
        rng=random.Random(123),
        input_parser=PaperInputParser(llm=None),
        draft_writer=PaperDraftWriter(llm=None),
        reviewer_group=reviewer_group,
        section_llm=None,
        paper_template=load_paper_template("paper_template.yaml"),
    )


def test_paper_workflow_pass_path(tmp_path) -> None:
    """
    功能：验证 PASS 正常通过路径。
    参数：tmp_path。
    返回：无。
    流程：reviewer 恒 PASS → controller.run → status=PASS。
    说明：不调用真实 LLM。
    """
    pass_reviews = [
        _make_review("theory", "T", 8.0, "PASS"),
        _make_review("experiment", "E", 8.0, "PASS"),
        _make_review("novelty", "N", 8.0, "PASS"),
        _make_review("practical", "P", 8.0, "PASS"),
    ]
    controller = _build_controller(tmp_path, _StaticReviewerGroup([pass_reviews]))
    report = controller.run("我想写一篇用 Diffusion 做推理的论文，实验要扎实")
    assert report.status == "PASS"
    assert report.draft is not None


def test_paper_workflow_backoff_clarity(tmp_path) -> None:
    """
    功能：验证 CLARITY_ISSUE 会回退到抽象/引言重写。
    参数：tmp_path。
    返回：无。
    流程：第一次评审给 CLARITY_ISSUE → 第二次 PASS → rewrite_retry=1。
    说明：重写走分章节离线兜底逻辑。
    """
    clarity_defect = [PaperReviewDefect(type="clarity_low", location="writing", detail="表达不清晰")]
    fail_reviews = [
        _make_review("theory", "T", 5.0, "CLARITY_ISSUE", defects=clarity_defect),
        _make_review("experiment", "E", 5.0, "CLARITY_ISSUE", defects=clarity_defect),
        _make_review("novelty", "N", 5.0, "CLARITY_ISSUE", defects=clarity_defect),
        _make_review("practical", "P", 5.0, "CLARITY_ISSUE", defects=clarity_defect),
    ]
    pass_reviews = [
        _make_review("theory", "T", 8.0, "PASS"),
        _make_review("experiment", "E", 8.0, "PASS"),
        _make_review("novelty", "N", 8.0, "PASS"),
        _make_review("practical", "P", 8.0, "PASS"),
    ]
    controller = _build_controller(tmp_path, _StaticReviewerGroup([fail_reviews, pass_reviews]))
    report = controller.run("我想写一篇用 Diffusion 做推理的论文，表达要清晰")
    assert report.status == "PASS"
    assert report.metrics.get("rewrite_retry") == 1


def test_paper_workflow_backoff_soundness(tmp_path) -> None:
    """
    功能：验证 SOUNDNESS_RISK 会回退到 Method/Experiments 重写。
    参数：tmp_path。
    返回：无。
    流程：第一次评审 SOUNDNESS_RISK → 第二次 PASS → rewrite_retry=1。
    说明：该测试同时验证缺陷会触发 fix_tricks 查询（通过 KG fixed_by 边）。
    """
    sound_defect = [PaperReviewDefect(type="theory_missing", location="method", detail="缺少理论支撑")]
    fail_reviews = [
        _make_review("theory", "T", 5.0, "SOUNDNESS_RISK", defects=sound_defect),
        _make_review("experiment", "E", 5.0, "SOUNDNESS_RISK", defects=sound_defect),
        _make_review("novelty", "N", 5.0, "SOUNDNESS_RISK", defects=sound_defect),
        _make_review("practical", "P", 5.0, "SOUNDNESS_RISK", defects=sound_defect),
    ]
    pass_reviews = [
        _make_review("theory", "T", 8.0, "PASS"),
        _make_review("experiment", "E", 8.0, "PASS"),
        _make_review("novelty", "N", 8.0, "PASS"),
        _make_review("practical", "P", 8.0, "PASS"),
    ]
    controller = _build_controller(tmp_path, _StaticReviewerGroup([fail_reviews, pass_reviews]))
    report = controller.run("我想写一篇用 Diffusion 做推理的论文，理论要严谨")
    assert report.status == "PASS"
    assert report.metrics.get("rewrite_retry") == 1
    assert report.tricks  # 应能正常产出 tricks


def test_paper_workflow_max_retry_failed(tmp_path) -> None:
    """
    功能：验证超过 max_rewrite_retry 会进入 FAILED。
    参数：tmp_path。
    返回：无。
    流程：reviewer 永远 SOUNDNESS_RISK → rewrite_retry 递增 → 超过上限失败。
    说明：用于覆盖“最大重试保护”逻辑。
    """
    sound_defect = [PaperReviewDefect(type="theory_missing", location="method", detail="缺少理论支撑")]
    fail_reviews = [
        _make_review("theory", "T", 5.0, "SOUNDNESS_RISK", defects=sound_defect),
        _make_review("experiment", "E", 5.0, "SOUNDNESS_RISK", defects=sound_defect),
        _make_review("novelty", "N", 5.0, "SOUNDNESS_RISK", defects=sound_defect),
        _make_review("practical", "P", 5.0, "SOUNDNESS_RISK", defects=sound_defect),
    ]

    reviewer_group = _StaticReviewerGroup([fail_reviews, fail_reviews, fail_reviews])
    controller = _build_controller(tmp_path, reviewer_group)
    # 覆盖配置：让最大 rewrite_retry 更小，便于快速触发 FAILED
    controller.config.max_rewrite_retry = 1
    controller.config.max_iterations = 5

    report = controller.run("我想写一篇用 Diffusion 做推理的论文，理论要严谨")
    assert report.status == "FAILED"
    assert report.metrics.get("rewrite_retry", 0) >= 2
