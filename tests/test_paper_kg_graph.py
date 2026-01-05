# 中文注释: 本文件测试知识图谱节点/边写入与查询。
from paper_kg.graph_networkx import NetworkXGraphStore
from paper_kg.models import Idea, Domain, Pattern, Trick


def test_graph_query_patterns() -> None:
    """
    功能：验证图谱写入与套路查询是否正确。
    参数：无。
    返回：无。
    流程：写入节点与边 → 调用 query_patterns_for_idea_domain。
    说明：用于覆盖节点/边查询逻辑。
    """
    graph = NetworkXGraphStore()
    idea = Idea(id="idea_1", core_idea="用 Diffusion 做推理")
    domain = Domain(id="domain_nlp", name="自然语言处理")
    pattern = Pattern(
        id="pat_1",
        pattern_name="研究缺口放大",
        pattern_summary="强调现有方法不足",
        writing_guide="",
    )
    trick = Trick(id="trick_1", name="消融实验", trick_type="experiment-level", purpose="可信度")

    graph.upsert_node("Idea", idea.id, idea.to_dict())
    graph.upsert_node("Domain", domain.id, domain.to_dict())
    graph.upsert_node("Pattern", pattern.id, pattern.to_dict())
    graph.upsert_node("Trick", trick.id, trick.to_dict())

    graph.upsert_edge("Pattern", pattern.id, "CONTAINS_TRICK", "Trick", trick.id, {"order": 1})

    graph.upsert_node("Paper", "paper_1", {"id": "paper_1", "title": "Demo"})
    graph.upsert_edge("Paper", "paper_1", "HAS_IDEA", "Idea", idea.id, {})
    graph.upsert_edge("Paper", "paper_1", "IN_DOMAIN", "Domain", domain.id, {})
    graph.upsert_edge("Paper", "paper_1", "BELONGS_TO_PATTERN", "Pattern", pattern.id, {})

    results = graph.query_patterns_for_idea_domain(idea.id, domain.id)
    assert len(results) == 1
    assert results[0].id == pattern.id

    tricks = graph.get_tricks_for_pattern(pattern.id)
    assert tricks[0]["id"] == trick.id


def test_resolve_paper_id_with_openreview() -> None:
    """
    功能：验证 openreview_id 到 paper_id 的兼容映射是否可用。
    参数：无。
    返回：无。
    流程：写入 Paper 节点 → 使用 resolve_paper_id 校验。
    说明：确保 Review.paper_id 能兼容外部 ID。
    """
    graph = NetworkXGraphStore()
    graph.upsert_node(
        "Paper",
        "paper_1",
        {"id": "paper_1", "openreview_id": "or_1", "title": "Demo"},
    )

    assert graph.resolve_paper_id("paper_1") == "paper_1"
    assert graph.resolve_paper_id("or_1") == "paper_1"
