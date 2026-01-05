# 中文注释: 本文件测试 pattern effectiveness 的规则计算。
import math
from paper_kg.graph_networkx import NetworkXGraphStore


def test_pattern_effectiveness() -> None:
    """
    功能：验证 pattern effectiveness 的规则计算。
    参数：无。
    返回：无。
    流程：写入 Pattern/Domain/Paper → 调用 query_pattern_effectiveness。
    说明：覆盖 sample_count + coherence_score 的计算逻辑。
    """
    graph = NetworkXGraphStore()
    graph.upsert_node(
        "Pattern",
        "pat_1",
        {"id": "pat_1", "pattern_name": "测试", "coherence_score": 0.5},
    )
    graph.upsert_node("Domain", "domain_nlp", {"id": "domain_nlp", "name": "自然语言处理"})

    # 中文注释：构造两篇论文使用该 pattern，形成 sample_count=2。
    for i in range(2):
        paper_id = f"paper_{i}"
        graph.upsert_node("Paper", paper_id, {"id": paper_id, "title": "Demo"})
        graph.upsert_edge("Paper", paper_id, "IN_DOMAIN", "Domain", "domain_nlp", {})
        graph.upsert_edge("Paper", paper_id, "BELONGS_TO_PATTERN", "Pattern", "pat_1", {})

    effectiveness = graph.query_pattern_effectiveness("pat_1", "domain_nlp")
    expected = math.log1p(2) * (0.5 + 0.5)
    assert abs(effectiveness - expected) < 1e-6
