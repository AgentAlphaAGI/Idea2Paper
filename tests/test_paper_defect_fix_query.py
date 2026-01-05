# 中文注释: 本文件测试 KG 中“缺陷→修复 Trick”查询能力。
import json

from api.paper_cli import build_mock_kg
from paper_kg.graph_networkx import NetworkXGraphStore


def test_query_tricks_for_defect_sorted(tmp_path) -> None:
    """
    功能：验证 query_tricks_for_defect 能返回修复 Trick 列表。
    参数：tmp_path。
    返回：无。
    流程：构建快照 → 加载 → 查询 novelty_low → 校验数量与去重。
    说明：该测试不依赖外部网络或数据库，排序不做强约束。
    """
    snapshot_path = tmp_path / "paper_kg_snapshot.json"
    build_mock_kg(str(snapshot_path), raw_path=None)

    graph = NetworkXGraphStore()
    graph.load_dict(json.loads(snapshot_path.read_text(encoding="utf-8")))

    tricks = graph.query_tricks_for_defect("novelty_low", top_k=2)
    assert len(tricks) == 2
    ids = [trick.id for trick in tricks]
    assert len(ids) == len(set(ids))
