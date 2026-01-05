# 中文注释: 本文件测试故事状态动作的不可变迁移。
from paper_kg.models import Defect, Pattern, Trick
from paper_kg.story_actions import Action
from paper_kg.story_state import StoryState
from paper_kg.graph_networkx import NetworkXGraphStore


def test_story_actions_deepcopy() -> None:
    """
    功能：验证动作不会修改原状态。
    参数：无。
    返回：无。
    流程：执行动作 → 对比原状态。
    说明：确保 deepcopy 生效。
    """
    state = StoryState()
    pattern = Pattern(
        id="pat_1",
        pattern_name="方法叙述递进",
        pattern_summary="",
        writing_guide="",
    )
    trick = Trick(id="trick_1", name="消融实验", trick_type="experiment-level", purpose="可信度")
    new_state = Action.add_pattern(state, pattern, tricks=[trick])

    assert len(state.patterns) == 0
    assert len(new_state.patterns) == 1
    assert new_state.tricks[0].id == "trick_1"


def test_fix_defect_action() -> None:
    """
    功能：验证缺陷修复动作。
    参数：无。
    返回：无。
    流程：添加缺陷 → 使用 Trick 修复。
    说明：缺陷应从新状态中移除。
    """
    defect = Defect(defect_id="def_1", type="clarity", description="表达不清")
    trick = Trick(id="trick_2", name="案例研究", trick_type="writing-level", purpose="清晰度")
    state = StoryState(defects=[defect])
    new_state = Action.fix_defect(state, defect, trick)

    assert len(state.defects) == 1
    assert len(new_state.defects) == 0
    assert new_state.tricks[0].id == "trick_2"


def test_merge_patterns_action() -> None:
    """
    功能：验证合并套路动作。
    参数：无。
    返回：无。
    流程：构造 merge 关系 → 执行动作。
    说明：新状态包含合并后的套路。
    """
    graph = NetworkXGraphStore()
    pattern1 = Pattern(id="pat_a", pattern_name="A", pattern_summary="", writing_guide="")
    pattern2 = Pattern(id="pat_b", pattern_name="B", pattern_summary="", writing_guide="")
    merged = Pattern(id="pat_c", pattern_name="C", pattern_summary="", writing_guide="")

    graph.upsert_node("Pattern", pattern1.id, pattern1.to_dict())
    graph.upsert_node("Pattern", pattern2.id, pattern2.to_dict())
    graph.upsert_node("Pattern", merged.id, merged.to_dict())
    graph.upsert_edge(
        "Pattern",
        pattern1.id,
        "mutation:merge",
        "Pattern",
        merged.id,
        {"with": pattern2.id},
    )

    state = StoryState(patterns=[pattern1, pattern2])
    new_state = Action.merge_patterns(state, pattern1, pattern2, graph)
    ids = [p.id for p in new_state.patterns]
    assert "pat_c" in ids
