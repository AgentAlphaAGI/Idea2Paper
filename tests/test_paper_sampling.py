# 中文注释: 本文件测试多样性采样逻辑。
import random

from paper_kg.models import Pattern
from paper_kg.sampling import softmax, select_patterns_diverse


def test_sampling_softmax_temperature() -> None:
    """
    功能：验证 softmax 温度影响概率分布。
    参数：无。
    返回：无。
    流程：计算不同温度的 softmax 概率。
    说明：温度越低越偏向高分。
    """
    scores = [10.0, 1.0, 1.0]
    low_temp = softmax(scores, temperature=0.1)
    high_temp = softmax(scores, temperature=5.0)
    assert low_temp[0] > high_temp[0]


def test_select_patterns_diverse() -> None:
    """
    功能：验证无放回采样不会重复。
    参数：无。
    返回：无。
    流程：采样套路 → 检查唯一性。
    说明：结果长度应等于 max_patterns。
    """
    rng = random.Random(123)
    patterns = [
        Pattern(id=f"pat_{i}", pattern_name=f"P{i}", pattern_summary="", writing_guide="")
        for i in range(5)
    ]
    scores = [1.0, 2.0, 3.0, 4.0, 5.0]
    selected = select_patterns_diverse(patterns, scores, max_patterns=3, temperature=1.0, rng=rng)
    ids = [p.id for p in selected]
    assert len(ids) == len(set(ids))
