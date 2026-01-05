# 中文注释: 本文件测试论文 Reviewer 聚合逻辑（trimmed mean 等）。

from paper_kg.paper_reviewers import trimmed_mean


def test_trimmed_mean_basic() -> None:
    """
    功能：验证 trimmed_mean 会去掉最高与最低分后求均值。
    参数：无。
    返回：无。
    流程：输入 [1,9,5,7] → 去掉 1 与 9 → (5+7)/2=6。
    说明：用于降低极端 reviewer 的影响。
    """
    assert trimmed_mean([1, 9, 5, 7]) == 6.0


def test_trimmed_mean_small_n() -> None:
    """
    功能：验证当样本数 < 3 时，trimmed_mean 等同于普通均值。
    参数：无。
    返回：无。
    流程：输入 [6,8] → 均值 7。
    说明：兼容少 reviewer 情况。
    """
    assert trimmed_mean([6, 8]) == 7.0

