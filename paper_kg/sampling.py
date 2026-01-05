# 中文注释: 本文件提供多样性采样与降权策略。
from __future__ import annotations

import math
import random
from typing import Iterable, List, Sequence

from paper_kg.models import Pattern


def softmax(scores: Sequence[float], temperature: float = 1.0) -> List[float]:
    """
    功能：计算 softmax 概率分布。
    参数：scores、temperature。
    返回：概率列表。
    流程：数值平移 → exp → 归一化。
    说明：temperature 越小越偏向高分。
    """
    if not scores:
        return []
    temp = max(temperature, 1e-6)
    max_score = max(scores)
    exp_scores = [math.exp((score - max_score) / temp) for score in scores]
    total = sum(exp_scores)
    if total <= 0:
        return [1.0 / len(scores) for _ in scores]
    return [value / total for value in exp_scores]


def sample_without_replacement(
    items: Sequence, probs: Sequence[float], k: int, rng: random.Random
) -> List:
    """
    功能：按概率无放回采样。
    参数：items、probs、k、rng。
    返回：采样结果列表。
    流程：按概率抽取 → 移除 → 重新归一化。
    说明：k 超出长度时返回所有元素。
    """
    items = list(items)
    probs = list(probs)
    if k <= 0 or not items:
        return []
    k = min(k, len(items))
    selected = []
    for _ in range(k):
        total = sum(probs)
        if total <= 0:
            probs = [1.0 / len(items) for _ in items]
            total = 1.0
        threshold = rng.random() * total
        cumulative = 0.0
        index = 0
        for i, p in enumerate(probs):
            cumulative += p
            if cumulative >= threshold:
                index = i
                break
        selected.append(items.pop(index))
        probs.pop(index)
    return selected


def select_patterns_diverse(
    candidates: Sequence[Pattern],
    effectiveness_scores: Sequence[float],
    max_patterns: int,
    temperature: float,
    rng: random.Random,
) -> List[Pattern]:
    """
    功能：基于 softmax 的多样性采样。
    参数：candidates、effectiveness_scores、max_patterns、temperature、rng。
    返回：套路列表。
    流程：softmax → 无放回采样。
    说明：保持输出不重复。
    """
    probs = softmax(effectiveness_scores, temperature=temperature)
    return sample_without_replacement(candidates, probs, max_patterns, rng)


def downweight_scores(
    pattern_ids: Sequence[str],
    scores: Sequence[float],
    penalized_ids: Iterable[str],
    penalty: float,
) -> List[float]:
    """
    功能：对指定 pattern 进行降权。
    参数：pattern_ids、scores、penalized_ids、penalty。
    返回：新的分数列表。
    流程：命中 ID 的分数乘以 (1-penalty)。
    说明：penalty 建议在 0~1 之间。
    """
    penalized = set(penalized_ids)
    updated = []
    factor = max(0.0, 1.0 - penalty)
    for pid, score in zip(pattern_ids, scores):
        if pid in penalized:
            updated.append(score * factor)
        else:
            updated.append(score)
    return updated
