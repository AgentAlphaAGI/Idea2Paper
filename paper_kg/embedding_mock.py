# 中文注释: 本文件提供论文链路的本地 embedding 工具与 mock 实现（用于离线测试）。
from __future__ import annotations

import hashlib
import math
import re
from typing import List

from paper_kg.interfaces import IEmbeddingModel


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> List[str]:
    """
    功能：将输入文本切分为 token 列表。
    参数：text。
    返回：token 列表。
    流程：正则提取英文/数字/下划线 token → 若为空则按字符拆分。
    说明：用于 mock embedding 的简单特征构建。
    """
    tokens = _TOKEN_RE.findall(text.lower())
    if tokens:
        return tokens
    return [ch for ch in text if not ch.isspace()]


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """
    功能：计算向量余弦相似度。
    参数：vec_a、vec_b。
    返回：相似度浮点数。
    流程：点积 / (范数乘积)。
    说明：用于内存向量库的相似度排序。
    """
    if not vec_a or not vec_b:
        return 0.0
    if len(vec_a) != len(vec_b):
        raise ValueError("Vectors must be the same length")
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class MockEmbeddingModel(IEmbeddingModel):
    """
    功能：基于 hash 的轻量 embedding mock。
    参数：dim（向量维度）。
    返回：MockEmbeddingModel 实例。
    流程：token → hash → 计数向量 → 归一化。
    说明：仅用于测试/离线模式，不应在生产中使用。
    """

    def __init__(self, dim: int = 64) -> None:
        """
        功能：初始化 mock embedding。
        参数：dim。
        返回：无。
        流程：保存向量维度。
        说明：dim 越大向量越稀疏，但计算更慢。
        """
        self.dim = dim

    def embed(self, text: str) -> List[float]:
        """
        功能：生成 mock embedding 向量。
        参数：text。
        返回：浮点向量列表。
        流程：token hash → 索引计数 → L2 归一化。
        说明：用于离线测试与单元测试。
        """
        vector = [0.0] * self.dim
        for token in tokenize(text):
            digest = hashlib.md5(token.encode("utf-8")).hexdigest()
            idx = int(digest[:8], 16) % self.dim
            vector[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0:
            return vector
        return [v / norm for v in vector]
