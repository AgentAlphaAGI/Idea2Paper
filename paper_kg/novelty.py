# 中文注释: 本文件实现套路配置的新颖性查重。
from __future__ import annotations

import json
from typing import Dict, List

from paper_kg.interfaces import IEmbeddingModel, IVectorStore
from paper_kg.models import NoveltyCheckResult, Pattern, SimilarItem


def _truncate(text: str, limit: int) -> str:
    """
    功能：截断文本长度。
    参数：text、limit。
    返回：截断后的文本。
    流程：长度判断 → 截断。
    说明：用于控制 embedding 输入长度。
    """
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit]


def encode_pattern_config(
    patterns: List[Pattern],
    method: str = "structured",
    summary_limit: int = 300,
) -> str:
    """
    功能：将套路配置编码为文本。
    参数：patterns、method、summary_limit。
    返回：编码后的文本。
    流程：结构化编码或拼接描述。
    说明：默认使用结构化编码以提高稳定性。
    """
    if method == "concat":
        parts = []
        for pattern in patterns:
            parts.append(
                f"{pattern.pattern_name} {_truncate(pattern.pattern_summary, summary_limit)}"
            )
        return " | ".join(parts)

    # 中文注释：结构化编码，避免引入 writing_guide 导致文本过长。
    payload: Dict[str, List[Dict[str, str]]] = {"patterns": []}
    for pattern in patterns:
        payload["patterns"].append(
            {
                "id": str(pattern.id),
                "name": pattern.pattern_name,
                "summary": _truncate(pattern.pattern_summary, summary_limit),
            }
        )
    return json.dumps(payload, ensure_ascii=False)


def check_novelty(
    patterns: List[Pattern],
    vector_store: IVectorStore,
    embedding_model: IEmbeddingModel,
    threshold: float = 0.85,
    top_k: int = 10,
) -> NoveltyCheckResult:
    """
    功能：检查套路配置的新颖性。
    参数：patterns、vector_store、embedding_model、threshold、top_k。
    返回：NoveltyCheckResult。
    流程：编码 → 向量检索 → 计算相似度阈值。
    说明：返回中文建议。
    """
    text = encode_pattern_config(patterns)
    embedding = embedding_model.embed(text)
    results = vector_store.search(embedding, top_k=top_k)
    if not results:
        return NoveltyCheckResult(
            is_novel=True,
            max_similarity=0.0,
            top_similar=[],
            suggestion="暂无相似配置，判定为新颖。",
        )

    max_similarity = float(results[0].get("score", 0.0))
    top_similar = [
        SimilarItem(
            item_id=str(item.get("item_id")),
            score=float(item.get("score", 0.0)),
            metadata=item.get("metadata", {}),
        )
        for item in results[:3]
    ]

    if max_similarity > threshold:
        return NoveltyCheckResult(
            is_novel=False,
            max_similarity=max_similarity,
            top_similar=top_similar,
            suggestion="相似度过高，建议调整套路组合以降低相似度。",
        )

    return NoveltyCheckResult(
        is_novel=True,
        max_similarity=max_similarity,
        top_similar=top_similar,
        suggestion="相似度在可接受范围内，可进入生成阶段。",
    )
