# 中文注释: 本文件实现内存向量库，用于原型检索。
from __future__ import annotations

from typing import Dict, List

from paper_kg.embedding_mock import cosine_similarity
from paper_kg.interfaces import IVectorStore


class InMemoryVectorStore(IVectorStore):
    """
    功能：内存向量存储与检索。
    参数：无。
    返回：InMemoryVectorStore 实例。
    流程：在内存中保存向量与元数据。
    说明：适合原型与测试，不做持久化。
    """

    def __init__(self) -> None:
        """
        功能：初始化向量列表。
        参数：无。
        返回：无。
        流程：创建空列表。
        说明：调用 upsert 进行写入。
        """
        self.items: List[Dict] = []

    def upsert(self, item_id: str, vector: List[float], metadata: Dict) -> None:
        """
        功能：写入或更新向量记录。
        参数：item_id、vector、metadata。
        返回：无。
        流程：查找同 ID → 覆盖或追加。
        说明：保持 item_id 唯一。
        """
        for item in self.items:
            if item["item_id"] == item_id:
                item["vector"] = vector
                item["metadata"] = metadata
                return
        self.items.append({"item_id": item_id, "vector": vector, "metadata": metadata})

    def search(self, query_vector: List[float], top_k: int) -> List[Dict]:
        """
        功能：向量检索。
        参数：query_vector、top_k。
        返回：相似项列表。
        流程：计算余弦相似度 → 排序 → 截断。
        说明：按 score 降序返回。
        """
        scored = []
        for item in self.items:
            score = cosine_similarity(query_vector, item["vector"])
            scored.append(
                {"item_id": item["item_id"], "score": score, "metadata": item["metadata"]}
            )
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]
