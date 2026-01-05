# 中文注释: 本文件测试新颖性查重逻辑。
from paper_kg.novelty import check_novelty, encode_pattern_config
from paper_kg.models import Pattern
from paper_kg.vectorstore_memory import InMemoryVectorStore
from paper_kg.embedding_mock import MockEmbeddingModel


def test_novelty_check_trigger() -> None:
    """
    功能：验证相似度超过阈值时判为不新颖。
    参数：无。
    返回：无。
    流程：构建相同向量 → 进行查重。
    说明：相似度应高于阈值。
    """
    embedding_model = MockEmbeddingModel()
    store = InMemoryVectorStore()

    pattern = Pattern(
        id="pat_1",
        pattern_name="研究缺口放大",
        pattern_summary="强调现有方法不足",
        writing_guide="",
    )
    text = encode_pattern_config([pattern])
    vec = embedding_model.embed(text)
    store.upsert("paper_1", vec, {"pattern_ids": [pattern.id]})

    result = check_novelty([pattern], store, embedding_model, threshold=0.1, top_k=5)
    assert result.is_novel is False
    assert result.max_similarity >= 0.1
