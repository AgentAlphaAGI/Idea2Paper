# 中文注释: 本文件定义知识图谱与 RAG 组件的可替换接口。
from __future__ import annotations

from typing import Dict, List, Optional, Protocol

from paper_kg.models import Defect, NodeID, NodeType, EdgeType, Pattern, Trick


class IGraphStore(Protocol):
    """
    功能：图存储接口定义。
    参数：无。
    返回：实现该接口的图存储实例。
    流程：定义节点/边的增删改查与查询方法。
    说明：用于解耦 NetworkX/Neo4j 等实现。
    """

    def upsert_node(self, node_type: NodeType, node_id: NodeID, attrs: Dict) -> None:
        """
        功能：新增或更新节点。
        参数：node_type、node_id、attrs。
        返回：无。
        流程：写入节点属性。
        说明：重复调用应覆盖更新。
        """
        ...

    def upsert_edge(
        self,
        src_type: NodeType,
        src_id: NodeID,
        rel: EdgeType,
        dst_type: NodeType,
        dst_id: NodeID,
        attrs: Dict,
    ) -> None:
        """
        功能：新增或更新边。
        参数：src_type/src_id/rel/dst_type/dst_id/attrs。
        返回：无。
        流程：写入边关系与属性。
        说明：允许多条同类型边。
        """
        ...

    def get_node(self, node_type: NodeType, node_id: NodeID) -> Optional[Dict]:
        """
        功能：获取节点属性。
        参数：node_type、node_id。
        返回：节点属性字典或 None。
        流程：按节点键查找。
        说明：用于读取节点详情。
        """
        ...

    def neighbors(
        self,
        node_type: NodeType,
        node_id: NodeID,
        rel: Optional[EdgeType] = None,
        direction: str = "out",
    ) -> List[Dict]:
        """
        功能：获取相邻节点。
        参数：node_type、node_id、rel、direction。
        返回：相邻节点属性列表。
        流程：根据方向与关系筛选。
        说明：direction 可选 out/in/both。
        """
        ...

    def list_nodes(self, node_type: Optional[NodeType] = None) -> List[Dict]:
        """
        功能：列出节点列表。
        参数：node_type。
        返回：节点属性列表。
        流程：过滤节点类型 → 返回。
        说明：用于遍历图谱。
        """
        ...

    def query_patterns_for_idea_domain(
        self, idea_id: NodeID, domain_id: NodeID
    ) -> List[Pattern]:
        """
        功能：查询某 idea 在指定 domain 的候选套路。
        参数：idea_id、domain_id。
        返回：Pattern 列表。
        流程：通过 Paper-HAS_IDEA/IN_DOMAIN/BELONGS_TO_PATTERN 关系聚合。
        说明：无结果时可回退到按 Domain 过滤或全量 Pattern。
        """
        ...

    def query_pattern_effectiveness(self, pattern_id: NodeID, domain_id: NodeID) -> float:
        """
        功能：查询套路在领域的有效性。
        参数：pattern_id、domain_id。
        返回：effectiveness 数值。
        流程：基于样本数/一致性等统计规则计算。
        说明：找不到则返回 0。
        """
        ...

    def find_merged_pattern(
        self, pattern1_id: NodeID, pattern2_id: NodeID
    ) -> Optional[Pattern]:
        """
        功能：查找合并套路结果。
        参数：pattern1_id、pattern2_id。
        返回：合并后的 Pattern 或 None。
        流程：查找 mutation:merge 边。
        说明：用于 Action.merge_patterns。
        """
        ...

    def get_tricks_for_pattern(self, pattern_id: NodeID) -> List[Dict]:
        """
        功能：获取套路关联的 Trick 节点。
        参数：pattern_id。
        返回：Trick 节点属性列表。
        流程：读取 CONTAINS_TRICK 边。
        说明：按边出现顺序返回。
        """
        ...

    def query_tricks_for_defect(self, defect_type: str, top_k: int = 3) -> List[Trick]:
        """
        功能：根据缺陷类型查询可用于修复的 Tricks。
        参数：defect_type、top_k。
        返回：Trick 列表（通常按 success_rate 降序）。
        流程：定位 Defect 节点 → 沿 fixed_by 边检索 Trick → 排序截断。
        说明：
        - 这是一个“便捷查询”接口，便于上层编排器实现“缺陷驱动修复”。
        - 默认实现可能将边上的 success_rate 映射到 Trick.effectiveness 字段返回。
        """
        ...

    def query_defects_for_review(self, review_id: NodeID) -> List[Defect]:
        """
        功能：查询某条 Review 指向的缺陷列表。
        参数：review_id。
        返回：Defect 列表。
        流程：沿 identifies 边检索 Defect 节点。
        说明：该方法用于从历史 Review 反推缺陷分布（可选能力）。
        """
        ...

    def resolve_paper_id(self, paper_id_or_openreview_id: NodeID) -> Optional[NodeID]:
        """
        功能：解析论文主键（兼容 paper_id / openreview_id）。
        参数：paper_id_or_openreview_id（可能是内部 paper_id 或外部 openreview_id）。
        返回：内部 paper_id 或 None。
        流程：优先按 paper_id 命中 → 否则按 openreview_id 映射。
        说明：
        - 该方法用于 Review.paper_id 兼容处理，避免要求数据源改格式。
        - 默认实现可维护 openreview_id 索引，以提高解析效率。
        """
        ...

    def to_dict(self) -> Dict:
        """
        功能：导出图谱快照为字典。
        参数：无。
        返回：包含 nodes 与 edges 的字典。
        流程：遍历节点/边 → 序列化。
        说明：用于保存 JSON 快照。
        """
        ...

    def load_dict(self, data: Dict) -> None:
        """
        功能：从字典加载图谱。
        参数：data。
        返回：无。
        流程：重建节点与边。
        说明：会清空现有图。
        """
        ...


class IDocStore(Protocol):
    """
    功能：原始文档存储接口定义。
    参数：无。
    返回：实现该接口的文档存储。
    流程：保存/读取原始论文与审稿数据。
    说明：用于替换 MongoDB 等存储。
    """

    def save_raw_paper(self, paper_id: NodeID, raw: Dict) -> None:
        """
        功能：保存原始论文数据。
        参数：paper_id、raw。
        返回：无。
        流程：写入存储介质。
        说明：raw 保持原始结构。
        """
        ...

    def save_raw_review(self, review_id: NodeID, raw: Dict) -> None:
        """
        功能：保存原始审稿数据。
        参数：review_id、raw。
        返回：无。
        流程：写入存储介质。
        说明：raw 保持原始结构。
        """
        ...

    def load_raw_paper(self, paper_id: NodeID) -> Optional[Dict]:
        """
        功能：读取原始论文数据。
        参数：paper_id。
        返回：原始数据或 None。
        流程：读取存储介质。
        说明：不存在时返回 None。
        """
        ...

    def load_raw_review(self, review_id: NodeID) -> Optional[Dict]:
        """
        功能：读取原始审稿数据。
        参数：review_id。
        返回：原始数据或 None。
        流程：读取存储介质。
        说明：不存在时返回 None。
        """
        ...


class IVectorStore(Protocol):
    """
    功能：向量存储接口定义。
    参数：无。
    返回：向量存储实现。
    流程：写入向量与检索相似项。
    说明：可替换为 FAISS/PGVector 等。
    """

    def upsert(self, item_id: str, vector: List[float], metadata: Dict) -> None:
        """
        功能：写入或更新向量。
        参数：item_id、vector、metadata。
        返回：无。
        流程：存储向量与元信息。
        说明：重复 item_id 应覆盖。
        """
        ...

    def search(self, query_vector: List[float], top_k: int) -> List[Dict]:
        """
        功能：向量检索。
        参数：query_vector、top_k。
        返回：相似项列表（包含 score/metadata）。
        流程：计算相似度 → 排序。
        说明：返回 score 降序。
        """
        ...


class IEmbeddingModel(Protocol):
    """
    功能：向量化模型接口。
    参数：无。
    返回：向量列表。
    流程：将文本编码为向量。
    说明：可替换为真实 embedding API。
    """

    def embed(self, text: str) -> List[float]:
        """
        功能：生成文本向量。
        参数：text。
        返回：向量列表。
        流程：编码文本。
        说明：向量维度由实现决定。
        """
        ...


class IKGQueryService(Protocol):
    """
    功能：知识图谱查询服务接口。
    参数：无。
    返回：查询结果。
    流程：对图谱做高层封装查询。
    说明：用于解耦编排器与底层图存储。
    """

    def query_patterns_for_idea_domain(
        self, idea_id: NodeID, domain_id: NodeID
    ) -> List[Pattern]:
        """
        功能：查询指定 Idea/Domain 下的候选套路。
        参数：idea_id、domain_id。
        返回：Pattern 列表。
        流程：组合图关系进行检索（HAS_IDEA/IN_DOMAIN/BELONGS_TO_PATTERN）。
        说明：可在无结果时回退到 Domain 维度的候选模式。
        """
        ...

    def query_pattern_effectiveness(self, pattern_id: NodeID, domain_id: NodeID) -> float:
        """
        功能：查询套路在领域的有效性。
        参数：pattern_id、domain_id。
        返回：effectiveness 数值。
        流程：基于统计规则计算（如样本数/一致性分数）。
        说明：找不到时返回 0。
        """
        ...

    def find_merged_pattern(
        self, pattern1_id: NodeID, pattern2_id: NodeID
    ) -> Optional[Pattern]:
        """
        功能：查找合并套路结果。
        参数：pattern1_id、pattern2_id。
        返回：合并后的 Pattern 或 None。
        流程：查询 mutation:merge 边（预留）。
        说明：用于合并动作。
        """
        ...

    def get_tricks_for_pattern(self, pattern_id: NodeID) -> List[Dict]:
        """
        功能：获取套路关联的 Trick 节点。
        参数：pattern_id。
        返回：Trick 节点属性列表。
        流程：读取 CONTAINS_TRICK 边。
        说明：按边出现顺序返回。
        """
        ...
