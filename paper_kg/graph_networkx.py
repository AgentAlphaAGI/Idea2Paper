# 中文注释: 本文件提供基于 NetworkX 的图存储实现。
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import networkx as nx

from paper_kg.interfaces import IGraphStore
from paper_kg.models import Defect, Domain, Idea, Paper, Pattern, Trick


class NetworkXGraphStore(IGraphStore):
    """
    功能：NetworkX 图存储实现。
    参数：无。
    返回：可执行的图存储对象。
    流程：使用 MultiDiGraph 存储节点与边。
    说明：关系名与字段名保持 knowledge_graph.json 原样。
    """

    def __init__(self) -> None:
        """
        功能：初始化图存储。
        参数：无。
        返回：无。
        流程：创建 MultiDiGraph 与缓存。
        说明：节点键使用 "type:id" 形式。
        """
        self.graph = nx.MultiDiGraph()
        # 中文注释：兼容保留 openreview_id -> paper_id 的索引。
        self._openreview_index: Dict[str, str] = {}
        # 中文注释：缓存 (pattern_id, domain_id) 的统计信息，避免重复计算。
        self._pattern_domain_cache: Dict[Tuple[str, str], int] = {}

    @staticmethod
    def _node_key(node_type: str, node_id: str) -> str:
        """
        功能：构造节点唯一键。
        参数：node_type、node_id。
        返回：节点键字符串。
        流程：拼接类型与 ID。
        说明：避免不同类型 ID 冲突。
        """
        return f"{node_type}:{node_id}"

    def _node_attrs(self, node_key: str) -> Dict:
        """
        功能：获取节点属性。
        参数：node_key。
        返回：属性字典。
        流程：读取 graph 节点数据。
        说明：过滤掉内部字段。
        """
        attrs = dict(self.graph.nodes[node_key])
        attrs.pop("_node_key", None)
        return attrs

    def upsert_node(self, node_type: str, node_id: str, attrs: Dict) -> None:
        """
        功能：新增或更新节点。
        参数：node_type、node_id、attrs。
        返回：无。
        流程：添加节点并更新属性。
        说明：会补齐 node_type/node_id 与 id 字段。
        """
        node_key = self._node_key(node_type, node_id)
        payload = dict(attrs)
        payload.setdefault("id", node_id)
        payload["node_type"] = node_type
        payload["node_id"] = node_id
        payload["_node_key"] = node_key
        self.graph.add_node(node_key, **payload)
        # 中文注释：记录 openreview_id 映射，便于 Review 兼容解析。
        if node_type == "Paper":
            self._index_openreview_id(node_id, payload)

    def _index_openreview_id(self, paper_id: str, attrs: Dict) -> None:
        """
        功能：写入 openreview_id -> paper_id 的索引。
        参数：paper_id、attrs。
        返回：无。
        流程：读取 attrs.openreview_id → 写入缓存。
        说明：仅在 openreview_id 非空时生效。
        """
        openreview_id = attrs.get("openreview_id")
        if openreview_id:
            self._openreview_index[str(openreview_id)] = str(paper_id)

    def _rebuild_openreview_index(self) -> None:
        """
        功能：重建 openreview_id 索引。
        参数：无。
        返回：无。
        流程：遍历 Paper 节点 → 收集 openreview_id。
        说明：用于加载快照后的索引恢复。
        """
        self._openreview_index.clear()
        for _node_key, attrs in self.graph.nodes(data=True):
            if attrs.get("node_type") != "Paper":
                continue
            paper_id = str(attrs.get("node_id", "")).strip()
            if not paper_id:
                continue
            openreview_id = attrs.get("openreview_id")
            if openreview_id:
                self._openreview_index[str(openreview_id)] = paper_id

    def upsert_edge(
        self,
        src_type: str,
        src_id: str,
        rel: str,
        dst_type: str,
        dst_id: str,
        attrs: Dict,
    ) -> None:
        """
        功能：新增或更新边。
        参数：src_type/src_id/rel/dst_type/dst_id/attrs。
        返回：无。
        流程：确保节点存在 → 添加边。
        说明：支持多条同类型边。
        """
        src_key = self._node_key(src_type, src_id)
        dst_key = self._node_key(dst_type, dst_id)
        if src_key not in self.graph:
            self.upsert_node(src_type, src_id, {"id": src_id})
        if dst_key not in self.graph:
            self.upsert_node(dst_type, dst_id, {"id": dst_id})
        payload = dict(attrs)
        payload["rel"] = rel
        self.graph.add_edge(src_key, dst_key, **payload)

    def get_node(self, node_type: str, node_id: str) -> Optional[Dict]:
        """
        功能：获取节点属性。
        参数：node_type、node_id。
        返回：节点属性字典或 None。
        流程：按节点键读取。
        说明：不存在则返回 None。
        """
        node_key = self._node_key(node_type, node_id)
        if node_key not in self.graph:
            return None
        return self._node_attrs(node_key)

    def resolve_paper_id(self, paper_id_or_openreview_id: str) -> Optional[str]:
        """
        功能：解析论文主键（兼容 paper_id / openreview_id）。
        参数：paper_id_or_openreview_id。
        返回：内部 paper_id 或 None。
        流程：优先按 paper_id 命中 → 否则按 openreview_id 映射。
        说明：用于 Review.paper_id 兼容解析。
        """
        if not paper_id_or_openreview_id:
            return None
        if self.get_node("Paper", paper_id_or_openreview_id):
            return paper_id_or_openreview_id
        if not self._openreview_index:
            self._rebuild_openreview_index()
        return self._openreview_index.get(str(paper_id_or_openreview_id))

    def list_nodes(self, node_type: Optional[str] = None) -> List[Dict]:
        """
        功能：列出节点。
        参数：node_type。
        返回：节点属性列表。
        流程：遍历节点 → 按类型过滤。
        说明：返回包含 node_type/node_id 的字典。
        """
        results: List[Dict] = []
        for _node_key, attrs in self.graph.nodes(data=True):
            if node_type and attrs.get("node_type") != node_type:
                continue
            results.append(dict(attrs))
        return results

    def neighbors(
        self,
        node_type: str,
        node_id: str,
        rel: Optional[str] = None,
        direction: str = "out",
    ) -> List[Dict]:
        """
        功能：获取相邻节点。
        参数：node_type、node_id、rel、direction。
        返回：节点属性列表。
        流程：根据方向遍历边 → 过滤 rel。
        说明：direction 支持 out/in/both。
        """
        node_key = self._node_key(node_type, node_id)
        if node_key not in self.graph:
            return []

        results: List[Dict] = []
        if direction in {"out", "both"}:
            for _src, dst, edge_attrs in self.graph.out_edges(node_key, data=True):
                if rel and edge_attrs.get("rel") != rel:
                    continue
                results.append(self._node_attrs(dst))
        if direction in {"in", "both"}:
            for src, _dst, edge_attrs in self.graph.in_edges(node_key, data=True):
                if rel and edge_attrs.get("rel") != rel:
                    continue
                results.append(self._node_attrs(src))
        return results

    def _edge_matches(
        self, src_key: str, dst_key: str, rel: str, attrs_match: Optional[Dict] = None
    ) -> bool:
        """
        功能：检查是否存在匹配边。
        参数：src_key、dst_key、rel、attrs_match。
        返回：布尔值。
        流程：遍历边数据 → 匹配条件。
        说明：用于关系过滤。
        """
        attrs_match = attrs_match or {}
        edge_data = self.graph.get_edge_data(src_key, dst_key, default={})
        for _key, attrs in edge_data.items():
            if attrs.get("rel") != rel:
                continue
            if all(attrs.get(k) == v for k, v in attrs_match.items()):
                return True
        return False

    def query_patterns_for_idea_domain(self, idea_id: str, domain_id: str) -> List[Pattern]:
        """
        功能：查询某 idea 在指定 domain 的候选套路。
        参数：idea_id、domain_id。
        返回：Pattern 列表。
        流程：Paper-HAS_IDEA/IN_DOMAIN → Paper-BELONGS_TO_PATTERN 聚合。
        说明：为空时回退到仅按 domain 搜索。
        """
        patterns: Dict[str, Pattern] = {}
        idea_key = self._node_key("Idea", idea_id)
        domain_key = self._node_key("Domain", domain_id)

        for node_key, attrs in self.graph.nodes(data=True):
            if attrs.get("node_type") != "Paper":
                continue
            if idea_id and idea_key in self.graph:
                if not self._edge_matches(node_key, idea_key, "HAS_IDEA"):
                    continue
            if domain_id and domain_key in self.graph:
                if not self._edge_matches(node_key, domain_key, "IN_DOMAIN"):
                    continue
            for _src, dst, edge_attrs in self.graph.out_edges(node_key, data=True):
                if edge_attrs.get("rel") != "BELONGS_TO_PATTERN":
                    continue
                pattern_attrs = self._node_attrs(dst)
                pattern = Pattern.from_dict(pattern_attrs)
                patterns[str(pattern.id)] = pattern

        if patterns:
            return list(patterns.values())

        # 中文注释：回退到按 domain 过滤。
        if domain_id and domain_key in self.graph:
            for node_key, attrs in self.graph.nodes(data=True):
                if attrs.get("node_type") != "Paper":
                    continue
                if not self._edge_matches(node_key, domain_key, "IN_DOMAIN"):
                    continue
                for _src, dst, edge_attrs in self.graph.out_edges(node_key, data=True):
                    if edge_attrs.get("rel") != "BELONGS_TO_PATTERN":
                        continue
                    pattern_attrs = self._node_attrs(dst)
                    pattern = Pattern.from_dict(pattern_attrs)
                    patterns[str(pattern.id)] = pattern

        if patterns:
            return list(patterns.values())

        # 中文注释：仍为空时返回所有 Pattern 供上层兜底。
        for attrs in self.list_nodes("Pattern"):
            pattern = Pattern.from_dict(attrs)
            patterns[str(pattern.id)] = pattern
        return list(patterns.values())

    def _count_pattern_in_domain(self, pattern_id: str, domain_id: str) -> int:
        """
        功能：统计某 Pattern 在某 Domain 的样本数。
        参数：pattern_id、domain_id。
        返回：样本数。
        流程：遍历 Paper → 同时满足 IN_DOMAIN 与 BELONGS_TO_PATTERN。
        说明：结果会缓存以加速后续查询。
        """
        cache_key = (pattern_id, domain_id)
        if cache_key in self._pattern_domain_cache:
            return self._pattern_domain_cache[cache_key]

        domain_key = self._node_key("Domain", domain_id)
        pattern_key = self._node_key("Pattern", pattern_id)
        count = 0
        for node_key, attrs in self.graph.nodes(data=True):
            if attrs.get("node_type") != "Paper":
                continue
            if domain_id and domain_key in self.graph:
                if not self._edge_matches(node_key, domain_key, "IN_DOMAIN"):
                    continue
            if pattern_id and pattern_key in self.graph:
                if not self._edge_matches(node_key, pattern_key, "BELONGS_TO_PATTERN"):
                    continue
            count += 1

        self._pattern_domain_cache[cache_key] = count
        return count

    def query_pattern_effectiveness(self, pattern_id: str, domain_id: str) -> float:
        """
        功能：查询套路在领域的有效性。
        参数：pattern_id、domain_id。
        返回：effectiveness 数值。
        流程：样本数 + coherence_score 计算。
        说明：KG 无 works_well_in 时使用规则打分。
        """
        pattern_attrs = self.get_node("Pattern", pattern_id) or {}
        coherence_score = float(pattern_attrs.get("coherence_score") or 0.0)
        sample_count = self._count_pattern_in_domain(pattern_id, domain_id)
        base = 0.5 + max(0.0, coherence_score)
        return float(math.log1p(sample_count) * base)

    def find_merged_pattern(self, pattern1_id: str, pattern2_id: str) -> Optional[Pattern]:
        """
        功能：查找合并套路结果。
        参数：pattern1_id、pattern2_id。
        返回：合并后的 Pattern 或 None。
        流程：查询 mutation:merge（大小写兼容）。
        说明：用于 Action.merge_patterns。
        """
        p1_key = self._node_key("Pattern", pattern1_id)
        if p1_key not in self.graph:
            return None
        for _src, dst, edge_attrs in self.graph.out_edges(p1_key, data=True):
            if edge_attrs.get("rel") not in {"mutation:merge", "MUTATION:MERGE"}:
                continue
            attrs = self._node_attrs(dst)
            return Pattern.from_dict(attrs)
        return None

    def get_tricks_for_pattern(self, pattern_id: str) -> List[Dict]:
        """
        功能：获取套路关联的 Trick 节点。
        参数：pattern_id。
        返回：Trick 节点属性列表。
        流程：读取 CONTAINS_TRICK 边。
        说明：按边出现顺序返回。
        """
        pattern_key = self._node_key("Pattern", pattern_id)
        if pattern_key not in self.graph:
            return []
        results: List[Dict] = []
        for _src, dst, edge_attrs in self.graph.out_edges(pattern_key, data=True):
            if edge_attrs.get("rel") != "CONTAINS_TRICK":
                continue
            results.append(self._node_attrs(dst))
        return results

    def query_tricks_for_defect(self, defect_type: str, top_k: int = 3) -> List[Trick]:
        """
        功能：根据缺陷类型查询可用于修复的 Tricks。
        参数：defect_type、top_k。
        返回：Trick 列表。
        流程：沿 fixed_by/FIXED_BY 边检索。
        说明：若无匹配缺陷则返回空列表。
        """
        scored: List[Tuple[float, Trick]] = []
        for node_key, attrs in self.graph.nodes(data=True):
            if attrs.get("node_type") != "Defect":
                continue
            if str(attrs.get("type")) != defect_type:
                continue
            for _src, dst, edge_attrs in self.graph.out_edges(node_key, data=True):
                if edge_attrs.get("rel") not in {"fixed_by", "FIXED_BY"}:
                    continue
                score = float(edge_attrs.get("success_rate") or edge_attrs.get("effectiveness") or 0.0)
                scored.append((score, Trick.from_dict(self._node_attrs(dst))))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [trick for _score, trick in scored[:top_k]]

    def query_defects_for_review(self, review_id: str) -> List[Defect]:
        """
        功能：查询某条 Review 指向的缺陷列表。
        参数：review_id。
        返回：Defect 列表。
        流程：沿 identifies/IDENTIFIES 边检索。
        说明：若不存在则返回空列表。
        """
        review_key = self._node_key("Review", review_id)
        if review_key not in self.graph:
            return []
        results: List[Defect] = []
        for _src, dst, edge_attrs in self.graph.out_edges(review_key, data=True):
            if edge_attrs.get("rel") not in {"identifies", "IDENTIFIES"}:
                continue
            results.append(Defect.from_dict(self._node_attrs(dst)))
        return results

    def to_dict(self) -> Dict:
        """
        功能：导出图谱快照为字典（内部格式）。
        参数：无。
        返回：包含 nodes 与 edges 的字典。
        流程：遍历节点/边 → 序列化。
        说明：用于保存内部快照（非 node-link）。
        """
        nodes: List[Dict] = []
        for _node_key, attrs in self.graph.nodes(data=True):
            nodes.append(dict(attrs))

        edges: List[Dict] = []
        for src, dst, attrs in self.graph.edges(data=True):
            src_attrs = self.graph.nodes[src]
            dst_attrs = self.graph.nodes[dst]
            edges.append(
                {
                    "src_type": src_attrs.get("node_type"),
                    "src_id": src_attrs.get("node_id"),
                    "dst_type": dst_attrs.get("node_type"),
                    "dst_id": dst_attrs.get("node_id"),
                    "rel": attrs.get("rel"),
                    "attrs": {k: v for k, v in attrs.items() if k != "rel"},
                }
            )
        return {"nodes": nodes, "edges": edges}

    def load_dict(self, data: Dict) -> None:
        """
        功能：从字典加载图谱。
        参数：data。
        返回：无。
        流程：清空现有图 → 重建节点与边。
        说明：支持 node-link 与内部 nodes/edges 格式。
        """
        self.graph.clear()
        self._pattern_domain_cache.clear()
        if "nodes" in data and "links" in data:
            self._load_node_link(data)
        elif "nodes" in data and "edges" in data:
            self._load_internal_nodes_edges(data)
        else:
            raise ValueError("图谱快照格式不支持，请提供 node-link 或 nodes/edges 格式")
        self._rebuild_openreview_index()

    def _load_node_link(self, data: Dict) -> None:
        """
        功能：加载 node-link 格式图谱。
        参数：data。
        返回：无。
        流程：先建节点 → 再建边。
        说明：node.type/node.id 与 link.relation/source/target 保持原样。
        """
        nodes = data.get("nodes", [])
        links = data.get("links", [])
        node_id_to_type: Dict[str, str] = {}

        for node in nodes:
            node_type = str(node.get("type", ""))
            node_id = str(node.get("id", ""))
            if not node_type or not node_id:
                continue
            node_id_to_type[node_id] = node_type
            attrs = dict(node)
            attrs.pop("type", None)
            self.upsert_node(node_type, node_id, attrs)

        for link in links:
            rel = str(link.get("relation", ""))
            src_id = str(link.get("source", ""))
            dst_id = str(link.get("target", ""))
            if not rel or not src_id or not dst_id:
                continue
            src_type = node_id_to_type.get(src_id)
            dst_type = node_id_to_type.get(dst_id)
            if not src_type or not dst_type:
                continue
            self.upsert_edge(src_type, src_id, rel, dst_type, dst_id, {})

    def _load_internal_nodes_edges(self, data: Dict) -> None:
        """
        功能：加载内部 nodes/edges 格式图谱。
        参数：data。
        返回：无。
        流程：遍历 nodes/edges 重建。
        说明：nodes 应包含 type/id 字段，edges 应包含 src_type/src_id/dst_type/dst_id/rel。
        """
        for node in data.get("nodes", []):
            node_type = node.get("node_type") or node.get("type")
            node_id = node.get("node_id") or node.get("id")
            if not node_type or not node_id:
                continue
            attrs = dict(node)
            # 中文注释：node_type/node_id 是内部字段，需移除；但保留 Defect.type 等业务字段。
            attrs.pop("node_type", None)
            attrs.pop("node_id", None)
            # 中文注释：仅当 "type" 被当作 node_type 使用时才移除，避免覆盖 Defect.type。
            if node.get("node_type") is None and node.get("type") == node_type:
                attrs.pop("type", None)
            self.upsert_node(str(node_type), str(node_id), attrs)

        for edge in data.get("edges", []):
            src_type = edge.get("src_type")
            src_id = edge.get("src_id")
            dst_type = edge.get("dst_type")
            dst_id = edge.get("dst_id")
            rel = edge.get("rel")
            attrs = edge.get("attrs") or {}
            if not (src_type and src_id and dst_type and dst_id and rel):
                continue
            self.upsert_edge(str(src_type), str(src_id), str(rel), str(dst_type), str(dst_id), attrs)
