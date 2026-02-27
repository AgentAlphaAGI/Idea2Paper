"""
三路召回系统 Demo - Idea2Pattern (V3版本)

基于知识图谱的三路召回策略：
  路径1: Idea → Idea → Pattern (相似Idea召回)
  路径2: Idea → Domain → Pattern (领域相关性召回)
  路径3: Idea → Paper → Pattern (相似Paper召回)

V3版本更新:
  - 适配V3节点结构 (Paper.idea为字符串，非嵌套字典)
  - 路径1直接使用Idea.pattern_ids，无需通过Paper中转
  - Paper通过review_stats获取质量分数，支持兼容旧结构
"""

import hashlib
import json
import pickle
import time
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from idea2paper.config import OUTPUT_DIR, PipelineConfig
from idea2paper.infra.embeddings import get_embedding, get_embeddings_batch, EMBEDDING_MODEL
from idea2paper.infra.subdomain_taxonomy import (
    validate_subdomain_taxonomy,
    resolve_subdomain_taxonomy_paths,
)
from idea2paper.recall.recall_text import build_recall_idea_text, build_recall_paper_text, truncate_for_embedding
from idea2paper.recall.tokenize import to_token_set, jaccard_from_sets
from pipeline.run_context import get_logger

# 输入文件
NODES_IDEA = OUTPUT_DIR / "nodes_idea.json"
NODES_PATTERN = OUTPUT_DIR / "nodes_pattern.json"
NODES_DOMAIN = OUTPUT_DIR / "nodes_domain.json"
NODES_PAPER = OUTPUT_DIR / "nodes_paper.json"
EDGES_FILE = OUTPUT_DIR / "edges.json"
GRAPH_FILE = OUTPUT_DIR / "knowledge_graph_v2.gpickle"


# ===================== 召回参数配置 =====================
class RecallConfig:
    """召回系统配置"""
    # 每路召回的Top-K
    PATH1_TOP_K_IDEAS = 20       # 路径1: 召回前K个最相似的Idea
    PATH1_FINAL_TOP_K = 10       # 路径1: 最终只保留Top-K个Pattern（重要通道）

    PATH2_TOP_K_DOMAINS = 5      # 路径2: 召回前K个最相关的Domain
    PATH2_FINAL_TOP_K = 5        # 路径2: 最终只保留Top-K个Pattern（辅助通道）
    PATH2_TOP_K_SUBDOMAINS = 5   # 路径2: 每个Domain保留Top-K个子领域
    PATH2_SUBDOMAIN_BOOST = 1.0  # 子领域命中后对Pattern得分的加成倍率系数
    PATH2_SUBDOMAIN_POOL_SIZE = 50  # 子领域候选池大小（由Pattern子领域压缩而来）

    PATH3_TOP_K_PAPERS = 20      # 路径3: 召回前K个最相似的Paper
    PATH3_FINAL_TOP_K = 10       # 路径3: 最终只保留Top-K个Pattern（重要通道）

    # 各路召回的权重
    PATH1_WEIGHT = 0.4  # 路径1权重（相似Idea - 重要）
    PATH2_WEIGHT = 0.2  # 路径2权重（领域相关 - 辅助）
    PATH3_WEIGHT = 0.4  # 路径3权重（相似Paper - 重要）

    # 最终召回的Top-K
    FINAL_TOP_K = 10

    # 相似度计算方式
    USE_EMBEDDING = True  # 使用embedding计算相似度（推荐），False则使用Jaccard

    # 两阶段召回优化（粗排+精排）
    TWO_STAGE_RECALL = True      # 启用两阶段召回（大幅提速）
    COARSE_RECALL_SIZE = 100     # 粗召回数量（Jaccard快速筛选）
    FINE_RECALL_SIZE = 20        # 精排数量（Embedding精确排序）


# ===================== 召回系统 =====================
class RecallSystem:
    """三路召回系统"""

    def __init__(self, logger=None):
        print("🚀 初始化召回系统...")
        self.logger = logger or get_logger()

        # 加载数据
        self.ideas = self._load_json(NODES_IDEA)
        self.patterns = self._load_json(NODES_PATTERN)
        self.domains = self._load_json(NODES_DOMAIN)
        self.papers = self._load_json(NODES_PAPER)

        # 加载图谱
        with open(GRAPH_FILE, 'rb') as f:
            self.G = pickle.load(f)

        # 构建索引
        self.idea_id_to_idea = {i['idea_id']: i for i in self.ideas}
        self.pattern_id_to_pattern = {p['pattern_id']: p for p in self.patterns}
        self.domain_id_to_domain = {d['domain_id']: d for d in self.domains}
        self.paper_id_to_paper = {p['paper_id']: p for p in self.papers}

        # Subdomain taxonomy (optional)
        self._subdomain_mapping = {}
        self._subdomain_stoplist = set()
        self._subdomain_taxonomy_manifest = {}
        self._subdomain_taxonomy_counts = {"raw_count": 0, "canonical_count": 0, "stoplist_count": 0}
        self._subdomain_stoplist_mode = str(getattr(PipelineConfig, "SUBDOMAIN_TAXONOMY_STOPLIST_MODE", "drop") or "drop").lower()
        self._subdomain_taxonomy_used = False
        self._load_subdomain_taxonomy()

        # Path2: Domain -> Pattern 边索引 & 子领域压缩池
        self._domain_to_pattern_edges = {}
        self._domain_subdomain_pool = {}
        self._domain_subdomain_to_patterns = {}
        for domain in self.domains:
            domain_id = domain.get("domain_id")
            if not domain_id or not self.G.has_node(domain_id):
                continue
            edges = []
            for predecessor in self.G.predecessors(domain_id):
                edge_data = self.G[predecessor][domain_id]
                if edge_data.get("relation") == "works_well_in":
                    edges.append((predecessor, edge_data))
            if not edges:
                continue
            self._domain_to_pattern_edges[domain_id] = edges
            sub_counter = Counter()
            sub_to_patterns = defaultdict(set)
            for pattern_id, _edge in edges:
                pattern = self.pattern_id_to_pattern.get(pattern_id) or {}
                for sd in self._map_subdomains(pattern.get("sub_domains", []) or []):
                    sub_counter[sd] += 1
                    sub_to_patterns[sd].add(pattern_id)
            if sub_counter:
                ranked = sorted(sub_counter.items(), key=lambda x: (-x[1], x[0]))
                self._domain_subdomain_pool[domain_id] = [sd for sd, _ in ranked[:RecallConfig.PATH2_SUBDOMAIN_POOL_SIZE]]
            else:
                self._domain_subdomain_pool[domain_id] = []
            self._domain_subdomain_to_patterns[domain_id] = sub_to_patterns

        self._use_embed_batch = True
        self._use_token_cache = True
        self._use_offline_index = bool(PipelineConfig.RECALL_USE_OFFLINE_INDEX)
        self._embed_batch_size = int(PipelineConfig.RECALL_EMBED_BATCH_SIZE)
        self._embed_max_retries = int(PipelineConfig.RECALL_EMBED_MAX_RETRIES)
        self._embed_sleep_sec = float(PipelineConfig.RECALL_EMBED_SLEEP_SEC)
        self._recall_index_dir = Path(PipelineConfig.RECALL_INDEX_DIR)

        self._offline_index_loaded = False
        self._offline_index_ok = False
        self._offline_index_reason = None
        self._idea_emb = None
        self._idea_meta = None
        self._idea_id_to_idx = {}
        self._paper_emb = None
        self._paper_meta = None
        self._paper_id_to_idx = {}

        self._idea_token_sets = {}
        self._paper_token_sets = {}
        if self._use_token_cache:
            for idea in self.ideas:
                idea_id = idea.get("idea_id")
                if idea_id:
                    self._idea_token_sets[idea_id] = to_token_set(build_recall_idea_text(idea))
            for paper in self.papers:
                paper_id = paper.get("paper_id")
                if paper_id:
                    self._paper_token_sets[paper_id] = to_token_set(build_recall_paper_text(paper))

        print(f"  ✓ 加载 {len(self.ideas)} 个Idea")
        print(f"  ✓ 加载 {len(self.patterns)} 个Pattern")
        print(f"  ✓ 加载 {len(self.domains)} 个Domain")
        print(f"  ✓ 加载 {len(self.papers)} 个Paper")
        print(f"  ✓ 图谱节点: {self.G.number_of_nodes()}, 边: {self.G.number_of_edges()}")
        print()

    def _load_json(self, filepath: Path) -> List[Dict]:
        """加载JSON文件"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _file_hash(self, path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    def _load_subdomain_taxonomy(self):
        if not bool(getattr(PipelineConfig, "SUBDOMAIN_TAXONOMY_ENABLE", False)):
            return
        tax_path, patterns_path = resolve_subdomain_taxonomy_paths()
        if not tax_path.exists():
            if self.logger:
                self.logger.log_event("subdomain_taxonomy_missing", {"path": str(tax_path)})
            return
        status = validate_subdomain_taxonomy(tax_path, patterns_path)
        if not status.get("ok"):
            if self.logger:
                self.logger.log_event("subdomain_taxonomy_mismatch", {
                    "path": str(tax_path),
                    "status": status,
                })
            return
        try:
            data = json.loads(tax_path.read_text(encoding="utf-8"))
            mapping = data.get("mapping") or {}
            stoplist = set(data.get("stoplist") or [])
            if mapping:
                self._subdomain_mapping = mapping
                self._subdomain_stoplist = stoplist
                self._subdomain_taxonomy_manifest = data.get("manifest") or {}
                self._subdomain_taxonomy_counts = {
                    "raw_count": int(data.get("stats", {}).get("raw_count", len(mapping)) or len(mapping)),
                    "canonical_count": int(data.get("stats", {}).get("canonical_count", len(set(mapping.values()))) or len(set(mapping.values()))),
                    "stoplist_count": int(data.get("stats", {}).get("stoplist_count", len(stoplist)) or len(stoplist)),
                }
                self._subdomain_taxonomy_used = True
        except Exception:
            if self.logger:
                self.logger.log_event("subdomain_taxonomy_invalid", {"path": str(tax_path)})

    def _map_subdomains(self, subdomains: List[str]) -> List[str]:
        if not subdomains:
            return []
        if not self._subdomain_mapping:
            return [sd for sd in subdomains if sd]
        mapped = []
        seen = set()
        for sd in subdomains:
            if not sd:
                continue
            canonical = self._subdomain_mapping.get(sd, sd)
            if self._subdomain_stoplist_mode == "drop" and canonical in self._subdomain_stoplist:
                continue
            if canonical not in seen:
                mapped.append(canonical)
                seen.add(canonical)
        return mapped

    def _load_index_kind(self, kind: str, emb_path: Path, meta_path: Path, manifest_path: Path, expected_hash: str):
        if not emb_path.exists() or not meta_path.exists() or not manifest_path.exists():
            return None
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("embedding_model") != EMBEDDING_MODEL:
                return None
            if manifest.get(f"nodes_{kind}_hash") != expected_hash:
                return None
            emb = np.load(emb_path)
            meta = [json.loads(l) for l in meta_path.read_text(encoding="utf-8").splitlines() if l.strip()]
            id_key = f"{kind}_id"
            id_to_idx = {m.get(id_key): i for i, m in enumerate(meta) if m.get(id_key)}
            return {"emb": emb, "meta": meta, "id_to_idx": id_to_idx, "manifest": manifest}
        except Exception:
            return None

    def _load_offline_index(self) -> bool:
        if not self._use_offline_index:
            return False
        if self._offline_index_loaded:
            return self._offline_index_ok

        self._offline_index_loaded = True
        self._offline_index_ok = False

        idea_manifest = self._recall_index_dir / "idea_manifest.json"
        idea_emb = self._recall_index_dir / "idea_emb.npy"
        idea_meta = self._recall_index_dir / "idea_meta.jsonl"

        paper_manifest = self._recall_index_dir / "paper_manifest.json"
        paper_emb = self._recall_index_dir / "paper_emb.npy"
        paper_meta = self._recall_index_dir / "paper_meta.jsonl"

        idea_hash = self._file_hash(NODES_IDEA) if NODES_IDEA.exists() else None
        paper_hash = self._file_hash(NODES_PAPER) if NODES_PAPER.exists() else None

        idea_idx = self._load_index_kind("idea", idea_emb, idea_meta, idea_manifest, idea_hash)
        paper_idx = self._load_index_kind("paper", paper_emb, paper_meta, paper_manifest, paper_hash)

        if not idea_idx or not paper_idx:
            self._offline_index_reason = "missing_or_mismatch"
            if self.logger:
                self.logger.log_event("recall_offline_index_fallback", {
                    "reason": self._offline_index_reason,
                    "index_dir": str(self._recall_index_dir),
                })
            return False

        self._idea_emb = idea_idx["emb"]
        self._idea_meta = idea_idx["meta"]
        self._idea_id_to_idx = idea_idx["id_to_idx"]
        self._paper_emb = paper_idx["emb"]
        self._paper_meta = paper_idx["meta"]
        self._paper_id_to_idx = paper_idx["id_to_idx"]
        self._offline_index_ok = True
        if self.logger:
            self.logger.log_event("recall_offline_index_used", {
                "index_dir": str(self._recall_index_dir),
                "idea_manifest": idea_idx["manifest"],
                "paper_manifest": paper_idx["manifest"],
            })
        return True

    def _get_offline_embeddings(self, kind: str, ids: List[str]):
        if not self._load_offline_index():
            return None
        if kind == "idea":
            id_to_idx = self._idea_id_to_idx
            emb = self._idea_emb
        else:
            id_to_idx = self._paper_id_to_idx
            emb = self._paper_emb
        idxs = []
        for _id in ids:
            idx = id_to_idx.get(_id)
            if idx is None:
                return None
            idxs.append(idx)
        return emb[np.array(idxs, dtype=int)]

    def _cosine_scores(self, query_emb: np.ndarray, cand_embs: np.ndarray) -> List[float]:
        # Use float64 to minimize numeric drift vs. per-item cosine computation.
        q = np.array(query_emb, dtype=float).reshape(-1)
        c = np.array(cand_embs, dtype=float)
        # Edge cases:
        # - no candidates -> empty scores
        # - single candidate returned as 1D vector -> treat as (1, D)
        if c.ndim == 1:
            if c.size == 0:
                return []
            c = c.reshape(1, -1)
        q_norm = np.linalg.norm(q)
        c_norms = np.linalg.norm(c, axis=1)
        c_norms[c_norms == 0] = 1.0
        if q_norm == 0:
            return [0.0 for _ in range(c.shape[0])]
        scores = (c @ q) / (c_norms * q_norm)
        return [float(s) for s in scores]

    def _batch_embeddings(self, texts: List[str]):
        if not texts:
            return []
        payload = [truncate_for_embedding(t) for t in texts]
        for attempt in range(self._embed_max_retries + 1):
            embs = get_embeddings_batch(payload, logger=self.logger, timeout=10)
            if embs is not None:
                return embs
            time.sleep(self._embed_sleep_sec * (attempt + 1))
        return None

    def _compute_embedding_similarities(self, user_idea: str, candidate_ids: List[str], kind: str) -> List[Tuple[str, float]]:
        if kind == "idea":
            texts = [build_recall_idea_text(self.idea_id_to_idea[i]) for i in candidate_ids]
        else:
            texts = [build_recall_paper_text(self.paper_id_to_paper[i]) for i in candidate_ids]

        if not self._use_embed_batch:
            return [(cid, self._compute_embedding_similarity(user_idea, text)) for cid, text in zip(candidate_ids, texts)]

        query_emb = self._get_embedding(truncate_for_embedding(user_idea))
        if query_emb is None:
            return [(cid, self._compute_jaccard_similarity(user_idea, text)) for cid, text in zip(candidate_ids, texts)]

        cand_embs = None
        if self._use_offline_index:
            cand_embs = self._get_offline_embeddings(kind, candidate_ids)
            if cand_embs is None and self.logger:
                self.logger.log_event("recall_offline_index_fallback", {
                    "reason": self._offline_index_reason or "missing_candidate",
                    "index_dir": str(self._recall_index_dir),
                })

        if cand_embs is None:
            cand_embs = self._batch_embeddings(texts)
            if cand_embs is None:
                return [(cid, self._compute_embedding_similarity(user_idea, text)) for cid, text in zip(candidate_ids, texts)]

        scores = self._cosine_scores(query_emb, cand_embs)
        return [(cid, sim) for cid, sim in zip(candidate_ids, scores)]

    def _compute_text_similarity(self, text1: str, text2: str) -> float:
        """计算两个文本的相似度

        支持两种模式:
        1. USE_EMBEDDING=True: 使用Qwen3-Embedding-4B计算语义相似度（推荐）
        2. USE_EMBEDDING=False: 使用词袋Jaccard相似度（快速但不准确）
        """
        if not text1 or not text2:
            return 0.0

        if RecallConfig.USE_EMBEDDING:
            return self._compute_embedding_similarity(text1, text2)
        else:
            return self._compute_jaccard_similarity(text1, text2)

    def _compute_jaccard_similarity(self, text1: str, text2: str) -> float:
        """词袋Jaccard相似度（快速但不准确）"""
        tokens1 = to_token_set(text1)
        tokens2 = to_token_set(text2)
        return jaccard_from_sets(tokens1, tokens2)

    def _compute_embedding_similarity(self, text1: str, text2: str) -> float:
        """基于embedding的余弦相似度（更准确）"""
        # 获取两个文本的embedding
        emb1 = self._get_embedding(text1)
        emb2 = self._get_embedding(text2)

        if emb1 is None or emb2 is None:
            # 降级到Jaccard相似度
            return self._compute_jaccard_similarity(text1, text2)

        # 计算余弦相似度
        emb1 = np.array(emb1)
        emb2 = np.array(emb2)

        cosine_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return float(cosine_sim)

    def _get_embedding(self, text: str, max_retries: int = 3) -> List[float]:
        """调用Embedding接口获取文本embedding（统一走可配置的infra实现）"""
        text = truncate_for_embedding(text)
        for attempt in range(max_retries):
            emb = get_embedding(text, logger=self.logger, timeout=10)
            if emb is not None:
                return emb
            if attempt < max_retries - 1:
                time.sleep(0.5)
        if not hasattr(self, "_embedding_error_shown"):
            print("  ⚠️  Embedding 不可用或调用失败，降级到Jaccard相似度")
            self._embedding_error_shown = True
        return None

    def _get_paper_quality(self, paper: Dict) -> float:
        """计算Paper的综合质量分数

        基于review的评分，归一化到[0, 1]
        如果没有review数据，返回默认值0.5
        """
        # 优先使用新结构中的 review_stats.avg_score
        review_stats = paper.get('review_stats', {})

        if review_stats and review_stats.get('avg_score'):
            # 已经是 0-1 的分数
            return float(review_stats['avg_score'])

        # 备选方案：兼容旧结构（review 列表）
        reviews = paper.get('reviews', [])

        if not reviews:
            return 0.5  # 默认中等质量

        # 提取所有评分
        scores = []
        for review in reviews:
            score_str = review.get('overall_score', '')
            # 尝试解析评分（可能是 "7", "7/10", "7.0" 等格式）
            try:
                if '/' in score_str:
                    score_str = score_str.split('/')[0]
                score = float(score_str.strip())
                scores.append(score)
            except (ValueError, AttributeError):
                continue

        if not scores:
            return 0.5

        # 计算平均分并归一化
        import numpy as np
        avg_score = np.mean(scores)
        # 假设评分范围是 1-10，归一化到 [0, 1]
        normalized_score = (avg_score - 1) / 9

        return min(max(normalized_score, 0.0), 1.0)

    # ===================== 路径1: Idea → Idea → Pattern =====================

    def _recall_path1_similar_ideas(self, user_idea: str) -> Tuple[Dict[str, float], List[Tuple[str, float]]]:
        """路径1: 通过相似Idea召回Pattern (V3版本 + 两阶段优化)

        流程:
          1. 【粗排】使用Jaccard快速筛选Top-N候选（N=100）
          2. 【精排】对候选使用Embedding重新排序，选择Top-K（K=10）
          3. 直接获取这些Idea的pattern_ids
          4. 按相似度加权计算Pattern得分

        返回: (pattern_scores, top_ideas)
            - pattern_scores: {pattern_id: score}
            - top_ideas: [(idea_id, similarity), ...] 用于路径2的Domain查找
        """
        print("\n🔍 [路径1] 相似Idea召回...")

        # Step 1: 粗排 - 使用Jaccard快速筛选
        if RecallConfig.TWO_STAGE_RECALL and RecallConfig.USE_EMBEDDING:
            print(f"  [粗排] 使用Jaccard快速筛选Top-{RecallConfig.COARSE_RECALL_SIZE}...")
            coarse_similarities = []
            user_tokens = to_token_set(user_idea)
            for idea in self.ideas:
                idea_id = idea.get("idea_id")
                if self._use_token_cache and idea_id in self._idea_token_sets:
                    sim = jaccard_from_sets(user_tokens, self._idea_token_sets[idea_id])
                else:
                    sim = self._compute_jaccard_similarity(user_idea, idea.get('description', ''))
                if sim > 0:
                    coarse_similarities.append((idea['idea_id'], sim))

            coarse_similarities.sort(key=lambda x: x[1], reverse=True)
            candidates = coarse_similarities[:RecallConfig.COARSE_RECALL_SIZE]
            self._last_path3_candidates = candidates
            self._last_path1_candidates = candidates

            print(f"  [精排] 使用Embedding重排Top-{RecallConfig.FINE_RECALL_SIZE}...")
            # Step 2: 精排 - 对候选使用Embedding重新计算
            fine_similarities = []
            candidate_ids = [idea_id for idea_id, _ in candidates]
            sims = self._compute_embedding_similarities(user_idea, candidate_ids, kind="idea")
            for idea_id, sim in sims:
                if sim > 0:
                    fine_similarities.append((idea_id, sim))

            fine_similarities.sort(key=lambda x: x[1], reverse=True)
            top_ideas = fine_similarities[:RecallConfig.PATH1_TOP_K_IDEAS]
            self._last_path1_top_ideas = top_ideas

            print(f"  ✓ 粗排{len(coarse_similarities)}个 → 精排{len(candidates)}个 → 最终{len(top_ideas)}个")
        else:
            # 单阶段召回（原逻辑）
            similarities = []
            for idea in self.ideas:
                sim = self._compute_text_similarity(user_idea, idea['description'])
                if sim > 0:
                    similarities.append((idea['idea_id'], sim))

            similarities.sort(key=lambda x: x[1], reverse=True)
            top_ideas = similarities[:RecallConfig.PATH1_TOP_K_IDEAS]
            self._last_path1_candidates = similarities[:RecallConfig.COARSE_RECALL_SIZE]
            self._last_path1_top_ideas = top_ideas
            print(f"  找到 {len(similarities)} 个相似Idea，选择Top-{RecallConfig.PATH1_TOP_K_IDEAS}")

        # Step 3: 直接从Idea节点获取pattern_ids并计算得分
        pattern_scores = defaultdict(float)

        for idea_id, similarity in top_ideas:
            idea = self.idea_id_to_idea[idea_id]
            pattern_ids = idea.get('pattern_ids', [])

            # 打印Idea的前300个字符用于调试
            idea_desc = idea.get('description', '')[:300]
            print(f"  - [{idea_id}] {idea_desc}... (相似度={similarity:.3f}, {len(pattern_ids)}个Pattern)")

            # V3版本: 直接使用Idea节点中的pattern_ids
            for pattern_id in pattern_ids:
                # 得分 = 相似度 (Paper质量暂时默认0.5，已集成在相似度中)
                pattern_scores[pattern_id] += similarity

        # 排序并只保留Top-K个Pattern
        sorted_patterns = sorted(pattern_scores.items(), key=lambda x: x[1], reverse=True)
        top_patterns = dict(sorted_patterns[:RecallConfig.PATH1_FINAL_TOP_K])

        print(f"  ✓ 召回 {len(pattern_scores)} 个Pattern，保留Top-{RecallConfig.PATH1_FINAL_TOP_K}")
        return top_patterns, top_ideas

    # ===================== 路径2: Idea → Domain → Pattern =====================

    def _recall_path2_domain_patterns(self, user_idea: str, top_ideas: List[Tuple[str, float]] = None) -> Dict[str, float]:
        """路径2: 通过领域相关性召回Pattern

        流程:
          1. 【粗层】先匹配一级Domain（embedding 或关键词/Jaccard）
          2. 【精层】在Domain内匹配 sub_domain（embedding 或关键词/Jaccard）
          3. 只从命中 sub_domain 对应的 Pattern 集合中召回并打分

        Args:
            user_idea: 用户输入的Idea描述
            top_ideas: 路径1召回的Top Ideas [(idea_id, similarity), ...]

        返回: {pattern_id: score}
        """
        print("\n🌍 [路径2] 领域相关性召回...")

        # Step 1: 粗层匹配一级Domain
        domain_ids = list(self._domain_to_pattern_edges.keys())
        domain_texts = []
        for domain_id in domain_ids:
            domain = self.domain_id_to_domain.get(domain_id, {})
            name = domain.get("name", "N/A")
            subs = self._domain_subdomain_pool.get(domain_id, []) or []
            if subs:
                domain_texts.append(truncate_for_embedding(f"{name}\nSub-domains: {', '.join(subs)}"))
            else:
                domain_texts.append(truncate_for_embedding(name))

        user_tokens = to_token_set(user_idea)
        query_emb = self._get_embedding(user_idea) if RecallConfig.USE_EMBEDDING else None
        domain_scores = []
        if query_emb is not None:
            cand_embs = self._batch_embeddings(domain_texts)
            if cand_embs is not None:
                sims = self._cosine_scores(query_emb, np.array(cand_embs))
                domain_scores = [(did, max(0.0, float(sim))) for did, sim in zip(domain_ids, sims)]
            else:
                domain_scores = [(did, float(jaccard_from_sets(user_tokens, to_token_set(txt))))
                                 for did, txt in zip(domain_ids, domain_texts)]
        else:
            domain_scores = [(did, float(jaccard_from_sets(user_tokens, to_token_set(txt))))
                             for did, txt in zip(domain_ids, domain_texts)]

        domain_scores.sort(key=lambda x: x[1], reverse=True)
        top_domains = [(did, score) for did, score in domain_scores[:RecallConfig.PATH2_TOP_K_DOMAINS] if score > 0]

        # Fallback：如果完全没有Domain命中，用Top-1 Idea的belongs_to
        if not top_domains and top_ideas:
            top_idea_id = top_ideas[0][0]
            top_idea = self.idea_id_to_idea.get(top_idea_id)
            if top_idea and self.G.has_node(top_idea.get("idea_id")):
                for successor in self.G.successors(top_idea["idea_id"]):
                    edge_data = self.G[top_idea["idea_id"]][successor]
                    if edge_data.get("relation") == "belongs_to":
                        domain_id = successor
                        weight = float(edge_data.get("weight", 0.5))
                        top_domains = [(domain_id, max(0.0, weight))]
                        break

        self._last_path2_top_domains = top_domains
        print(f"  找到 {len(domain_scores)} 个候选Domain，选择Top-{RecallConfig.PATH2_TOP_K_DOMAINS} (有效={len(top_domains)})")

        # Step 2: 精层匹配 sub_domain（每个Domain Top-K）
        domain_to_top_subdomains = {}
        for domain_id, _domain_weight in top_domains:
            cand_subs = self._domain_subdomain_pool.get(domain_id, []) or []
            if not cand_subs:
                domain_to_top_subdomains[domain_id] = []
                continue

            if query_emb is not None:
                sub_embs = self._batch_embeddings(cand_subs)
                if sub_embs is not None:
                    sims = self._cosine_scores(query_emb, np.array(sub_embs))
                    scored = [(sd, max(0.0, float(sim))) for sd, sim in zip(cand_subs, sims)]
                else:
                    scored = [(sd, float(jaccard_from_sets(user_tokens, to_token_set(sd)))) for sd in cand_subs]
            else:
                scored = [(sd, float(jaccard_from_sets(user_tokens, to_token_set(sd)))) for sd in cand_subs]

            scored.sort(key=lambda x: x[1], reverse=True)
            domain_to_top_subdomains[domain_id] = scored[:RecallConfig.PATH2_TOP_K_SUBDOMAINS]

        self._last_path2_top_subdomains = domain_to_top_subdomains

        # Step 3: 从命中 sub_domain 对应的Pattern集合中召回并打分
        pattern_scores = defaultdict(float)
        self._last_path2_candidate_stats = []

        for domain_id, domain_weight in top_domains:
            domain = self.domain_id_to_domain.get(domain_id)
            if not domain:
                continue

            domain_name = domain.get('name', 'N/A')
            paper_count = domain.get('paper_count', 0)
            top_subs = domain_to_top_subdomains.get(domain_id, []) or []
            sub_domain_str = ", ".join([sd for sd, _s in top_subs]) if top_subs else ""

            print(f"  - {domain_id} (名称={domain_name}, 相关度={domain_weight:.3f}, 论文数={paper_count})")
            if sub_domain_str:
                print(f"    命中子领域Top-{RecallConfig.PATH2_TOP_K_SUBDOMAINS}: {sub_domain_str}")

            edges = self._domain_to_pattern_edges.get(domain_id, []) or []
            all_pattern_ids = {pid for pid, _ in edges}

            top_sub_set = {sd for sd, sim in top_subs if sim > 0}
            if top_sub_set:
                sub_to_patterns = self._domain_subdomain_to_patterns.get(domain_id, {})
                candidate_patterns = set()
                for sd in top_sub_set:
                    candidate_patterns.update(sub_to_patterns.get(sd, set()))
            else:
                candidate_patterns = all_pattern_ids

            self._last_path2_candidate_stats.append({
                "domain_id": domain_id,
                "candidates_before": len(all_pattern_ids),
                "candidates_after": len(candidate_patterns),
            })

            edge_map = {pid: edge for pid, edge in edges}
            sd_sim_map = {sd: sim for sd, sim in top_subs}
            for pattern_id in candidate_patterns:
                edge_data = edge_map.get(pattern_id)
                if not edge_data:
                    continue
                effectiveness = float(edge_data.get('effectiveness', 0.0) or 0.0)
                confidence = float(edge_data.get('confidence', 0.0) or 0.0)

                max_sd_sim = 0.0
                pattern = self.pattern_id_to_pattern.get(pattern_id) or {}
                for sd in self._map_subdomains(pattern.get("sub_domains", []) or []):
                    if sd in sd_sim_map:
                        max_sd_sim = max(max_sd_sim, float(sd_sim_map[sd]))

                score = float(domain_weight) * max(effectiveness, 0.1) * confidence
                score *= (1.0 + RecallConfig.PATH2_SUBDOMAIN_BOOST * max_sd_sim)
                pattern_scores[pattern_id] += score

        # 排序并只保留Top-K个Pattern（避免召回过多）
        sorted_patterns = sorted(pattern_scores.items(), key=lambda x: x[1], reverse=True)
        top_patterns = dict(sorted_patterns[:RecallConfig.PATH2_FINAL_TOP_K])

        print(f"  ✓ 召回 {len(pattern_scores)} 个Pattern，保留Top-{RecallConfig.PATH2_FINAL_TOP_K}")
        return top_patterns

    # ===================== 路径3: Idea → Paper → Pattern =====================

    def _recall_path3_similar_papers(self, user_idea: str, raw_idea: str | None = None) -> Dict[str, float]:
        """路径3: 通过相似Paper召回Pattern (V3版本 + 两阶段优化)

        流程:
          1. 【粗排】使用Jaccard快速筛选Top-N候选（N=100）
          2. 【精排】对候选使用Embedding重新排序，选择Top-K（K=20）
          3. 收集这些Paper使用的Pattern
          4. 按Paper相似度和质量加权计算得分

        注意:
          - 使用Paper的title进行相似度计算(与路径1的idea description互补)
          - 粗排(Jaccard)使用 raw_idea（原始短Idea），避免IdeaPackaging扩展后的超长query
            与短title做Jaccard时因分母膨胀（并集>>交集）导致所有候选都低于阈值。
          - 精排(Embedding)使用 user_idea（增强query），语义更丰富，匹配更准确。
          - raw_idea 若未提供，则粗排也回退到 user_idea（兼容未启用IdeaPackaging的调用）
          - V3版本Paper暂无review数据时，质量默认0.5

        返回: {pattern_id: score}
        """
        print("\n📄 [路径3] 相似Paper召回...")

        # 粗排用原始短 idea（防止扩展 query 分母膨胀导致 Jaccard 全部 < 阈值）
        # 精排用增强 query（语义更丰富，能更准确地从候选中找到真正相关的 paper）
        coarse_query = raw_idea if raw_idea else user_idea

        # Step 1: 粗排 - 使用Jaccard快速筛选
        if RecallConfig.TWO_STAGE_RECALL and RecallConfig.USE_EMBEDDING:
            print(f"  [粗排] 使用Jaccard快速筛选Top-{RecallConfig.COARSE_RECALL_SIZE}...")
            coarse_similarities = []
            user_tokens = to_token_set(coarse_query)

            for paper in self.papers:
                paper_title = paper.get('title', '')
                if not paper_title:
                    continue

                paper_id = paper.get("paper_id")
                if self._use_token_cache and paper_id in self._paper_token_sets:
                    sim = jaccard_from_sets(user_tokens, self._paper_token_sets[paper_id])
                else:
                    sim = self._compute_jaccard_similarity(coarse_query, paper_title)
                if sim > 0.05:  # 降低阈值以保留更多候选
                    coarse_similarities.append((paper['paper_id'], sim))

            coarse_similarities.sort(key=lambda x: x[1], reverse=True)
            candidates = coarse_similarities[:RecallConfig.COARSE_RECALL_SIZE]

            print(f"  [精排] 使用Embedding重排Top-{RecallConfig.PATH3_TOP_K_PAPERS}...")
            # Step 2: 精排 - 对候选使用Embedding重新计算（用增强query，语义更丰富）
            fine_similarities = []
            candidate_ids = [paper_id for paper_id, _ in candidates]
            sims = self._compute_embedding_similarities(user_idea, candidate_ids, kind="paper")
            for paper_id, sim in sims:
                if sim > 0.1:  # 过滤低相似度
                    paper = self.paper_id_to_paper[paper_id]
                    quality = self._get_paper_quality(paper)
                    combined_weight = sim * quality
                    fine_similarities.append((paper_id, sim, quality, combined_weight))

            fine_similarities.sort(key=lambda x: x[3], reverse=True)
            top_papers = fine_similarities[:RecallConfig.PATH3_TOP_K_PAPERS]

            print(f"  ✓ 粗排{len(coarse_similarities)}个 → 精排{len(candidates)}个 → 最终{len(top_papers)}个")
        else:
            # 单阶段召回（原逻辑，仅用 user_idea）
            similarities = []

            for paper in self.papers:
                paper_title = paper.get('title', '')
                if not paper_title:
                    continue

                sim = self._compute_text_similarity(user_idea, paper_title)
                if sim > 0.1:  # 过滤低相似度
                    quality = self._get_paper_quality(paper)
                    combined_weight = sim * quality
                    similarities.append((paper['paper_id'], sim, quality, combined_weight))

            similarities.sort(key=lambda x: x[3], reverse=True)
            top_papers = similarities[:RecallConfig.PATH3_TOP_K_PAPERS]
            self._last_path3_candidates = similarities[:RecallConfig.COARSE_RECALL_SIZE]

            print(f"  找到 {len(similarities)} 个相似Paper，选择Top-{RecallConfig.PATH3_TOP_K_PAPERS}")

        # 缓存用于审计
        self._last_path3_top_papers = top_papers

        # Step 3: 收集Pattern
        pattern_scores = defaultdict(float)

        for paper_id, similarity, quality, combined_weight in top_papers:
            paper = self.paper_id_to_paper.get(paper_id, {})
            # 判断质量来源：优先检查review_stats，然后是reviews，否则是默认值
            if paper.get('review_stats'):
                quality_source = f"review({paper['review_stats'].get('review_count', 0)}条)"
            elif paper.get('reviews'):
                quality_source = "review"
            else:
                quality_source = "默认"
            title = paper.get('title', 'N/A')
            print(f"  - {paper_id} (相似度={similarity:.3f}, 质量={quality:.3f} [{quality_source}])")
            print(f"    标题: {title}")

            # 从图谱中找到Paper使用的Pattern
            if not self.G.has_node(paper_id):
                continue

            for successor in self.G.successors(paper_id):
                edge_data = self.G[paper_id][successor]
                if edge_data.get('relation') == 'uses_pattern':
                    pattern_id = successor
                    pattern_quality = edge_data.get('quality', 0.5)

                    # 得分 = Paper相似度 × Paper质量 × Pattern质量
                    score = combined_weight * pattern_quality
                    pattern_scores[pattern_id] += score

        # 排序并只保留Top-K个Pattern
        sorted_patterns = sorted(pattern_scores.items(), key=lambda x: x[1], reverse=True)
        top_patterns = dict(sorted_patterns[:RecallConfig.PATH3_FINAL_TOP_K])

        print(f"  ✓ 召回 {len(pattern_scores)} 个Pattern，保留Top-{RecallConfig.PATH3_FINAL_TOP_K}")
        return top_patterns

    # ===================== 审计工具 =====================

    def _truncate(self, text: str, n: int) -> str:
        if not text:
            return ""
        if len(text) <= n:
            return text
        return text[:n] + "…"

    def _topn_dict(self, d: Dict[str, float], n: int, key_name: str = "pattern_id") -> List[Dict]:
        ranked = sorted(d.items(), key=lambda x: x[1], reverse=True)[:n]
        return [{key_name: k, "score": v} for k, v in ranked]

    # ===================== 多路融合 =====================

    def recall(self, user_idea: str, verbose: bool = True, raw_idea: str | None = None) -> List[Tuple[str, Dict, float]]:
        """三路召回融合

        Args:
            user_idea: 用户输入的Idea描述（可能是经IdeaPackaging扩展后的检索query）
            verbose: 是否打印详细信息
            raw_idea: 用户原始的短Idea文本（未经扩展），用于路径3的Jaccard/Embedding匹配。
                      若为None则路径3使用user_idea（兼容旧调用）。

        Returns:
            [(pattern_id, pattern_info, score), ...] 按得分排序
        """
        print("=" * 80)
        print("🎯 开始三路召回")
        print("=" * 80)
        print(f"\n【用户Idea】\n{user_idea}\n")
        if self.logger:
            self.logger.log_event("recall_start", {"user_idea": user_idea})

        # 路径1: 相似Idea召回
        path1_scores, top_ideas = self._recall_path1_similar_ideas(user_idea)

        # 路径2: 领域相关性召回（使用路径1的Top Ideas）
        path2_scores = self._recall_path2_domain_patterns(user_idea, top_ideas=top_ideas)

        # 路径3: 相似Paper召回
        # 传入 raw_idea，避免扩展 query 因与短 title 做 Jaccard 时分母膨胀导致粗排结果为空
        path3_scores = self._recall_path3_similar_papers(user_idea, raw_idea=raw_idea)

        # 融合三路得分
        print("\n🔗 融合三路召回结果...")
        all_patterns = set(path1_scores.keys()) | set(path2_scores.keys()) | set(path3_scores.keys())

        final_scores = {}
        for pattern_id in all_patterns:
            score1 = path1_scores.get(pattern_id, 0.0) * RecallConfig.PATH1_WEIGHT
            score2 = path2_scores.get(pattern_id, 0.0) * RecallConfig.PATH2_WEIGHT
            score3 = path3_scores.get(pattern_id, 0.0) * RecallConfig.PATH3_WEIGHT

            final_scores[pattern_id] = score1 + score2 + score3

        # 排序并返回Top-K
        ranked = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        top_k = ranked[:RecallConfig.FINAL_TOP_K]

        # 构建返回结果
        results = []
        for pattern_id, score in top_k:
            pattern_info = self.pattern_id_to_pattern.get(pattern_id, {})
            results.append((pattern_id, pattern_info, score))

        # 打印结果
        if verbose:
            self._print_results(results, path1_scores, path2_scores, path3_scores)

        # 召回审计（可选）
        if PipelineConfig.RECALL_AUDIT_ENABLE:
            topn = max(0, int(PipelineConfig.RECALL_AUDIT_TOPN))
            snippet_len = max(0, int(PipelineConfig.RECALL_AUDIT_SNIPPET_CHARS))

            # 路径1 top ideas
            top_ideas_audit = []
            for idea_id, sim in top_ideas[:RecallConfig.PATH1_TOP_K_IDEAS]:
                idea = self.idea_id_to_idea.get(idea_id, {})
                desc = idea.get("description", "")
                pattern_ids = idea.get("pattern_ids", []) or []
                top_ideas_audit.append({
                    "idea_id": idea_id,
                    "similarity": float(sim),
                    "snippet": self._truncate(desc, snippet_len),
                    "pattern_count": len(pattern_ids),
                })

            # 路径2 top domains
            top_domains_raw = getattr(self, "_last_path2_top_domains", []) or []
            top_domains_audit = []
            for domain_id, weight in top_domains_raw[:RecallConfig.PATH2_TOP_K_DOMAINS]:
                domain = self.domain_id_to_domain.get(domain_id, {})
                top_domains_audit.append({
                    "domain_id": domain_id,
                    "name": domain.get("name", ""),
                    "weight": float(weight),
                    "paper_count": int(domain.get("paper_count", 0) or 0),
                })
            # 路径2 top subdomains
            top_subdomains_raw = getattr(self, "_last_path2_top_subdomains", {}) or {}
            top_subdomains_audit = []
            for domain_id, _weight in top_domains_raw[:RecallConfig.PATH2_TOP_K_DOMAINS]:
                subs = top_subdomains_raw.get(domain_id, [])[:RecallConfig.PATH2_TOP_K_SUBDOMAINS]
                top_subdomains_audit.append({
                    "domain_id": domain_id,
                    "subdomains": [{"name": sd, "score": float(sim)} for sd, sim in subs],
                })
            candidate_stats_raw = getattr(self, "_last_path2_candidate_stats", []) or []
            candidate_stats_map = {s.get("domain_id"): s for s in candidate_stats_raw}
            taxonomy_stats = getattr(self, "_subdomain_taxonomy_counts", {}) or {}
            taxonomy_used = bool(getattr(self, "_subdomain_taxonomy_used", False))

            # 路径3 top papers
            top_papers_raw = getattr(self, "_last_path3_top_papers", []) or []
            top_papers_audit = []
            for paper_id, sim, quality, _combined in top_papers_raw[:RecallConfig.PATH3_TOP_K_PAPERS]:
                paper = self.paper_id_to_paper.get(paper_id, {})
                review_stats = paper.get("review_stats") or {}
                top_papers_audit.append({
                    "paper_id": paper_id,
                    "similarity": float(sim),
                    "title": paper.get("title", ""),
                    "quality": float(quality),
                    "review_count": int(review_stats.get("review_count", 0) or 0),
                })

            # 记录各路 score 的 Top-N（加权后的分数）
            path1_weighted = {k: v * RecallConfig.PATH1_WEIGHT for k, v in path1_scores.items()}
            path2_weighted = {k: v * RecallConfig.PATH2_WEIGHT for k, v in path2_scores.items()}
            path3_weighted = {k: v * RecallConfig.PATH3_WEIGHT for k, v in path3_scores.items()}

            final_top_k_audit = []
            for pattern_id, final_score in top_k:
                pattern_info = self.pattern_id_to_pattern.get(pattern_id, {})
                final_top_k_audit.append({
                    "pattern_id": pattern_id,
                    "name": pattern_info.get("name", "N/A"),
                    "final_score": float(final_score),
                    "path1_score": float(path1_weighted.get(pattern_id, 0.0)),
                    "path2_score": float(path2_weighted.get(pattern_id, 0.0)),
                    "path3_score": float(path3_weighted.get(pattern_id, 0.0)),
                    "cluster_size": int(pattern_info.get("size", 0) or 0),
                })

            self.last_audit = {
                "final_top_k": final_top_k_audit,
                "path1": {
                    "top_ideas": top_ideas_audit,
                    "pattern_scores_topn": self._topn_dict(path1_weighted, topn),
                },
                "path2": {
                    "top_domains": top_domains_audit,
                    "top_subdomains": top_subdomains_audit,
                    "candidate_stats": [candidate_stats_map.get(d.get("domain_id")) for d in top_domains_audit],
                    "pattern_scores_topn": self._topn_dict(path2_weighted, topn),
                    "subdomain_taxonomy_used": taxonomy_used,
                    "raw_subdomain_count": int(taxonomy_stats.get("raw_count", 0) or 0),
                    "canonical_subdomain_count": int(taxonomy_stats.get("canonical_count", 0) or 0),
                    "stoplist_count": int(taxonomy_stats.get("stoplist_count", 0) or 0),
                },
                "path3": {
                    "top_papers": top_papers_audit,
                    "pattern_scores_topn": self._topn_dict(path3_weighted, topn),
                },
            }
        else:
            self.last_audit = None

        if self.logger:
            self.logger.log_event("recall_end", {"top_k": len(results)})

        return results

    def _print_results(self, results: List[Tuple[str, Dict, float]],
                      path1_scores: Dict, path2_scores: Dict, path3_scores: Dict):
        """打印召回结果"""
        print("\n" + "=" * 80)
        print(f"📊 召回结果 Top-{RecallConfig.FINAL_TOP_K}")
        print("=" * 80)

        for rank, (pattern_id, pattern_info, final_score) in enumerate(results, 1):
            print(f"\n【Rank {rank}】 {pattern_id}")
            print(f"  名称: {pattern_info.get('name', 'N/A')}")
            print(f"  最终得分: {final_score:.4f}")

            # 显示各路得分
            score1 = path1_scores.get(pattern_id, 0.0) * RecallConfig.PATH1_WEIGHT
            score2 = path2_scores.get(pattern_id, 0.0) * RecallConfig.PATH2_WEIGHT
            score3 = path3_scores.get(pattern_id, 0.0) * RecallConfig.PATH3_WEIGHT

            print(f"  - 路径1 (相似Idea):   {score1:.4f} (占比 {score1/final_score*100:.1f}%)")
            print(f"  - 路径2 (领域相关):   {score2:.4f} (占比 {score2/final_score*100:.1f}%)")
            print(f"  - 路径3 (相似Paper):  {score3:.4f} (占比 {score3/final_score*100:.1f}%)")

            print(f"  聚类大小: {pattern_info.get('size', 0)} 篇论文")

            # V3版本: 优先显示LLM增强的总结，否则显示原始示例
            if pattern_info.get('llm_enhanced_summary'):
                llm_summary = pattern_info['llm_enhanced_summary'].get('representative_ideas', '')
                print(f"  归纳总结: {llm_summary[:120]}...")
            else:
                summary = pattern_info.get('summary', {})
                ideas = summary.get('representative_ideas', [])
                if ideas:
                    print(f"  示例Idea: {ideas[0][:120] if ideas else 'N/A'}...")

        print("\n" + "=" * 80)


# ===================== Demo 测试用例 =====================
def demo():
    """运行Demo"""

    # 初始化召回系统
    system = RecallSystem()

    # 测试用例
    test_ideas = [
        "使用Transformer模型进行文本分类任务，在多个数据集上验证效果",
        "提出一种新的注意力机制改进神经机器翻译的对齐质量",
        "通过对抗训练提升模型在对话系统中的鲁棒性",
        "利用知识图谱增强预训练语言模型的语义理解能力",
    ]

    for i, user_idea in enumerate(test_ideas, 1):
        print("\n\n")
        print("🎬" * 40)
        print(f"测试用例 {i}/{len(test_ideas)}")
        print("🎬" * 40)

        results = system.recall(user_idea, verbose=True)

        # 等待用户查看结果
        if i < len(test_ideas):
            input("\n按Enter继续下一个测试用例...")


if __name__ == '__main__':
    demo()
