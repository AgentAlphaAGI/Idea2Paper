# 中文注释: 本文件实现论文链路的 LangGraph 状态机 + 多 Reviewer 评审反馈循环。
from __future__ import annotations

import logging
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Set, Tuple, TypedDict

from langgraph.graph import END, StateGraph
from langchain_core.language_models.chat_models import BaseChatModel

from core.config import PaperWorkflowConfig
from paper_kg.input_parser import PaperInputParser
from paper_kg.interfaces import IEmbeddingModel, IGraphStore, IVectorStore
from paper_kg.citations.claim_resolver import resolve_claims_in_sections
from paper_kg.citations.extractor import extract_cite_candidates_from_sections
from paper_kg.citations.models import CitationCandidate, CitationNeed, RetrievedCitation
from paper_kg.citations.need_reviewer import CitationNeedReviewer
from paper_kg.citations.retriever import ICitationProvider, get_citation_provider
from paper_kg.latex_renderer import render_sections_to_latex
from paper_kg.latex_fixer import LatexFixer
from paper_kg.latex_toolchain import ILatexToolchain
from paper_kg.llm_schemas import PaperIssueType, PaperReviewDefect
from paper_kg.models import (
    Domain,
    NoveltyCheckResult,
    PaperPlan,
    PaperSectionPlan,
    PaperSectionOutput,
    Pattern,
    SimilarItem,
    StoryDraft,
    Trick,
)
from paper_kg.novelty import encode_pattern_config
from paper_kg.orchestrator import PaperDraftWriter
from paper_kg.overleaf_project import build_overleaf_project
from paper_kg.paper_plan_builder import build_paper_plan
from paper_kg.paper_template import load_paper_template
from paper_kg.paper_reviewers import (
    PaperReviewResult,
    PaperReviewerGroup,
    aggregate_defects,
    aggregate_reviews,
    pick_fix_tricks_for_defects,
)
from paper_kg.section_writer import PaperSectionWriter
from paper_kg.sampling import downweight_scores, select_patterns_diverse
from paper_kg.workflow_report import PaperWorkflowReport


logger = logging.getLogger(__name__)


class ICitationRetrievalAgent(Protocol):
    """引用检索 Agent 接口（LLM + 工具）。"""

    def retrieve(
        self, needs: List[CitationNeed], candidates: List[CitationCandidate]
    ) -> List[RetrievedCitation]:
        """根据需求与候选点检索真实引用。"""


class IClaimResolutionAgent(Protocol):
    """Claim 降级 Agent 接口（LLM）。"""

    def resolve(
        self,
        sections: List[Tuple[str, str]],
        candidates: List[CitationCandidate],
        unresolved_ids: List[str],
    ) -> List[Tuple[str, str]]:
        """对未命中的引用做降级改写。"""


class ILatexRendererAgent(Protocol):
    """LaTeX 渲染与修复 Agent 接口（LLM）。"""

    def render_sections(
        self, sections: List[PaperSectionOutput], citation_map: Dict[str, str]
    ) -> Tuple[List[PaperSectionOutput], str]:
        """渲染章节 Markdown 为 LaTeX。"""

    def fix_sections(
        self,
        sections: List[PaperSectionOutput],
        compile_log: str,
        citation_map: Dict[str, str],
        targets: Optional[Set[str]] = None,
    ) -> Tuple[List[PaperSectionOutput], str]:
        """根据编译日志修复 LaTeX。"""


class PaperWorkflowStateData(TypedDict):
    """
    功能：LangGraph 状态数据定义（论文链路）。
    参数：见字段定义。
    返回：用于 StateGraph 的 TypedDict。
    流程：每个节点读写此状态，实现可回退的工作流。
    说明：字段尽量保持“可序列化/可打印”，便于日志与调试。
    """

    user_input: str
    core_idea: str
    domain: str
    idea_id: str
    domain_id: str

    candidate_pattern_ids: List[str]
    candidate_pattern_scores: List[float]

    selected_patterns: List[Pattern]
    selected_tricks: List[Trick]
    pattern_guides: str

    paper_plan: Optional[PaperPlan]
    paper_sections: List[PaperSectionOutput]
    paper_markdown: str
    paper_latex: str
    section_summaries: Dict[str, str]
    rewrite_targets: List[str]

    citation_candidates: List[Dict]
    citation_needs: List[Dict]
    citation_results: List[Dict]
    citation_rounds: int
    unresolved_citations: List[str]
    refs_bib_text: str
    citations_report: Dict[str, object]

    overleaf_project_dir: str
    overleaf_zip_path: str
    latex_compile_report: Dict[str, object]
    latex_format_retry: int

    novelty_candidates: List[Dict]
    novelty_report: Optional[NoveltyCheckResult]

    draft: Optional[StoryDraft]
    reviews: List[PaperReviewResult]

    defects_agg: List[PaperReviewDefect]
    fix_tricks: List[Trick]

    plan_retry: int
    rewrite_retry: int
    global_retry: int
    review_iterations: int
    draft_attempts: int
    warnings: List[str]
    status: str
    next_step: str


class PaperFeedbackLoopController:
    """
    功能：论文链路 LangGraph 控制器（选套路→查重→蓝图→分章节→拼接→多审稿→回退/通过）。
    参数：config、graph_store、vector_store、embedding_model、rng、input_parser、draft_writer、reviewer_group。
    返回：PaperWorkflowReport。
    流程：构建 LangGraph → invoke → 汇总 report。
    说明：该控制器不做 Router/意图识别，CLI 负责决定走论文链路。
    """

    def __init__(
        self,
        config: PaperWorkflowConfig,
        graph_store: IGraphStore,
        vector_store: IVectorStore,
        embedding_model: IEmbeddingModel,
        rng: random.Random,
        input_parser: PaperInputParser,
        draft_writer: PaperDraftWriter,
        reviewer_group: PaperReviewerGroup,
        section_llm: Optional[BaseChatModel] = None,
        citation_need_llm: Optional[BaseChatModel] = None,
        citation_llm: Optional[BaseChatModel] = None,
        citation_retrieval_agent: Optional[ICitationRetrievalAgent] = None,
        citation_provider: Optional[ICitationProvider] = None,
        claim_resolution_agent: Optional[IClaimResolutionAgent] = None,
        claim_resolver=None,
        latex_renderer_agent: Optional[ILatexRendererAgent] = None,
        paper_template: Optional[PaperPlan] = None,
        latex_toolchain: Optional[ILatexToolchain] = None,
        latex_fixer: Optional[LatexFixer] = None,
    ) -> None:
        """
        功能：初始化论文工作流控制器。
        参数：同类属性 + section_llm + paper_template + latex_toolchain + latex_fixer。
        返回：无。
        流程：保存依赖 → 编译 LangGraph。
        说明：所有阈值与重试次数由 config 控制。
        """
        self.config = config
        self.graph_store = graph_store
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.rng = rng
        self.input_parser = input_parser
        self.draft_writer = draft_writer
        self.reviewer_group = reviewer_group
        self.section_llm = section_llm
        self.citation_llm = citation_need_llm or citation_llm
        self.citation_provider = citation_provider
        self.citation_retrieval_agent = citation_retrieval_agent
        self.citation_reviewer = CitationNeedReviewer(llm=self.citation_llm)
        self.claim_resolution_agent = claim_resolution_agent
        self.claim_resolver = claim_resolver or resolve_claims_in_sections
        self.latex_renderer_agent = latex_renderer_agent
        self.paper_template = paper_template
        self.latex_toolchain = latex_toolchain
        self.latex_fixer = latex_fixer
        self.graph = self._build_graph()

    def init_state(self, user_input: str) -> PaperWorkflowStateData:
        """
        功能：对外暴露的“初始化状态”方法（供 Web UI/SSE 等调用）。
        参数：user_input（用户输入的一句自然语言）。
        返回：PaperWorkflowStateData。
        流程：调用内部 _init_state 构造完整字段的初始状态。
        说明：
        - 该方法是对私有实现的稳定封装，避免外部代码直接依赖 _init_state。
        - 返回值可直接用于 self.graph.stream / self.graph.invoke。
        """
        return self._init_state(user_input)

    def estimate_recursion_limit(self) -> int:
        """
        功能：对外暴露的“递归限制估算”方法。
        参数：无。
        返回：recursion_limit（整数）。
        流程：调用内部 _estimate_recursion_limit。
        说明：用于 LangGraph invoke/stream 的 config，避免默认限制过低导致提前失败。
        """
        return self._estimate_recursion_limit()

    def build_report_from_state(self, state: PaperWorkflowStateData) -> PaperWorkflowReport:
        """
        功能：把 LangGraph 最终状态转换为报告（供 Web UI/CLI 复用）。
        参数：state（工作流最终状态）。
        返回：PaperWorkflowReport。
        流程：调用内部 _build_report。
        说明：对外提供稳定方法，避免调用方依赖私有实现细节。
        """
        return self._build_report(state)

    def _build_graph(self):
        """
        功能：构建 LangGraph 状态机图。
        参数：无。
        返回：编译后的图对象。
        流程：注册节点 → 定义边与条件 → compile。
        说明：节点命名尽量与步骤一致，便于对齐工作流写法。
        """
        graph = StateGraph(PaperWorkflowStateData)
        graph.add_node("parse_input", self._parse_input_node)
        graph.add_node("plan_from_kg", self._plan_from_kg_node)
        graph.add_node("retrieve_similar", self._retrieve_similar_node)
        graph.add_node("novelty_check", self._novelty_check_node)
        graph.add_node("build_blueprint", self._build_blueprint_node)
        graph.add_node("write_abstract", self._write_abstract_node)
        graph.add_node("write_introduction", self._write_introduction_node)
        graph.add_node("write_related_work", self._write_related_work_node)
        graph.add_node("write_method", self._write_method_node)
        graph.add_node("write_experiments", self._write_experiments_node)
        graph.add_node("write_conclusion", self._write_conclusion_node)
        graph.add_node("write_limitations", self._write_limitations_node)
        graph.add_node("write_ethics", self._write_ethics_node)
        graph.add_node("assemble_paper", self._assemble_paper_node)
        graph.add_node("extract_cite_candidates", self._extract_cite_candidates_node)
        graph.add_node("citation_need_review", self._citation_need_review_node)
        graph.add_node("citation_retrieval", self._citation_retrieval_node)
        graph.add_node("claim_resolution", self._claim_resolution_node)
        graph.add_node("render_latex", self._render_latex_node)
        graph.add_node("export_overleaf", self._export_overleaf_node)
        graph.add_node("compile_overleaf", self._compile_overleaf_node)
        graph.add_node("review", self._review_node)
        graph.add_node("decide", self._decide_node)

        graph.set_entry_point("parse_input")
        graph.add_edge("parse_input", "plan_from_kg")
        graph.add_conditional_edges(
            "plan_from_kg",
            self._post_plan_route,
            {"retrieve_similar": "retrieve_similar", "build_blueprint": "build_blueprint"},
        )
        graph.add_edge("retrieve_similar", "novelty_check")
        graph.add_conditional_edges(
            "novelty_check",
            self._novelty_route,
            {"plan_from_kg": "plan_from_kg", "build_blueprint": "build_blueprint"},
        )
        graph.add_edge("build_blueprint", "write_abstract")
        graph.add_edge("write_abstract", "write_introduction")
        graph.add_edge("write_introduction", "write_related_work")
        graph.add_edge("write_related_work", "write_method")
        graph.add_edge("write_method", "write_experiments")
        graph.add_edge("write_experiments", "write_conclusion")
        graph.add_edge("write_conclusion", "write_limitations")
        graph.add_edge("write_limitations", "write_ethics")
        graph.add_edge("write_ethics", "assemble_paper")
        graph.add_conditional_edges(
            "assemble_paper",
            self._post_assemble_route,
            {
                "review": "review",
                "complete": END,
                "export_overleaf": "export_overleaf",
                "extract_cite_candidates": "extract_cite_candidates",
                "render_latex": "render_latex",
            },
        )
        graph.add_conditional_edges(
            "extract_cite_candidates",
            self._post_extract_cite_candidates_route,
            {"citation_need_review": "citation_need_review", "render_latex": "render_latex"},
        )
        graph.add_edge("citation_need_review", "citation_retrieval")
        graph.add_conditional_edges(
            "citation_retrieval",
            self._post_citation_retrieval_route,
            {"claim_resolution": "claim_resolution", "render_latex": "render_latex"},
        )
        graph.add_conditional_edges(
            "claim_resolution",
            self._post_claim_resolution_route,
            {
                "extract_cite_candidates": "extract_cite_candidates",
                "render_latex": "render_latex",
                "failed": END,
            },
        )
        graph.add_edge("render_latex", "export_overleaf")
        graph.add_edge("export_overleaf", "compile_overleaf")
        graph.add_conditional_edges(
            "compile_overleaf",
            self._post_compile_route,
            {"review": "review", "complete": END, "render_latex": "render_latex", "failed": END},
        )
        graph.add_edge("review", "decide")
        graph.add_conditional_edges(
            "decide",
            self._decide_route,
            {
                "complete": END,
                "failed": END,
                "write_abstract": "write_abstract",
                "write_method": "write_method",
                "write_experiments": "write_experiments",
                "plan_from_kg": "plan_from_kg",
            },
        )
        return graph.compile()

    def run(self, user_input: str) -> PaperWorkflowReport:
        """
        功能：执行论文工作流并返回最终报告。
        参数：user_input（用户一句话想法）。
        返回：PaperWorkflowReport。
        流程：初始化 state → graph.invoke → state→report。
        说明：工作流失败也会尽量返回当前 draft 与日志信息。
        """
        state = self._init_state(user_input)
        # 中文注释：工作流包含多次回退与重采样，默认递归限制（25）可能不足；
        # 这里根据配置估算一个更稳妥的 recursion_limit，避免无意义的 GraphRecursionError。
        recursion_limit = self._estimate_recursion_limit()
        final_state: PaperWorkflowStateData = self.graph.invoke(
            state, config={"recursion_limit": recursion_limit}
        )  # type: ignore[assignment]
        return self._build_report(final_state)

    def _init_state(self, user_input: str) -> PaperWorkflowStateData:
        """
        功能：构造 LangGraph 初始状态。
        参数：user_input。
        返回：PaperWorkflowStateData。
        流程：填充默认值，确保所有字段存在。
        说明：避免节点中频繁使用 get 并导致类型不稳定。
        """
        return {
            "user_input": user_input,
            "core_idea": "",
            "domain": "",
            "idea_id": "",
            "domain_id": "",
            "candidate_pattern_ids": [],
            "candidate_pattern_scores": [],
            "selected_patterns": [],
            "selected_tricks": [],
            "pattern_guides": "",
            "paper_plan": None,
            "paper_sections": [],
            "paper_markdown": "",
            "paper_latex": "",
            "section_summaries": {},
            "rewrite_targets": [],
            "citation_candidates": [],
            "citation_needs": [],
            "citation_results": [],
            "citation_rounds": 0,
            "unresolved_citations": [],
            "refs_bib_text": "",
            "citations_report": {},
            "overleaf_project_dir": "",
            "overleaf_zip_path": "",
            "latex_compile_report": {},
            "latex_format_retry": 0,
            "novelty_candidates": [],
            "novelty_report": None,
            "draft": None,
            "reviews": [],
            "defects_agg": [],
            "fix_tricks": [],
            "plan_retry": 0,
            "rewrite_retry": 0,
            "global_retry": 0,
            "review_iterations": 0,
            "draft_attempts": 0,
            "warnings": [],
            "status": "RUNNING",
            "next_step": "",
        }

    def _estimate_recursion_limit(self) -> int:
        """
        功能：估算 LangGraph 运行所需的递归步数上限。
        参数：无。
        返回：recursion_limit 整数。
        流程：根据最大重试与迭代次数估算，并加上安全裕度。
        说明：该值用于 graph.invoke 的 config，避免默认 25 过低导致提前失败。
        """
        base = 40
        # 每次迭代可能触发：plan/novelty 循环 + write + review + decide
        # 中文注释：新增分章节节点后，单次迭代的节点数明显增多，适度提高估算值。
        per_iter = 16 + int(self.config.max_plan_retry) * 3
        total = base + per_iter * (int(self.config.max_iterations) + 1)
        # 额外覆盖 global/rewrite 重试的增量空间
        total += 5 * (int(self.config.max_global_retry) + int(self.config.max_rewrite_retry))
        # 中文注释：引用检索/降级与 LaTeX 格式自循环的额外空间。
        total += 4 * int(self.config.citations_max_rounds)
        total += 3 * int(self.config.latex_format_max_retries)
        return max(80, total)

    def _build_report(self, state: PaperWorkflowStateData) -> PaperWorkflowReport:
        """
        功能：把最终 state 转换为 PaperWorkflowReport。
        参数：state。
        返回：PaperWorkflowReport。
        流程：提取字段 → 组装 report。
        说明：CLI 会进一步格式化输出最终稿。
        """
        return PaperWorkflowReport(
            status=state.get("status", "FAILED"),
            core_idea=state.get("core_idea", ""),
            domain=state.get("domain", ""),
            patterns=state.get("selected_patterns", []),
            tricks=state.get("selected_tricks", []),
            novelty_report=state.get("novelty_report"),
            draft=state.get("draft"),
            reviews=state.get("reviews", []),
            metrics={
                "plan_retry": state.get("plan_retry", 0),
                "rewrite_retry": state.get("rewrite_retry", 0),
                "global_retry": state.get("global_retry", 0),
                "review_iterations": state.get("review_iterations", 0),
                "draft_attempts": state.get("draft_attempts", 0),
                "sections_written": len(state.get("paper_sections", [])),
                "paper_markdown_chars": len(state.get("paper_markdown", "") or ""),
                "citation_rounds": state.get("citation_rounds", 0),
                "latex_format_retry": state.get("latex_format_retry", 0),
            },
            warnings=state.get("warnings", []),
        )

    def _parse_input_node(self, state: PaperWorkflowStateData) -> Dict:
        """
        功能：解析用户输入，抽取 core_idea 与 domain。
        参数：state。
        返回：更新字段的 dict。
        流程：调用 PaperInputParser → 归一化 domain_id/idea_id → 打日志。
        说明：不做 Router；domain 只在 NLP/CV 中选择，默认 NLP。
        """
        core_idea, domain_hint = self.input_parser.parse(state["user_input"])
        domain_id, domain_name = _resolve_domain_id(self.graph_store, domain_hint)
        idea_id = _resolve_idea_id(self.graph_store, core_idea) or "custom"

        logger.info("PAPER_PARSE: domain=%s core_idea=%s", domain_name, core_idea)
        return {
            "core_idea": core_idea,
            "domain": domain_name,
            "domain_id": domain_id,
            "idea_id": idea_id,
        }

    def _plan_from_kg_node(self, state: PaperWorkflowStateData) -> Dict:
        """
        功能：从 KG 里选择 story patterns + tricks（多样性采样）。
        参数：state。
        返回：更新 selected_patterns/selected_tricks/candidate_pattern_scores 等。
        流程：查询候选 → 计算/复用分数 →（可选）根据 fix_tricks 做加权 → softmax 无放回采样。
        说明：当进入“硬问题回退”时，可通过 fix_tricks 引导选择更针对性的套路。
        """
        domain_id = state["domain_id"]
        candidates = self.graph_store.query_patterns_for_idea_domain(state["idea_id"], domain_id)
        if not candidates:
            # 中文注释：兜底——即使 query 方法返回空，也确保有可选套路。
            nodes = self.graph_store.list_nodes("Pattern")
            candidates = [Pattern.from_dict(node) for node in nodes]

        candidate_ids = [pattern.id for pattern in candidates]

        # 中文注释：如果已有 candidate_pattern_scores（例如 novelty 失败后降权），就复用；否则重新计算 base score。
        scores = list(state.get("candidate_pattern_scores") or [])
        if len(scores) != len(candidate_ids):
            scores = [
                float(self.graph_store.query_pattern_effectiveness(pattern.id, domain_id))
                for pattern in candidates
            ]

        # 中文注释：根据 defects_agg/fix_tricks 对候选分数进行轻量引导（只在回退后更有意义）。
        fix_trick_ids = {trick.id for trick in state.get("fix_tricks", [])}
        if fix_trick_ids:
            boosted: List[float] = []
            for pattern, score in zip(candidates, scores):
                trick_nodes = self.graph_store.get_tricks_for_pattern(pattern.id)
                trick_ids = {
                    str(node.get("id") or node.get("node_id") or "")
                    for node in trick_nodes
                    if node.get("id") or node.get("node_id")
                }
                if trick_ids & fix_trick_ids:
                    boosted.append(score * 1.25 + 0.05)
                else:
                    boosted.append(score)
            scores = boosted

        selected = select_patterns_diverse(
            candidates=candidates,
            effectiveness_scores=scores,
            max_patterns=int(self.config.max_patterns),
            temperature=float(self.config.temperature),
            rng=self.rng,
        )

        tricks = _collect_tricks(self.graph_store, selected)
        logger.info(
            "PAPER_PLAN: patterns=%s tricks=%s",
            "、".join([p.pattern_name for p in selected]),
            "、".join([t.name for t in tricks]),
        )
        updates = {
            "candidate_pattern_ids": candidate_ids,
            "candidate_pattern_scores": scores,
            "selected_patterns": selected,
            "selected_tricks": tricks,
            # 中文注释：进入新一轮 plan 时清空 novelty_candidates 与 novelty_report，避免污染。
            "novelty_candidates": [],
            "novelty_report": None,
            "draft": None,
            "reviews": [],
            "paper_plan": None,
            "paper_sections": [],
            "paper_markdown": "",
            "section_summaries": {},
            "rewrite_targets": [],
        }
        if bool(self.config.skip_novelty_check):
            updates["novelty_report"] = NoveltyCheckResult(
                is_novel=True,
                max_similarity=0.0,
                top_similar=[],
                suggestion="已按配置跳过查重与新颖性判定。",
            )
        return updates

    def _post_plan_route(self, state: PaperWorkflowStateData) -> str:
        """
        功能：plan_from_kg 后的路由选择。
        参数：state。
        返回：下一节点名称（retrieve_similar 或 build_blueprint）。
        说明：当 skip_novelty_check=true 时直接进入 build_blueprint。
        """
        if bool(self.config.skip_novelty_check):
            logger.info("PAPER_ROUTE: plan_from_kg -> build_blueprint (skip_novelty_check)")
            return "build_blueprint"
        return "retrieve_similar"

    def _retrieve_similar_node(self, state: PaperWorkflowStateData) -> Dict:
        """
        功能：执行向量检索，获取相似历史论文配置（用于查重/新颖性判断）。
        参数：state。
        返回：更新 novelty_candidates。
        流程：encode_pattern_config → embedding → vector_store.search。
        说明：检索结果会在 novelty_check 节点做阈值判定与降权策略。
        """
        patterns = state.get("selected_patterns", [])
        text = encode_pattern_config(patterns)
        vector = self.embedding_model.embed(text)
        candidates = self.vector_store.search(vector, top_k=int(self.config.retriever_top_k))
        logger.info("PAPER_RETRIEVE_SIMILAR: count=%s", len(candidates))
        return {"novelty_candidates": candidates}

    def _novelty_check_node(self, state: PaperWorkflowStateData) -> Dict:
        """
        功能：根据相似检索结果进行新颖性判断；不新颖则回退 plan 并降权重采样。
        参数：state。
        返回：更新 novelty_report/plan_retry/next_step/candidate_pattern_scores 等。
        流程：
        1) 计算 max_similarity，与阈值比较；
        2) 若超阈值且未超过 max_plan_retry：提取相似项中的 pattern_ids → downweight → 回退 plan；
        3) 若超过最大重采样次数：给 warning，继续写稿（避免死循环）。
        说明：该节点负责新颖性判断与循环回退逻辑。
        """
        results = state.get("novelty_candidates", [])
        top_similar: List[SimilarItem] = []
        max_similarity = 0.0
        if results:
            max_similarity = float(results[0].get("score", 0.0))
            for item in results[:3]:
                top_similar.append(
                    SimilarItem(
                        item_id=str(item.get("item_id")),
                        score=float(item.get("score", 0.0)),
                        metadata=item.get("metadata", {}) or {},
                    )
                )

        # 中文注释：测试模式下可强制跳过查重，避免 mock 向量导致反复循环。
        if bool(self.config.force_novelty_pass):
            is_novel = True
            suggestion = "测试模式：已跳过查重，直接通过新颖性判断。"
            novelty_report = NoveltyCheckResult(
                is_novel=is_novel,
                max_similarity=max_similarity,
                top_similar=top_similar,
                suggestion=suggestion,
            )
            logger.info(
                "PAPER_NOVELTY: force_pass=True max_similarity=%.3f", max_similarity
            )
            return {"novelty_report": novelty_report, "next_step": "build_blueprint"}

        is_novel = max_similarity <= float(self.config.novelty_threshold)
        suggestion = (
            "相似度在可接受范围内，可进入生成阶段。"
            if is_novel
            else "相似度过高，建议调整套路组合以降低相似度。"
        )
        novelty_report = NoveltyCheckResult(
            is_novel=is_novel,
            max_similarity=max_similarity,
            top_similar=top_similar,
            suggestion=suggestion,
        )

        logger.info(
            "PAPER_NOVELTY: max_similarity=%.3f is_novel=%s",
            max_similarity,
            is_novel,
        )

        updates: Dict = {"novelty_report": novelty_report, "next_step": "build_blueprint"}
        if is_novel:
            return updates

        plan_retry = state.get("plan_retry", 0) + 1
        updates["plan_retry"] = plan_retry

        if plan_retry > int(self.config.max_plan_retry):
            warning = "超过最大新颖性重采样次数，仍存在较高相似度；将继续生成草稿并在评审中补救。"
            warnings = list(state.get("warnings", []))
            warnings.append(warning)
            updates["warnings"] = warnings
            updates["next_step"] = "build_blueprint"
            return updates

        # 中文注释：从相似项 metadata 中取出导致相似的 pattern_ids，并对候选分数降权。
        penalized_ids: List[str] = []
        for item in results[:3]:
            for pid in (item.get("metadata", {}) or {}).get("pattern_ids", []):
                penalized_ids.append(str(pid))
        if penalized_ids and state.get("candidate_pattern_ids"):
            new_scores = downweight_scores(
                state["candidate_pattern_ids"],
                state["candidate_pattern_scores"],
                penalized_ids,
                penalty=float(self.config.penalty),
            )
            updates["candidate_pattern_scores"] = new_scores
            logger.info("PAPER_NOVELTY_DOWNWEIGHT: %s", sorted(set(penalized_ids)))

        updates["next_step"] = "plan_from_kg"
        return updates

    def _novelty_route(self, state: PaperWorkflowStateData) -> str:
        """
        功能：根据新颖性节点的 next_step 决定下一步。
        参数：state。
        返回：下一节点名称。
        流程：读取 next_step → 返回 plan_from_kg/build_blueprint。
        说明：与当前工作流路由写法保持一致。
        """
        return "plan_from_kg" if state.get("next_step") == "plan_from_kg" else "build_blueprint"

    def _write_draft_node(self, state: PaperWorkflowStateData) -> Dict:
        """
        功能：兼容旧的 write_draft 节点实现，转调 build_blueprint。
        参数：state。
        返回：更新后的字段。
        流程：调用 _build_blueprint_node。
        说明：保留该方法避免外部调用出错。
        """
        return self._build_blueprint_node(state)

    def _build_blueprint_node(self, state: PaperWorkflowStateData) -> Dict:
        """
        功能：生成 Paper Blueprint（story_skeleton + paper_plan）。
        参数：state。
        返回：更新 paper_plan/draft/pattern_guides 等字段。
        流程：写稿器生成 story_skeleton → 模板生成 plan → 写入 draft。
        说明：该节点在新颖性通过后执行。
        """
        attempt = state.get("draft_attempts", 0) + 1
        core_idea = state["core_idea"]
        domain = state["domain"]
        patterns = state.get("selected_patterns", [])
        tricks = state.get("selected_tricks", [])
        novelty_report = state.get("novelty_report") or NoveltyCheckResult(
            is_novel=True, max_similarity=0.0, top_similar=[], suggestion="暂无相似配置，判定为新颖。"
        )

        skeleton, final_story = self.draft_writer.write_draft(
            idea=core_idea,
            domain=domain,
            patterns=patterns,
            tricks=tricks,
            attempt=attempt,
        )
        pattern_guides = self.draft_writer.build_pattern_guides(patterns)

        template = self.paper_template or load_paper_template(self.config.template_path)
        paper_plan = build_paper_plan(
            core_idea,
            domain,
            skeleton,
            patterns,
            tricks,
            template,
            paper_language=self.config.paper_language,
        )

        draft = StoryDraft(
            idea=core_idea,
            domain=domain,
            patterns=patterns,
            tricks=tricks,
            story_skeleton=skeleton,
            novelty=novelty_report,
            final_story=final_story,
            paper_plan=paper_plan,
            paper_sections=[],
            paper_markdown="",
            section_summaries={},
        )
        logger.info("PAPER_BLUEPRINT: sections=%s", len(paper_plan.sections))
        return {
            "draft": draft,
            "draft_attempts": attempt,
            "pattern_guides": pattern_guides,
            "paper_plan": paper_plan,
            "paper_sections": [],
            "paper_markdown": "",
            "section_summaries": {},
            "rewrite_targets": [],
        }

    def _write_abstract_node(self, state: PaperWorkflowStateData) -> Dict:
        """
        功能：写 Abstract 章节。
        参数：state。
        返回：更新 paper_sections/section_summaries/draft。
        流程：调用通用章节写作逻辑。
        说明：节点粒度便于局部重写。
        """
        return self._write_section(state, "abstract")

    def _write_introduction_node(self, state: PaperWorkflowStateData) -> Dict:
        """
        功能：写 Introduction 章节。
        参数：state。
        返回：更新 paper_sections/section_summaries/draft。
        流程：调用通用章节写作逻辑。
        说明：节点粒度便于局部重写。
        """
        return self._write_section(state, "introduction")

    def _write_related_work_node(self, state: PaperWorkflowStateData) -> Dict:
        """
        功能：写 Related Work 章节。
        参数：state。
        返回：更新 paper_sections/section_summaries/draft。
        流程：调用通用章节写作逻辑。
        说明：可通过配置关闭该章节生成。
        """
        return self._write_section(state, "related_work")

    def _write_method_node(self, state: PaperWorkflowStateData) -> Dict:
        """
        功能：写 Method 章节。
        参数：state。
        返回：更新 paper_sections/section_summaries/draft。
        流程：调用通用章节写作逻辑。
        说明：节点粒度便于局部重写。
        """
        return self._write_section(state, "method")

    def _write_experiments_node(self, state: PaperWorkflowStateData) -> Dict:
        """
        功能：写 Experiments 章节。
        参数：state。
        返回：更新 paper_sections/section_summaries/draft。
        流程：调用通用章节写作逻辑。
        说明：节点粒度便于局部重写。
        """
        return self._write_section(state, "experiments")

    def _write_conclusion_node(self, state: PaperWorkflowStateData) -> Dict:
        """
        功能：写 Conclusion 章节。
        参数：state。
        返回：更新 paper_sections/section_summaries/draft。
        流程：调用通用章节写作逻辑。
        说明：节点粒度便于局部重写。
        """
        return self._write_section(state, "conclusion")

    def _write_limitations_node(self, state: PaperWorkflowStateData) -> Dict:
        """
        功能：写 Limitations 章节。
        参数：state。
        返回：更新 paper_sections/section_summaries/draft。
        流程：调用通用章节写作逻辑。
        说明：节点粒度便于局部重写。
        """
        return self._write_section(state, "limitations")

    def _write_ethics_node(self, state: PaperWorkflowStateData) -> Dict:
        """
        功能：写 Ethics 章节。
        参数：state。
        返回：更新 paper_sections/section_summaries/draft。
        流程：调用通用章节写作逻辑。
        说明：节点粒度便于局部重写。
        """
        return self._write_section(state, "ethics")

    def _write_section(self, state: PaperWorkflowStateData, section_id: str) -> Dict:
        """
        功能：通用章节写作逻辑。
        参数：state、section_id。
        返回：更新 paper_sections/section_summaries/draft。
        流程：查找章节计划 → 判断是否需要重写 → 调用 SectionWriter。
        说明：rewrite_targets 非空时，仅重写指定章节。
        """
        plan = state.get("paper_plan")
        if plan is None:
            warning = f"缺少 paper_plan，跳过章节 {section_id}。"
            logger.warning(warning)
            warnings = list(state.get("warnings", []))
            warnings.append(warning)
            return {"warnings": warnings}

        # 中文注释：允许通过配置关闭 Related Work 章节。
        if section_id == "related_work" and not bool(self.config.enable_related_work):
            disabled_section = PaperSectionOutput(
                section_id="related_work",
                title="Related Work",
                content_markdown="# Related Work\n\n（已按配置关闭该章节）",
                used_tricks=[],
                covered_points=[],
                todo=["如需该章节请开启 enable_related_work"],
                word_count=0,
                summary="Related Work 章节已关闭。",
            )
            return self._commit_section_output(state, disabled_section)

        rewrite_targets = set(state.get("rewrite_targets") or [])
        existing_sections = state.get("paper_sections", [])
        existing_map = {section.section_id: section for section in existing_sections}

        if rewrite_targets and section_id not in rewrite_targets:
            return {}
        if not rewrite_targets and section_id in existing_map:
            return {}

        section_plan = _find_section_plan(plan, section_id)
        if section_plan is None:
            warning = f"未找到章节计划：{section_id}"
            logger.warning(warning)
            warnings = list(state.get("warnings", []))
            warnings.append(warning)
            return {"warnings": warnings}

        prior_summaries = {
            dep: state.get("section_summaries", {}).get(dep, "") for dep in section_plan.dependencies
        }
        writer = PaperSectionWriter(
            llm=self.section_llm,
            config=self.config,
            pattern_guides=state.get("pattern_guides", ""),
            latex_toolchain=self.latex_toolchain,
            latex_fixer=self.latex_fixer,
        )
        output = writer.write_section(
            plan=plan,
            section_plan=section_plan,
            prior_summaries=prior_summaries,
            attempt=state.get("draft_attempts", 1),
        )
        logger.info(
            "PAPER_SECTION: id=%s words=%s md_chars=%s",
            section_id,
            output.word_count,
            len(output.content_markdown or ""),
        )
        return self._commit_section_output(state, output)

    def _commit_section_output(
        self, state: PaperWorkflowStateData, output: PaperSectionOutput
    ) -> Dict:
        """
        功能：将章节输出写回 state，并同步 draft。
        参数：state、output。
        返回：更新字段 dict。
        流程：upsert section → 更新 summaries → 同步 draft。
        说明：确保 SSE 能看到逐章输出。
        """
        sections = _upsert_section_output(list(state.get("paper_sections", [])), output)
        summaries = dict(state.get("section_summaries", {}))
        summaries[output.section_id] = output.summary

        draft = state.get("draft")
        if draft is not None:
            draft = draft.model_copy(
                update={
                    "paper_sections": sections,
                    "section_summaries": summaries,
                    "paper_plan": state.get("paper_plan") or draft.paper_plan,
                }
            )

        return {
            "paper_sections": sections,
            "section_summaries": summaries,
            "draft": draft,
        }

    def _assemble_paper_node(self, state: PaperWorkflowStateData) -> Dict:
        """
        功能：拼接各章节输出，生成最终论文 Markdown。
        参数：state。
        返回：更新 paper_markdown/draft。
        流程：按 plan.sections 顺序拼接 → 写入 draft。
        说明：rewrite_targets 会在此阶段清空，避免重复改写。
        """
        logger.info("PAPER_ASSEMBLE_START")
        plan = state.get("paper_plan")
        if plan is None:
            warning = "缺少 paper_plan，无法组装全文。"
            warnings = list(state.get("warnings", []))
            warnings.append(warning)
            return {"warnings": warnings}

        section_map = {section.section_id: section for section in state.get("paper_sections", [])}
        parts_md: List[str] = []
        parts_tex: List[str] = []
        for section_plan in plan.sections:
            section = section_map.get(section_plan.section_id)
            if section:
                if section.content_markdown:
                    parts_md.append(section.content_markdown)
                if section.content_latex:
                    parts_tex.append(section.content_latex)
        assembled_md = "\n\n".join([p for p in parts_md if p and p.strip()])
        assembled_tex = "\n\n".join([p for p in parts_tex if p and p.strip()])

        paper_markdown = assembled_md
        paper_latex = assembled_tex
        draft = state.get("draft")
        if draft is not None:
            draft = draft.model_copy(
                update={
                    "paper_markdown": paper_markdown,
                    "paper_latex": paper_latex,
                    "paper_sections": list(section_map.values()),
                }
            )

        updates = {
            "paper_markdown": paper_markdown,
            "paper_latex": paper_latex,
            "draft": draft,
            "rewrite_targets": [],
        }
        # 中文注释：assemble 阶段只负责拼接内容；
        # 若非 LaTeX 且 format_only_mode=true，可直接结束；
        # 若为 LaTeX 模式，需进入导出/编译闸门。
        if bool(self.config.format_only_mode) and str(self.config.output_format) != "latex":
            updates["status"] = "PASS"
            updates["next_step"] = "complete"
        logger.info(
            "PAPER_ASSEMBLE_OK: sections=%s chars=%s",
            len(parts_md),
            len(assembled_md or ""),
        )
        return updates

    def _post_assemble_route(self, state: PaperWorkflowStateData) -> str:
        """
        功能：assemble 后的路由选择。
        参数：state。
        返回：下一节点名。
        说明：
        - LaTeX 模式：进入 export_overleaf（随后 compile_overleaf）；
        - 非 LaTeX：format_only_mode 可直接结束，否则进入 review。
        """
        if str(self.config.output_format) == "latex":
            if self.config.latex_generation_mode == "final_only":
                if bool(self.config.citations_enable):
                    logger.info("PAPER_ROUTE: assemble -> extract_cite_candidates (latex final_only)")
                    return "extract_cite_candidates"
                logger.info("PAPER_ROUTE: assemble -> render_latex (latex final_only)")
                return "render_latex"
            logger.info("PAPER_ROUTE: assemble -> export_overleaf (latex)")
            return "export_overleaf"
        if bool(self.config.format_only_mode):
            logger.info("PAPER_ROUTE: assemble -> complete (format_only_mode, markdown)")
            return "complete"
        logger.info("PAPER_ROUTE: assemble -> review")
        return "review"

    def _extract_cite_candidates_node(self, state: PaperWorkflowStateData) -> Dict:
        """
        功能：从 Markdown 中提取引用候选点并补齐 ID。
        参数：state。
        返回：更新后的 sections、paper_markdown、citation_candidates。
        """
        logger.info("PAPER_CITATION_EXTRACT_START")
        plan = state.get("paper_plan")
        if plan is None:
            warning = "缺少 paper_plan，无法提取引用候选点。"
            warnings = list(state.get("warnings", []))
            warnings.append(warning)
            return {"warnings": warnings, "next_step": "render_latex"}

        sections = list(state.get("paper_sections", []))
        section_pairs = [(s.section_id, s.content_markdown or "") for s in sections]
        updated_pairs, candidates = extract_cite_candidates_from_sections(section_pairs)

        updated_map = {sid: md for sid, md in updated_pairs}
        updated_sections: List[PaperSectionOutput] = []
        for section in sections:
            updated_sections.append(
                section.model_copy(update={"content_markdown": updated_map.get(section.section_id, "")})
            )

        parts_md: List[str] = []
        for section_plan in plan.sections:
            section = next(
                (s for s in updated_sections if s.section_id == section_plan.section_id), None
            )
            if section and section.content_markdown:
                parts_md.append(section.content_markdown)
        paper_markdown = "\n\n".join([p for p in parts_md if p.strip()])

        draft = state.get("draft")
        if draft is not None:
            draft = draft.model_copy(
                update={"paper_sections": updated_sections, "paper_markdown": paper_markdown}
            )

        next_step = "citation_need_review" if candidates else "render_latex"
        logger.info("PAPER_CITATION_EXTRACT_OK: candidates=%s", len(candidates))
        return {
            "paper_sections": updated_sections,
            "paper_markdown": paper_markdown,
            "draft": draft,
            "citation_candidates": [c.model_dump() for c in candidates],
            "citation_needs": [],
            "citation_results": [],
            "unresolved_citations": [],
            "next_step": next_step,
        }

    def _post_extract_cite_candidates_route(self, state: PaperWorkflowStateData) -> str:
        next_step = state.get("next_step") or ""
        if next_step:
            return next_step
        if not state.get("citation_candidates"):
            return "render_latex"
        return "citation_need_review"

    def _citation_need_review_node(self, state: PaperWorkflowStateData) -> Dict:
        """
        功能：判断引用必要性（HARD/SOFT/NONE）。
        参数：state。
        返回：citation_needs。
        """
        candidates = [CitationCandidate.model_validate(c) for c in state.get("citation_candidates", [])]
        needs = self.citation_reviewer.review(candidates, year_range=str(self.config.citations_year_range))
        logger.info("PAPER_CITATION_NEED_OK: needs=%s", len(needs))
        return {"citation_needs": [n.model_dump() for n in needs], "next_step": ""}

    def _citation_retrieval_node(self, state: PaperWorkflowStateData) -> Dict:
        """
        功能：检索真实引用并生成 refs.bib。
        参数：state。
        返回：citation_results/unresolved_citations/refs_bib_text/citations_report。
        """
        needs = [CitationNeed.model_validate(n) for n in state.get("citation_needs", [])]
        if not needs:
            return {"citation_results": [], "unresolved_citations": [], "refs_bib_text": "", "next_step": "render_latex"}

        candidates = [CitationCandidate.model_validate(c) for c in state.get("citation_candidates", [])]
        results: List[RetrievedCitation] = []
        if self.citation_retrieval_agent is not None:
            results = self.citation_retrieval_agent.retrieve(needs, candidates)
        else:
            provider = self.citation_provider or get_citation_provider(
                str(self.config.citations_provider),
                require_verifiable=bool(self.config.citations_require_verifiable),
            )
            for need in needs:
                if need.need_type == "NONE":
                    continue
                results.append(provider.search_and_resolve(need))

        found_ids = {r.candidate_id for r in results if r.status == "FOUND"}
        unresolved = [
            need.candidate_id
            for need in needs
            if need.need_type != "NONE" and need.candidate_id not in found_ids
        ]

        bib_entries: List[str] = []
        seen = set()
        for result in results:
            if result.status == "FOUND" and result.bibkey and result.bibkey not in seen:
                seen.add(result.bibkey)
                if result.bibtex:
                    bib_entries.append(result.bibtex)
        refs_bib_text = "\n".join(bib_entries)

        report = {
            "retrieved": [r.model_dump() for r in results],
            "unresolved": list(unresolved),
            "round": state.get("citation_rounds", 0),
        }

        next_step = "claim_resolution" if unresolved else "render_latex"
        logger.info(
            "PAPER_CITATION_RETRIEVE_DONE: found=%s unresolved=%s",
            sum(1 for r in results if r.status == "FOUND"),
            len(unresolved),
        )
        return {
            "citation_results": [r.model_dump() for r in results],
            "unresolved_citations": unresolved,
            "refs_bib_text": refs_bib_text,
            "citations_report": report,
            "next_step": next_step,
        }

    def _post_citation_retrieval_route(self, state: PaperWorkflowStateData) -> str:
        return str(state.get("next_step") or "render_latex")

    def _claim_resolution_node(self, state: PaperWorkflowStateData) -> Dict:
        """
        功能：对无法检索的引用做降级改写。
        参数：state。
        返回：更新后的 sections/markdown 与 next_step。
        """
        unresolved = list(state.get("unresolved_citations", []))
        if not unresolved:
            return {"next_step": "render_latex"}

        rounds = int(state.get("citation_rounds", 0)) + 1
        if rounds > int(self.config.citations_max_rounds):
            logger.warning("PAPER_CITATION_RESOLVE_FAIL: rounds_exceeded=%s", rounds)
            return {"status": "FAILED", "next_step": "failed", "citation_rounds": rounds}

        candidates = [CitationCandidate.model_validate(c) for c in state.get("citation_candidates", [])]
        sections = list(state.get("paper_sections", []))
        section_pairs = [(s.section_id, s.content_markdown or "") for s in sections]
        if self.claim_resolution_agent is not None:
            updated_pairs = self.claim_resolution_agent.resolve(section_pairs, candidates, unresolved)
        else:
            updated_pairs = self.claim_resolver(section_pairs, candidates, unresolved)
        updated_map = {sid: md for sid, md in updated_pairs}

        updated_sections: List[PaperSectionOutput] = []
        for section in sections:
            updated_sections.append(
                section.model_copy(update={"content_markdown": updated_map.get(section.section_id, "")})
            )

        plan = state.get("paper_plan")
        parts_md: List[str] = []
        if plan is not None:
            for section_plan in plan.sections:
                section = next(
                    (s for s in updated_sections if s.section_id == section_plan.section_id), None
                )
                if section and section.content_markdown:
                    parts_md.append(section.content_markdown)
        paper_markdown = "\n\n".join([p for p in parts_md if p.strip()])

        draft = state.get("draft")
        if draft is not None:
            draft = draft.model_copy(
                update={"paper_sections": updated_sections, "paper_markdown": paper_markdown}
            )

        logger.info("PAPER_CITATION_RESOLVE_OK: rounds=%s", rounds)
        return {
            "paper_sections": updated_sections,
            "paper_markdown": paper_markdown,
            "draft": draft,
            "citation_rounds": rounds,
            "unresolved_citations": [],
            "next_step": "extract_cite_candidates",
        }

    def _post_claim_resolution_route(self, state: PaperWorkflowStateData) -> str:
        return str(state.get("next_step") or "render_latex")

    def _render_latex_node(self, state: PaperWorkflowStateData) -> Dict:
        """
        功能：将 Markdown 统一渲染为 LaTeX（最终排版工）。
        参数：state。
        返回：更新后的 sections/paper_latex。
        """
        sections = list(state.get("paper_sections", []))
        results = [RetrievedCitation.model_validate(r) for r in state.get("citation_results", [])]
        citation_map = {r.candidate_id: r.bibkey for r in results if r.status == "FOUND" and r.bibkey}

        if self._should_fail_fast_on_fallback(sections):
            fallback_ids = [
                s.section_id for s in sections if self._is_fallback_section(s)
            ]
            raise RuntimeError(
                "Section writing failed quality gate; aborting before render_latex. "
                f"fallback_sections={fallback_ids}"
            )

        compile_report = state.get("latex_compile_report") or {}
        compile_ok = compile_report.get("ok") is True
        fix_mode = bool(state.get("latex_format_retry", 0)) > 0 and not compile_ok
        compile_log = "\n".join(
            [
                str(compile_report.get("stderr", "")),
                str(compile_report.get("stdout", "")),
                str(compile_report.get("log_excerpt", "")),
            ]
        )
        logger.info(
            "PAPER_RENDER_LATEX_START: sections=%s fix_mode=%s compile_log_chars=%s",
            len(sections),
            fix_mode,
            len(compile_log),
        )

        if self.latex_renderer_agent is not None:
            if fix_mode:
                targets = self._extract_latex_fix_targets(compile_report)
                logger.info(
                    "PAPER_RENDER_LATEX_FIX: targets=%s",
                    ",".join(sorted(targets or [])) if targets else "ALL",
                )
                updated_sections, paper_latex = self.latex_renderer_agent.fix_sections(
                    sections,
                    compile_log=compile_log,
                    citation_map=citation_map,
                    targets=targets,
                )
            else:
                updated_sections, paper_latex = self.latex_renderer_agent.render_sections(
                    sections, citation_map
                )
        else:
            updated_sections, paper_latex = render_sections_to_latex(sections, citation_map)

        draft = state.get("draft")
        if draft is not None:
            draft = draft.model_copy(
                update={"paper_sections": updated_sections, "paper_latex": paper_latex}
            )

        logger.info("PAPER_RENDER_LATEX_OK: sections=%s", len(updated_sections))
        return {"paper_sections": updated_sections, "paper_latex": paper_latex, "draft": draft}

    def _export_overleaf_node(self, state: PaperWorkflowStateData) -> Dict:
        """
        功能：导出 Overleaf 工程目录与 zip（LaTeX 模式）。
        参数：state。
        返回：更新 overleaf_project_dir/overleaf_zip_path/draft。
        流程：调用 build_overleaf_project 写入 main.tex/sections/*.tex/refs.bib。
        说明：main.tex 由系统固定生成，LLM 只负责章节正文。
        """
        logger.info("PAPER_EXPORT_OVERLEAF_NODE_ENTER")
        plan = state.get("paper_plan")
        sections = state.get("paper_sections", [])
        if plan is None:
            warning = "缺少 paper_plan，无法导出 Overleaf 工程。"
            warnings = list(state.get("warnings", []))
            warnings.append(warning)
            logger.warning("PAPER_EXPORT_OVERLEAF_FAIL: %s", warning)
            return {"warnings": warnings}

        run_id = f"paper_{int(self.rng.random() * 10**10)}"
        output_dir = str(self.config.latex_output_dir or "./outputs/overleaf")
        logger.info("PAPER_EXPORT_OVERLEAF_START: run_id=%s dir=%s", run_id, output_dir)
        result = build_overleaf_project(
            run_id=run_id,
            output_dir=output_dir,
            plan=plan,
            sections=sections,
            refs_bib_text=state.get("refs_bib_text") or None,
            citations_report=state.get("citations_report") or None,
            latex_template_preset=self.config.latex_template_preset,
            latex_titlepage_style=self.config.latex_titlepage_style,
            latex_title=self.config.latex_title,
            latex_author=self.config.latex_author,
            latex_hflink=self.config.latex_hflink,
            latex_mslink=self.config.latex_mslink,
            latex_ghlink=self.config.latex_ghlink,
            latex_template_assets_dir=self.config.latex_template_assets_dir,
        )
        project_dir = result.get("project_dir", "")
        zip_path = result.get("zip_path", "")

        draft = state.get("draft")
        if draft is not None:
            draft = draft.model_copy(
                update={
                    "overleaf_project_dir": project_dir,
                    "overleaf_zip_path": zip_path,
                }
            )

        logger.info("PAPER_EXPORT_OVERLEAF_OK: project_dir=%s zip=%s", project_dir, zip_path)
        return {
            "overleaf_project_dir": project_dir,
            "overleaf_zip_path": zip_path,
            "draft": draft,
        }

    def _compile_overleaf_node(self, state: PaperWorkflowStateData) -> Dict:
        """
        功能：对 Overleaf 工程做全量编译闸门（LaTeX 模式）。
        参数：state。
        返回：更新 latex_compile_report/draft。
        流程：根据配置调用 latex_toolchain.compile；可选跳过。
        说明：format_only_mode 也必须执行格式闸门（除非显式关闭 require_tools）。
        """
        logger.info("PAPER_COMPILE_LATEX_NODE_ENTER")
        project_dir = str(state.get("overleaf_project_dir") or "")
        if not project_dir:
            warning = "缺少 overleaf_project_dir，跳过编译闸门。"
            warnings = list(state.get("warnings", []))
            warnings.append(warning)
            logger.warning("PAPER_COMPILE_LATEX_SKIP: %s", warning)
            return {"warnings": warnings, "latex_compile_report": {"ok": False, "skipped": True, "reason": warning}}

        if self.latex_toolchain is None:
            if bool(self.config.latex_require_tools):
                raise RuntimeError("未提供 LaTeX 工具链，但 latex_require_tools=true。")
            report = {"ok": True, "skipped": True, "reason": "latex_require_tools=false，已跳过编译验证。"}
            draft = state.get("draft")
            if draft is not None:
                draft = draft.model_copy(update={"latex_compile_report": report})
            logger.info("PAPER_COMPILE_LATEX_SKIP: %s", report["reason"])
            updates: Dict[str, object] = {"latex_compile_report": report, "draft": draft}
            if bool(self.config.format_only_mode):
                updates["status"] = "PASS"
            updates["next_step"] = ""
            return updates

        if not bool(self.config.latex_final_compile):
            report = {"ok": True, "skipped": True, "reason": "latex_final_compile=false，已跳过全量编译。"}
            draft = state.get("draft")
            if draft is not None:
                draft = draft.model_copy(update={"latex_compile_report": report})
            logger.info("PAPER_COMPILE_LATEX_SKIP: %s", report["reason"])
            updates = {"latex_compile_report": report, "draft": draft}
            if bool(self.config.format_only_mode):
                updates["status"] = "PASS"
            updates["next_step"] = ""
            return updates

        logger.info("PAPER_COMPILE_LATEX_START: dir=%s engine=%s", project_dir, self.config.latex_engine)
        result = self.latex_toolchain.compile_project(
            project_dir=Path(project_dir),
            engine=str(self.config.latex_engine),
            timeout_sec=int(self.config.latex_compile_timeout_sec),
        )
        report = result.to_dict()

        draft = state.get("draft")
        if draft is not None:
            draft = draft.model_copy(update={"latex_compile_report": report})

        updates = {"latex_compile_report": report, "draft": draft}
        if result.ok:
            logger.info("PAPER_COMPILE_LATEX_OK")
            if bool(self.config.format_only_mode):
                updates["status"] = "PASS"
            updates["next_step"] = ""
            return updates

        logger.warning("PAPER_COMPILE_LATEX_FAIL: %s", report.get("log_excerpt", "")[:200])
        retry = int(state.get("latex_format_retry", 0)) + 1
        updates["latex_format_retry"] = retry
        if retry <= int(self.config.latex_format_max_retries):
            warnings = list(state.get("warnings", []))
            warnings.append(f"全量编译失败，触发 LaTeX 仅格式修复重试：{retry}")
            updates["warnings"] = warnings
            updates["next_step"] = "render_latex"
            return updates

        updates["status"] = "FAILED"
        updates["next_step"] = "failed"
        return updates

    def _post_compile_route(self, state: PaperWorkflowStateData) -> str:
        """
        功能：compile_overleaf 后的路由选择。
        参数：state。
        返回：下一节点名。
        说明：format_only_mode=true 时直接结束，否则进入 review。
        """
        next_step = state.get("next_step") or ""
        if next_step:
            return str(next_step)
        if bool(self.config.format_only_mode):
            logger.info("PAPER_ROUTE: compile_overleaf -> complete (format_only_mode)")
            return "complete"
        logger.info("PAPER_ROUTE: compile_overleaf -> review")
        return "review"

    def _extract_latex_fix_targets(self, report: Dict[str, object]) -> Optional[Set[str]]:
        targets: Set[str] = set()
        errors = report.get("errors") if isinstance(report, dict) else None
        if isinstance(errors, list):
            for err in errors:
                if not isinstance(err, dict):
                    continue
                file_path = str(err.get("file", ""))
                match = re.search(r"sections/([A-Za-z0-9_-]+)\\.tex", file_path)
                if match:
                    targets.add(match.group(1))

        log_text = "\n".join(
            [
                str(report.get("stderr", "")),
                str(report.get("stdout", "")),
                str(report.get("log_excerpt", "")),
            ]
        )
        for match in re.finditer(r"sections/([A-Za-z0-9_-]+)\\.tex", log_text):
            targets.add(match.group(1))

        return targets or None

    def _should_fail_fast_on_fallback(self, sections: List[PaperSectionOutput]) -> bool:
        if not bool(self.config.quality_fail_fast):
            return False
        if not str(self.config.paper_language or "").lower().startswith("en"):
            return False
        return any(self._is_fallback_section(section) for section in sections)

    def _is_fallback_section(self, section: PaperSectionOutput) -> bool:
        md = section.content_markdown or ""
        if "> 兜底输出" in md or "> Fallback output" in md:
            return True
        todo_text = " ".join(section.todo or [])
        if "待补充正文" in todo_text or "Content pending" in todo_text:
            return True
        return False

    def _review_node(self, state: PaperWorkflowStateData) -> Dict:
        """
        功能：多 Reviewer 评审节点。
        参数：state。
        返回：更新 reviews/review_iterations。
        流程：reviewer_group.review → aggregate_reviews 打印 avg_score。
        说明：评审输出将驱动下一步回退决策。
        """
        draft = state.get("draft")
        if draft is None:
            raise ValueError("Draft must exist before review")

        reviews = self.reviewer_group.review(draft)
        avg_score, _pass_check, _issue = aggregate_reviews(
            reviews, pass_threshold=float(self.config.pass_threshold), require_individual_threshold=bool(self.config.require_individual_threshold)
        )
        iterations = state.get("review_iterations", 0) + 1
        logger.info("PAPER_REVIEW: avg_score=%.2f", avg_score)
        return {"reviews": reviews, "review_iterations": iterations}

    def _decide_node(self, state: PaperWorkflowStateData) -> Dict:
        """
        功能：根据评审结果做通过/回退/失败决策。
        参数：state。
        返回：更新 status/next_step 与各类 retry 计数，以及 defects_agg/fix_tricks。
        流程：
        - PASS：结束；
        - CLARITY_ISSUE：回退重写 Abstract/Introduction；
        - SOUNDNESS_RISK：回退重写 Method/Experiments；
        - SIGNIFICANCE_LOW：回退重写 Experiments/Conclusion；
        - NOVELTY_LOW：回退 plan_from_kg；
        - 超过 max_iterations 则失败。
        说明：缺陷修复动作通过 KG 的 Defect->fixed_by 边查得（fix_tricks）。
        """
        reviews = state.get("reviews", [])
        avg_score, pass_check, main_issue = aggregate_reviews(
            reviews,
            pass_threshold=float(self.config.pass_threshold),
            require_individual_threshold=bool(self.config.require_individual_threshold),
        )

        updates: Dict = {"next_step": ""}

        if pass_check:
            updates["status"] = "PASS"
            updates["next_step"] = "complete"
            return updates

        # 中文注释：达到最大迭代次数直接失败，避免无限循环。
        if state.get("review_iterations", 0) >= int(self.config.max_iterations):
            updates["status"] = "FAILED"
            updates["next_step"] = "failed"
            return updates

        defects = aggregate_defects(reviews)
        fix_tricks = pick_fix_tricks_for_defects(defects, self.graph_store, top_k_per_defect=2)
        updates["defects_agg"] = defects
        updates["fix_tricks"] = fix_tricks

        # 中文注释：若 avg_score 低但 main_issue 仍为 PASS，默认当作“表达问题”处理，回退改稿。
        if main_issue == "PASS" and avg_score < float(self.config.pass_threshold):
            main_issue = "CLARITY_ISSUE"

        if main_issue in {"CLARITY_ISSUE", "SOUNDNESS_RISK", "SIGNIFICANCE_LOW"}:
            rewrite_retry = state.get("rewrite_retry", 0) + 1
            updates["rewrite_retry"] = rewrite_retry
            if rewrite_retry > int(self.config.max_rewrite_retry):
                updates["status"] = "FAILED"
                updates["next_step"] = "failed"
                return updates

            if main_issue == "CLARITY_ISSUE":
                updates["rewrite_targets"] = ["abstract", "introduction"]
                updates["next_step"] = "write_abstract"
            elif main_issue == "SOUNDNESS_RISK":
                updates["rewrite_targets"] = ["method", "experiments"]
                updates["next_step"] = "write_method"
            else:
                updates["rewrite_targets"] = ["experiments", "conclusion"]
                updates["next_step"] = "write_experiments"
            return updates

        # 中文注释：NOVELTY 仍回退到 plan。
        global_retry = state.get("global_retry", 0) + 1
        updates["global_retry"] = global_retry
        if global_retry > int(self.config.max_global_retry):
            updates["status"] = "FAILED"
            updates["next_step"] = "failed"
            return updates

        # 中文注释：回退 plan 时，清空改稿计数，并重置 plan_retry（重新开始 novelty 降权循环）。
        updates["rewrite_retry"] = 0
        updates["plan_retry"] = 0
        updates["candidate_pattern_scores"] = []
        updates["next_step"] = "plan_from_kg"
        return updates

    def _decide_route(self, state: PaperWorkflowStateData) -> str:
        """
        功能：把 decide 节点给出的 next_step 映射到 LangGraph 边。
        参数：state。
        返回：节点名或 END。
        流程：读取 next_step → 返回对应 key。
        说明：complete/failed 会终止工作流。
        """
        step = state.get("next_step", "")
        if step in {
            "complete",
            "failed",
            "write_abstract",
            "write_method",
            "write_experiments",
            "plan_from_kg",
        }:
            return step
        return "failed"


def _find_section_plan(plan: PaperPlan, section_id: str) -> Optional[PaperSectionPlan]:
    """
    功能：从 PaperPlan 中查找指定 section_id 的计划。
    参数：plan、section_id。
    返回：PaperSectionPlan 或 None。
    流程：遍历 sections → 匹配 section_id。
    说明：用于写作节点获取章节配置。
    """
    for section in plan.sections:
        if section.section_id == section_id:
            return section
    return None


def _upsert_section_output(
    sections: List[PaperSectionOutput], output: PaperSectionOutput
) -> List[PaperSectionOutput]:
    """
    功能：将章节输出写入列表（存在则替换，不存在则追加）。
    参数：sections、output。
    返回：更新后的章节列表。
    流程：遍历匹配 → 替换或追加。
    说明：确保同一章节不会重复堆叠。
    """
    updated: List[PaperSectionOutput] = []
    replaced = False
    for section in sections:
        if section.section_id == output.section_id:
            updated.append(output)
            replaced = True
        else:
            updated.append(section)
    if not replaced:
        updated.append(output)
    return updated


def _resolve_domain_id(graph_store: IGraphStore, domain_hint: str) -> tuple[str, str]:
    """
    功能：根据 domain_hint 解析 Domain 节点 ID。
    参数：graph_store、domain_hint。
    返回：(domain_id, domain_name)。
    流程：根据中文关键词匹配 Domain.name → 回退第一个 Domain。
    说明：用于适配 knowledge_graph 的中文领域命名。
    """
    hint = str(domain_hint or "").strip().upper()
    domains = [Domain.from_dict(node) for node in graph_store.list_nodes("Domain")]
    if not domains:
        return "", domain_hint or ""

    def _match_domain(keyword_list: list[str]) -> Optional[Domain]:
        for domain in domains:
            name = str(domain.name or "")
            if any(keyword in name for keyword in keyword_list):
                return domain
        return None

    if hint == "CV":
        matched = _match_domain(["计算机视觉", "视觉", "图像", "CV"])
        if matched:
            return matched.id, matched.name
    # 默认 NLP 或未知时尝试 NLP 匹配
    matched = _match_domain(["自然语言", "语言", "NLP"])
    if matched:
        return matched.id, matched.name

    # 中文注释：兜底使用第一个 Domain。
    return domains[0].id, domains[0].name


def _resolve_idea_id(graph_store: IGraphStore, core_idea: str) -> Optional[str]:
    """
    功能：在图谱中根据文本匹配 Idea 节点 ID（弱匹配）。
    参数：graph_store、core_idea。
    返回：匹配到的 idea_id 或 None。
    流程：遍历 Idea 节点 → core_idea 子串匹配。
    说明：匹配失败时返回 None，上层将触发 domain fallback。
    """
    text = str(core_idea or "").strip()
    if not text:
        return None
    for node in graph_store.list_nodes("Idea"):
        desc = str(node.get("core_idea", "")).strip()
        if not desc:
            continue
        if text in desc or desc in text:
            return str(node.get("id") or node.get("node_id") or "").strip() or None
    return None


def _collect_tricks(graph_store: IGraphStore, patterns: List[Pattern]) -> List[Trick]:
    """
    功能：根据选中的 patterns 收集 Tricks（去重）。
    参数：graph_store、patterns。
    返回：Trick 列表。
    流程：get_tricks_for_pattern → Trick.from_dict → 按 trick.id 去重。
    说明：保持顺序稳定，便于日志与复现。
    """
    seen: Set[str] = set()
    tricks: List[Trick] = []
    for pattern in patterns:
        for node in graph_store.get_tricks_for_pattern(pattern.id):
            trick = Trick.from_dict(node)
            if trick.id in seen:
                continue
            seen.add(trick.id)
            tricks.append(trick)
    return tricks
