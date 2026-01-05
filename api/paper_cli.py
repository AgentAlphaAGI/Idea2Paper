# 中文注释: 本文件提供 Paper KG RAG 的命令行入口。
from __future__ import annotations

import json
import logging
import os
import random
import secrets
from pathlib import Path
from typing import Dict, List, Optional

import typer

from core.config import load_config
from core.embeddings import create_embedding_model
from core.llm import create_chat_model, load_env
from core.logging import add_step_summary_file_handler, setup_logging
from paper_kg.docstore_json import JsonDocStore
from paper_kg.graph_networkx import NetworkXGraphStore
from paper_kg.input_parser import PaperInputParser
from paper_kg.models import (
    Defect,
    Domain,
    Idea,
    Paper,
    Pattern,
    Review,
    Skeleton,
    StoryDraft,
    Trick,
)
from paper_kg.novelty import encode_pattern_config
from paper_kg.orchestrator import PaperDraftWriter, StoryOrchestrator
from paper_kg.paper_template import load_paper_template
from paper_kg.embedding_mock import MockEmbeddingModel
from paper_kg.citations.claim_resolution_agent import ClaimResolutionAgent
from paper_kg.citations.retrieval_agent import CitationRetrievalAgent
from paper_kg.citations.semantic_scholar_tool import MockSemanticScholarTool
from paper_kg.pandoc_latex_renderer_agent import PandocLatexRendererAgent
from paper_kg.pandoc_renderer import SubprocessPandocRunner
from paper_kg.latex_toolchain import LocalLatexToolchain
from paper_kg.vectorstore_memory import InMemoryVectorStore
from workflow.paper_controller import PaperFeedbackLoopController

app = typer.Typer(add_completion=False)
logger = logging.getLogger(__name__)


def _ensure_path(path: str) -> Path:
    """
    功能：确保输出路径的父目录存在。
    参数：path。
    返回：Path 对象。
    流程：创建父目录 → 返回 Path。
    说明：用于保存快照文件。
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def _default_snapshot_path() -> str:
    """
    功能：解析默认快照路径。
    参数：无。
    返回：快照路径字符串。
    流程：优先知识图谱 → 回退 mock 快照。
    说明：若 knowledge_graph.json 存在则优先使用。
    """
    if Path("knowledge_graph.json").exists():
        return "knowledge_graph.json"
    return "./paper_kg_snapshot.json"


def _build_mock_data() -> Dict[str, List]:
    """
    功能：构造小规模 mock 图谱数据。
    参数：无。
    返回：包含节点列表的数据结构。
    流程：创建节点 → 返回。
    说明：用于单元测试与离线演示。
    """
    domains = [
        Domain(id="domain_nlp", name="自然语言处理", paper_count=3),
        Domain(id="domain_cv", name="计算机视觉", paper_count=2),
    ]

    ideas = [
        Idea(id="idea_diffusion", core_idea="用 Diffusion 做推理"),
        Idea(id="idea_multimodal", core_idea="多模态对齐的新框架"),
        Idea(id="idea_rl", core_idea="强化学习驱动的规划"),
    ]

    tricks = [
        Trick(id="trick_ablation", name="消融实验", trick_type="experiment-level", purpose="可信度", description="多角度消融验证模块贡献。"),
        Trick(id="trick_toy", name="玩具示例", trick_type="writing-level", purpose="可解释性", description="用极简示例解释核心机制。"),
        Trick(id="trick_scaling", name="规模扩展", trick_type="method-level", purpose="性能提升", description="展示规模扩展后的性能趋势。"),
        Trick(id="trick_sota", name="SOTA 对比", trick_type="experiment-level", purpose="说服力", description="对比当前最佳基线。"),
        Trick(id="trick_theory", name="理论收敛性证明", trick_type="method-level", purpose="严谨性", description="给出收敛性或复杂度论证。"),
        Trick(id="trick_case", name="案例研究", trick_type="writing-level", purpose="清晰度", description="通过案例增强可读性。"),
    ]

    patterns = [
        Pattern(
            id="pat_gap_analysis",
            pattern_name="研究缺口放大",
            pattern_summary="强调现有方法的核心不足并放大差距",
            writing_guide="先指出现有方法的痛点，再描述具体缺口，最后给出动机承接。",
            cluster_size=12,
            coherence_score=0.62,
            skeleton_count=3,
            trick_count=2,
        ),
        Pattern(
            id="pat_method_story",
            pattern_name="方法叙述递进",
            pattern_summary="按动机-设计-细节递进叙述",
            writing_guide="先动机，再设计，再细节，强调关键模块与优势。",
            cluster_size=10,
            coherence_score=0.58,
            skeleton_count=3,
            trick_count=2,
        ),
        Pattern(
            id="pat_experiment_rigorous",
            pattern_name="实验严谨组合",
            pattern_summary="对比+消融+泛化，形成闭环",
            writing_guide="先基线对比，再做消融，最后补充泛化测试。",
            cluster_size=8,
            coherence_score=0.65,
            skeleton_count=2,
            trick_count=2,
        ),
    ]

    papers = [
        Paper(id="paper_001", title="Diffusion Reasoning for NLP", conference="ICLR"),
        Paper(id="paper_002", title="Alignment Framework for Multimodal Agents", conference="NeurIPS"),
        Paper(id="paper_003", title="Cross-Modal Vision Alignment", conference="CVPR"),
    ]

    reviews = [
        Review(
            id="review_001",
            paper_id="paper_001",
            reviewer="reviewer_a",
            overall_score="7.5",
            confidence="4",
            strengths="新颖性强，实验充分。",
            weaknesses="缺少理论分析。",
        ),
        Review(
            id="review_002",
            paper_id="paper_002",
            reviewer="reviewer_b",
            overall_score="7.0",
            confidence="3",
            strengths="方法较新颖。",
            weaknesses="实验不足。",
        ),
        Review(
            id="review_003",
            paper_id="paper_003",
            reviewer="reviewer_c",
            overall_score="6.8",
            confidence="3",
            strengths="对比充分。",
            weaknesses="消融不足。",
        ),
    ]

    defect_types = [
        "novelty_low",
        "experiments_weak",
        "theory_missing",
        "clarity_low",
        "significance_low",
    ]
    defects = [
        Defect(
            defect_id=f"defect_{dtype}",
            type=dtype,
            description=f"{dtype} 的缺陷说明。",
            frequency=0.5,
        )
        for dtype in defect_types
    ]

    return {
        "domains": domains,
        "ideas": ideas,
        "tricks": tricks,
        "patterns": patterns,
        "papers": papers,
        "reviews": reviews,
        "defects": defects,
        "skeletons": [
            Skeleton(id="sk_problem", skeleton_type="problem", content="问题定义模板"),
            Skeleton(id="sk_gap", skeleton_type="gap", content="研究缺口模板"),
            Skeleton(id="sk_method", skeleton_type="method", content="方法叙述模板"),
            Skeleton(id="sk_exp", skeleton_type="experiments", content="实验策略模板"),
        ],
    }


def _build_mock_data_scaled(scale: int, seed: Optional[int]) -> Dict[str, List]:
    """
    功能：构造可扩展规模的 mock 图谱数据。
    参数：scale、seed。
    返回：包含节点列表的数据结构。
    流程：随机生成节点 → 返回。
    说明：用于压测与功能演示。
    """
    rng = random.Random(seed)

    domain_names = [
        "自然语言处理",
        "计算机视觉",
        "强化学习",
        "多模态",
        "语音",
        "推荐系统",
        "图学习",
        "安全",
        "系统",
        "数据库",
    ]
    domains = [
        Domain(id=f"domain_{i:02d}", name=domain_names[i % len(domain_names)], paper_count=0)
        for i in range(max(8, len(domain_names)))
    ]

    ideas = [
        Idea(id=f"idea_{i:03d}", core_idea=f"核心创意 {i}")
        for i in range(max(20, scale // 50))
    ]

    tricks = [
        Trick(
            id=f"trick_{i:03d}",
            name=f"技巧 {i}",
            trick_type=rng.choice(["method-level", "experiment-level", "writing-level"]),
            purpose=rng.choice(["创新性", "可信度", "清晰度", "稳健性"]),
            description=f"技巧 {i} 的描述。",
        )
        for i in range(max(30, scale // 20))
    ]

    patterns = []
    pattern_count = max(12, scale // 30)
    for i in range(pattern_count):
        patterns.append(
            Pattern(
                id=f"pat_{i:03d}",
                pattern_name=f"套路 {i}",
                pattern_summary=f"套路 {i} 的摘要说明。",
                writing_guide=f"套路 {i} 的写作指导示例。",
                cluster_size=rng.randint(5, 50),
                coherence_score=round(rng.uniform(0.3, 0.9), 2),
                skeleton_count=rng.randint(1, 5),
                trick_count=rng.randint(1, 4),
            )
        )

    papers = [
        Paper(id=f"paper_{i:05d}", title=f"Mock Paper {i}", conference=rng.choice(["ICLR", "NeurIPS", "CVPR"]))
        for i in range(scale)
    ]

    reviews = [
        Review(
            id=f"review_{i:05d}",
            paper_id=f"paper_{i:05d}",
            reviewer=f"reviewer_{i % 5}",
            overall_score=str(round(rng.uniform(5.0, 9.0), 1)),
            confidence=str(rng.randint(1, 5)),
            strengths="优点示例",
            weaknesses="缺点示例",
        )
        for i in range(scale)
    ]

    defect_types = [
        "novelty_low",
        "experiments_weak",
        "theory_missing",
        "clarity_low",
        "significance_low",
        "reproducibility_low",
        "robustness_weak",
        "efficiency_low",
        "ablation_missing",
        "baseline_weak",
    ]
    defects = [
        Defect(
            defect_id=f"defect_{dtype}",
            type=dtype,
            description=f"{dtype} 的缺陷说明。",
            frequency=round(rng.uniform(0.2, 0.8), 2),
        )
        for dtype in defect_types
    ]

    return {
        "domains": domains,
        "ideas": ideas,
        "tricks": tricks,
        "patterns": patterns,
        "papers": papers,
        "reviews": reviews,
        "defects": defects,
        "skeletons": [],
    }


def build_mock_kg(
    snapshot_path: str,
    raw_path: Optional[str] = None,
    scale: Optional[int] = None,
    seed: Optional[int] = None,
) -> str:
    """
    功能：构建 mock 知识图谱快照。
    参数：snapshot_path、raw_path、scale、seed。
    返回：快照文件路径。
    流程：生成 mock 数据 → 写入图谱与原始存储。
    说明：供 CLI 与测试使用。
    """
    graph = NetworkXGraphStore()
    data = _build_mock_data_scaled(scale, seed) if scale else _build_mock_data()
    rng = random.Random(seed)

    for domain in data["domains"]:
        graph.upsert_node("Domain", domain.id, domain.to_dict())
    for idea in data["ideas"]:
        graph.upsert_node("Idea", idea.id, idea.to_dict())
    for trick in data["tricks"]:
        graph.upsert_node("Trick", trick.id, trick.to_dict())
    for pattern in data["patterns"]:
        graph.upsert_node("Pattern", pattern.id, pattern.to_dict())
    for skeleton in data.get("skeletons", []):
        graph.upsert_node("Skeleton", skeleton.id, skeleton.to_dict())
    for defect in data.get("defects", []):
        graph.upsert_node("Defect", defect.defect_id, defect.to_dict())

    trick_ids = [trick.id for trick in data["tricks"]]
    pattern_trick_map: Dict[str, List[str]] = {}
    for pattern in data["patterns"]:
        if not trick_ids:
            break
        picked = rng.sample(trick_ids, k=min(len(trick_ids), rng.randint(2, 3)))
        pattern_trick_map[pattern.id] = picked
        for order, trick_id in enumerate(picked, start=1):
            graph.upsert_edge(
                "Pattern",
                pattern.id,
                "CONTAINS_TRICK",
                "Trick",
                trick_id,
                {"order": order},
            )

    idea_ids = [idea.id for idea in data["ideas"]]
    domain_ids = [domain.id for domain in data["domains"]]
    pattern_ids = [pattern.id for pattern in data["patterns"]]

    for paper in data["papers"]:
        graph.upsert_node("Paper", paper.id, paper.to_dict())
        idea_id = rng.choice(idea_ids) if idea_ids else ""
        domain_id = rng.choice(domain_ids) if domain_ids else ""
        pattern_id = rng.choice(pattern_ids) if pattern_ids else ""

        if idea_id:
            graph.upsert_edge("Paper", paper.id, "HAS_IDEA", "Idea", idea_id, {})
        if domain_id:
            graph.upsert_edge("Paper", paper.id, "IN_DOMAIN", "Domain", domain_id, {})
        if pattern_id:
            graph.upsert_edge(
                "Paper", paper.id, "BELONGS_TO_PATTERN", "Pattern", pattern_id, {}
            )
            for trick_id in pattern_trick_map.get(pattern_id, []):
                graph.upsert_edge("Paper", paper.id, "USES_TRICK", "Trick", trick_id, {})

    for review in data["reviews"]:
        graph.upsert_node("Review", review.id, review.to_dict())
        if review.paper_id:
            graph.upsert_edge("Paper", review.paper_id, "HAS_REVIEW", "Review", review.id, {})

    # 中文注释：为缺陷驱动修复准备 Review -> Defect / Defect -> Trick 的示例关系。
    defect_type_to_id = {defect.type: defect.defect_id for defect in data.get("defects", [])}
    if defect_type_to_id:
        for review in data["reviews"]:
            if rng.random() < 0.5:
                defect_type = rng.choice(list(defect_type_to_id.keys()))
                graph.upsert_edge(
                    "Review",
                    review.id,
                    "IDENTIFIES",
                    "Defect",
                    defect_type_to_id[defect_type],
                    {"type": defect_type},
                )
        for defect_type, defect_id in defect_type_to_id.items():
            fix_tricks = rng.sample(trick_ids, k=min(2, len(trick_ids))) if trick_ids else []
            for trick_id in fix_tricks:
                graph.upsert_edge(
                    "Defect",
                    defect_id,
                    "FIXED_BY",
                    "Trick",
                    trick_id,
                    {"success_rate": round(rng.uniform(0.5, 0.9), 2)},
                )

    snapshot = _ensure_path(snapshot_path)
    snapshot.write_text(json.dumps(graph.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    if raw_path:
        doc_store = JsonDocStore(raw_path)
        for paper in data["papers"]:
            doc_store.save_raw_paper(paper.id, paper.to_dict())
        for review in data["reviews"]:
            doc_store.save_raw_review(review.id, review.to_dict())

    return str(snapshot)


def _build_vector_store(
    graph: NetworkXGraphStore, embedding_model
) -> InMemoryVectorStore:
    """
    功能：从图谱构建向量索引。
    参数：graph、embedding_model。
    返回：InMemoryVectorStore。
    流程：遍历 Paper → 组合套路 → 嵌入 → 写入向量库。
    说明：用于新颖性查重。
    """
    store = InMemoryVectorStore()
    logger.info("VECTOR_STORE_BUILD_START")
    for paper in graph.list_nodes("Paper"):
        paper_id = paper.get("id") or paper.get("node_id")
        if not paper_id:
            continue
        pattern_nodes = graph.neighbors("Paper", paper_id, rel="BELONGS_TO_PATTERN", direction="out")
        patterns = [Pattern.from_dict(node) for node in pattern_nodes]
        if not patterns:
            continue
        text = encode_pattern_config(patterns)
        logger.info("EMBEDDING_START: paper_id=%s", paper_id)
        try:
            vector = embedding_model.embed(text)
            logger.info("EMBEDDING_OK: paper_id=%s", paper_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("EMBEDDING_FAIL: paper_id=%s err=%s", paper_id, exc)
            raise
        metadata = {
            "paper_id": paper_id,
            "pattern_ids": [p.id for p in patterns],
        }
        store.upsert(str(paper_id), vector, metadata)
    logger.info("VECTOR_STORE_BUILD_OK")
    return store


def _format_final_story(draft: StoryDraft) -> str:
    """
    功能：格式化最终故事输出。
    参数：draft。
    返回：格式化字符串。
    流程：拼接标题与框架文本。
    说明：用于控制台展示。
    """
    lines = [
        "",
        "=== Final Story Draft ===",
        f"Idea: {draft.idea}",
        f"Domain: {draft.domain}",
        f"Patterns: {', '.join([p.pattern_name for p in draft.patterns])}",
        f"Tricks: {', '.join([t.name for t in draft.tricks])}",
        "",
        draft.final_story,
    ]
    return "\n".join(lines)


def _format_final_paper_reply(report_dict: dict) -> str:
    """
    功能：把论文工作流输出整理为更像“最终给用户的回复”的控制台文本。
    参数：report_dict（PaperWorkflowReport.to_dict 的结果）。
    返回：格式化后的字符串。
    流程：读取草稿与骨架字段 → 拼接输出。
    说明：为了保持 CLI 简洁，这里直接基于 dict 进行格式化。
    """
    status = report_dict.get("status", "")
    core_idea = report_dict.get("core_idea", "")
    domain = report_dict.get("domain", "")
    patterns = report_dict.get("patterns") or []
    tricks = report_dict.get("tricks") or []
    draft = report_dict.get("draft") or {}
    skeleton = (draft.get("story_skeleton") or {}) if isinstance(draft, dict) else {}
    final_story = draft.get("final_story", "") if isinstance(draft, dict) else ""
    paper_plan = draft.get("paper_plan", {}) if isinstance(draft, dict) else {}
    paper_markdown = draft.get("paper_markdown", "") if isinstance(draft, dict) else ""
    paper_latex = draft.get("paper_latex", "") if isinstance(draft, dict) else ""
    overleaf_project_dir = draft.get("overleaf_project_dir", "") if isinstance(draft, dict) else ""
    overleaf_zip_path = draft.get("overleaf_zip_path", "") if isinstance(draft, dict) else ""
    latex_compile_report = draft.get("latex_compile_report", {}) if isinstance(draft, dict) else {}
    warnings = report_dict.get("warnings") or []

    lines = [
        "",
        "=== Final Paper Draft ===",
        f"Status: {status}",
        f"Domain: {domain}",
        f"Core Idea: {core_idea}",
        f"Patterns: {', '.join([p.get('pattern_name','') for p in patterns if isinstance(p, dict)])}",
        f"Tricks: {', '.join([t.get('name','') for t in tricks if isinstance(t, dict)])}",
        "",
        "Story Skeleton:",
        f"- 问题定义: {skeleton.get('problem_framing','')}",
        f"- 研究缺口: {skeleton.get('gap_pattern','')}",
        f"- 方法叙述: {skeleton.get('method_story','')}",
        f"- 实验策略: {skeleton.get('experiments_story','')}",
        "",
    ]
    if paper_plan:
        lines.append("Paper Plan:")
        for section in paper_plan.get("sections", []) or []:
            section_id = section.get("section_id", "")
            title = section.get("title", "")
            tricks = "、".join(section.get("required_tricks", []) or [])
            lines.append(f"- {section_id} {title} | tricks: {tricks}")
        lines.append("")

    if paper_latex:
        lines.append("Paper LaTeX:")
        lines.append(str(paper_latex).strip())
    elif paper_markdown:
        lines.append("Paper Markdown:")
        lines.append(str(paper_markdown).strip())
    else:
        lines.append("Final Story:")
        lines.append(str(final_story or "").strip())
    if overleaf_project_dir:
        lines.append("")
        lines.append(f"Overleaf Project Dir: {overleaf_project_dir}")
    if overleaf_zip_path:
        lines.append(f"Overleaf Zip: {overleaf_zip_path}")
    if latex_compile_report:
        lines.append(f"LaTeX Compile Report: {latex_compile_report}")
    if warnings:
        lines.append("")
        lines.append(f"Warnings: {', '.join([str(w) for w in warnings])}")
    return "\n".join(lines)


@app.command("build-mock-kg")
def build_mock_kg_cmd(
    snapshot_path: str = typer.Option(
        "./paper_kg_snapshot.json",
        "--out",
        "--snapshot-path",
        help="图谱快照路径（可自定义输出文件）",
    ),
    scale: Optional[int] = typer.Option(
        None,
        help="可选：生成大规模 mock 数据的论文数量（例如 5000），为空则生成小规模",
    ),
    seed: Optional[int] = typer.Option(None, help="随机种子，保证 mock 数据可复现"),
    raw_path: Optional[str] = typer.Option(None, help="原始数据 JSON 路径"),
) -> None:
    """
    功能：CLI 命令：构建 mock 图谱快照。
    参数：snapshot_path、scale、seed、raw_path。
    返回：无。
    流程：调用 build_mock_kg → 输出路径。
    说明：适合本地演示。
    """
    setup_logging()
    snapshot = build_mock_kg(snapshot_path, raw_path=raw_path, scale=scale, seed=seed)
    typer.echo(f"Mock KG snapshot saved: {snapshot}")


@app.command("generate-story")
def generate_story_cmd(
    idea: str = typer.Option(..., help="核心 Idea 文本"),
    domain: str = typer.Option(..., help="目标领域名称"),
    snapshot_path: Optional[str] = typer.Option(None, help="图谱快照路径"),
    max_patterns: int = typer.Option(3, help="最大套路数量"),
    temperature: float = typer.Option(0.8, help="多样性温度"),
    novelty_threshold: float = typer.Option(0.85, help="新颖性阈值"),
    use_llm: bool = typer.Option(False, help="是否使用 LLM 生成故事"),
    use_mock_embedding: bool = typer.Option(False, help="使用 Mock embedding（仅离线测试）"),
) -> None:
    """
    功能：CLI 命令：生成故事初版。
    参数：idea、domain、snapshot_path、max_patterns、temperature、novelty_threshold、use_llm。
    返回：无。
    流程：加载图谱 → 构建向量库 → 生成故事。
    说明：默认使用模板模式。
    """
    setup_logging()
    load_env()

    snapshot_path = snapshot_path or _default_snapshot_path()
    snapshot = Path(snapshot_path)
    if not snapshot.exists():
        if snapshot.name == "paper_kg_snapshot.json":
            build_mock_kg(str(snapshot), raw_path=None)
        else:
            raise FileNotFoundError(f"Snapshot not found: {snapshot}")

    graph = NetworkXGraphStore()
    graph.load_dict(json.loads(snapshot.read_text(encoding="utf-8")))

    if bool(config.paper_workflow.skip_novelty_check):
        logger.info("PAPER_SKIP_NOVELTY: skip embedding/vector_store build")
        embedding_model = MockEmbeddingModel()
        vector_store = InMemoryVectorStore()
    else:
        if use_mock_embedding:
            embedding_model = MockEmbeddingModel()
        else:
            embedding_model = create_embedding_model("paper_embedding", required=True)
            if embedding_model is None:
                raise ValueError("Embedding model is required, please set API key in .env")

        vector_store = _build_vector_store(graph, embedding_model)

    llm = create_chat_model("paper_story") if use_llm else None
    orchestrator = StoryOrchestrator(
        graph_store=graph,
        vector_store=vector_store,
        embedding_model=embedding_model,
        rng=random.Random(),
        llm=llm,
    )

    draft = orchestrator.build_story(
        idea_text=idea,
        domain_name=domain,
        max_patterns=max_patterns,
        temperature=temperature,
        novelty_threshold=novelty_threshold,
    )

    typer.echo(json.dumps(draft.model_dump(), ensure_ascii=False, indent=2))
    typer.echo(_format_final_story(draft))


@app.command("workflow")
def paper_workflow_cmd(
    text: str = typer.Argument(..., help="用户一句自然语言的论文想法"),
    snapshot_path: Optional[str] = typer.Option(None, help="图谱快照路径"),
    config_path: Optional[str] = typer.Option(None, "--config", help="YAML 配置文件路径"),
    use_llm: bool = typer.Option(True, help="是否使用真实 LLM（默认 True）"),
    use_mock_embedding: bool = typer.Option(False, help="使用 Mock embedding（仅离线测试）"),
    seed: Optional[int] = typer.Option(None, "--seed", help="覆盖随机种子（便于复现）"),
) -> None:
    """
    功能：论文链路一键工作流：输入一句话 → 选套路/查重 → 写稿 → 多审稿人评审回退 → 输出最终稿。
    参数：text、snapshot_path、config_path、use_llm、use_mock_embedding、seed。
    返回：无（控制台输出 JSON 报告 + 最终稿）。
    流程：
    1) 加载配置与环境变量；
    2) 确保图谱快照存在（不存在且为默认 mock 才自动构建）；
    3) 构建 vector_store（用于 novelty 查重）；
    4) 初始化 LangGraph controller 并运行；
    5) 输出 JSON 报告与 Final Paper Draft。
    """
    setup_logging()
    load_env()

    # 中文注释：当未显式传入 --config 时，自动加载项目根目录的 config_paper.yaml。
    if not config_path:
        default_config = Path("config_paper.yaml")
        if default_config.exists():
            config_path = str(default_config)

    config = load_config(config_path)
    prompt_dir = str(config.paper_workflow.paper_prompt_dir or "").strip()
    if prompt_dir:
        if not Path(prompt_dir).exists():
            raise FileNotFoundError(f"paper_prompt_dir not found: {prompt_dir}")
        os.environ["PAPER_PROMPT_DIR"] = str(Path(prompt_dir).resolve())
    else:
        language = str(config.paper_workflow.paper_language or "en").lower()
        default_dir = "prompts_paper_en" if language == "en" else "prompts_paper"
        os.environ["PAPER_PROMPT_DIR"] = str(Path(default_dir).resolve())
    step_log_path = ""
    if bool(config.paper_workflow.step_log_enable):
        step_log_path = add_step_summary_file_handler(
            str(config.paper_workflow.step_log_path or "")
        )
        logger.info("PAPER_STEP_LOG_PATH: %s", step_log_path)
        typer.echo(f"[step-log] {step_log_path}")

    if seed is not None:
        config.paper_workflow.seed = seed
    if config.paper_workflow.seed is None:
        config.paper_workflow.seed = secrets.randbits(32)

    snapshot_path = snapshot_path or _default_snapshot_path()
    snapshot = Path(snapshot_path)
    if not snapshot.exists():
        if snapshot.name == "paper_kg_snapshot.json":
            build_mock_kg(str(snapshot), raw_path=None)
        else:
            raise FileNotFoundError(f"Snapshot not found: {snapshot}")
    logger.info("PAPER_WORKFLOW_START: snapshot=%s", snapshot.as_posix())

    graph = NetworkXGraphStore()
    graph.load_dict(json.loads(snapshot.read_text(encoding="utf-8")))

    # 中文注释：当配置为跳过查重时，不构建向量库/不调用 embedding，避免不必要的耗时与费用。
    # 注意：工作流内部仍会需要 embedding_model/vector_store 这两个对象占位（用于统一接口与状态结构），
    # 但在 skip_novelty_check=true 的路径上不会实际执行检索与相似度计算。
    if bool(config.paper_workflow.skip_novelty_check):
        logger.info("PAPER_SKIP_NOVELTY: skip embedding/vector_store build")
        embedding_model = MockEmbeddingModel()
        vector_store = InMemoryVectorStore()
    else:
        if use_mock_embedding:
            embedding_model = MockEmbeddingModel()
        else:
            embedding_model = create_embedding_model("paper_embedding", required=True)
            if embedding_model is None:
                raise ValueError("Embedding model is required, please set API key in .env")

        vector_store = _build_vector_store(graph, embedding_model)

    # 中文注释：为每个 Agent 单独创建 LLM，支持按 agent 前缀配置 API/model/base_url。
    parse_llm = create_chat_model("paper_parse") if use_llm else None
    story_llm = create_chat_model("paper_story") if use_llm else None
    section_llm = create_chat_model("paper_section_writer") if use_llm else None
    citation_need_llm = create_chat_model("paper_citation_need") if use_llm else None
    citation_retrieval_llm = create_chat_model("paper_citation_retrieval") if use_llm else None
    claim_resolution_llm = create_chat_model("paper_claim_resolution") if use_llm else None
    latex_fix_llm = create_chat_model("paper_latex_fix") if use_llm else None

    input_parser = PaperInputParser(llm=parse_llm)
    draft_writer = PaperDraftWriter(
        llm=story_llm,
        guide_limit=int(config.paper_workflow.writing_guide_limit_chars),
        max_guides=int(config.paper_workflow.writing_guide_max_guides),
    )
    paper_template = load_paper_template(config.paper_workflow.template_path)

    # 中文注释：LaTeX 模式下初始化工具链，用于编译闸门。
    latex_toolchain = None
    if str(config.paper_workflow.output_format) == "latex":
        latex_toolchain = LocalLatexToolchain(
            require_tools=bool(config.paper_workflow.latex_require_tools),
        )

    citation_retrieval_agent = None
    if citation_retrieval_llm is not None:
        logger.warning("CITATIONS_MOCK_ENABLED: provider=fixed_mock")
        tool = MockSemanticScholarTool()
        citation_retrieval_agent = CitationRetrievalAgent(
            llm=citation_retrieval_llm,
            tool=tool,
            require_verifiable=bool(config.paper_workflow.citations_require_verifiable),
            max_candidates=int(config.paper_workflow.retriever_top_k),
        )

    claim_resolution_agent = (
        ClaimResolutionAgent(claim_resolution_llm) if claim_resolution_llm is not None else None
    )

    latex_renderer_agent = PandocLatexRendererAgent(
        pandoc_runner=SubprocessPandocRunner(
            pandoc_bin=str(config.paper_workflow.pandoc_bin),
            require=True,
        ),
        fix_llm=latex_fix_llm,
        consistency_threshold=float(config.paper_workflow.pandoc_consistency_threshold),
        fail_on_content_loss=bool(config.paper_workflow.pandoc_fail_on_content_loss),
    )

    controller = PaperFeedbackLoopController(
        config=config.paper_workflow,
        graph_store=graph,
        vector_store=vector_store,
        embedding_model=embedding_model,
        rng=random.Random(config.paper_workflow.seed),
        input_parser=input_parser,
        draft_writer=draft_writer,
        section_llm=section_llm,
        citation_need_llm=citation_need_llm,
        citation_retrieval_agent=citation_retrieval_agent,
        claim_resolution_agent=claim_resolution_agent,
        latex_renderer_agent=latex_renderer_agent,
        paper_template=paper_template,
        latex_toolchain=latex_toolchain,
    )

    try:
        report = controller.run(text)
    except Exception:
        logger.exception("PAPER_WORKFLOW_FAIL")
        if step_log_path:
            typer.echo(f"[step-log] {step_log_path}")
        raise

    report_dict = report.to_dict()
    logger.info("PAPER_WORKFLOW_OK: status=%s", report_dict.get("status"))
    typer.echo(json.dumps(report_dict, ensure_ascii=False, indent=2))
    typer.echo(_format_final_paper_reply(report_dict))


if __name__ == "__main__":
    app()
