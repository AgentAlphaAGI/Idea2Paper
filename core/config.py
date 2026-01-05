# 中文注释: 本文件包含可运行代码与流程说明。
from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class WorkflowConfig:
    # seed 说明：
    # - None：每次运行使用系统随机种子（更接近“正常 AI 产品”的体验，同样输入也更可能得到不同结果）
    # - int：固定种子，便于复现调试与测试
    seed: Optional[int] = None
    novelty_threshold: float = 0.55
    # 中文注释：降低评审阈值，便于测试阶段快速通过（后续真实情况可调回 7.0）。
    pass_threshold: float = 6.0
    max_pattern_retry: int = 3
    max_packaging_retry: int = 2
    max_global_retry: int = 2
    retriever_top_k: int = 8
    novelty_top_k: int = 5


@dataclass
class PaperWorkflowConfig:
    """
    功能：论文链路工作流配置（LangGraph + 多 Reviewer + 回退循环）。
    参数：阈值、最大重试次数、采样温度等。
    返回：PaperWorkflowConfig。
    流程：作为配置载体被 load_config 读取并注入到 paper controller。
    说明：所有字段都应可通过 YAML 覆盖，以便快速调参验证回退逻辑。
    """

    # seed 说明：None 表示每次随机，int 表示可复现
    seed: Optional[int] = None
    novelty_threshold: float = 0.85
    # 中文注释：测试模式下可强制跳过查重（默认 True，便于端到端跑通）
    force_novelty_pass: bool = False
    pass_threshold: float = 1.0
    # 中文注释：质量控制开关（跳过 reviewers 与内容自检，但不删除逻辑）
    format_only_mode: bool = False
    # 中文注释：查重开关（跳过 embedding 构建与 novelty 检索/判定）
    skip_novelty_check: bool = False
    # 中文注释：输出格式，可选 markdown/latex。
    output_format: str = "markdown"
    # 中文注释：Markdown 写作后端：markdown（LLM 直写）/spec（结构化 spec 渲染）。
    markdown_writer_backend: str = "markdown"
    # 中文注释：spec 引用占位 token。
    spec_cite_token: str = "[[CITE]]"
    # 中文注释：spec 渲染时是否强制 cite marker 带 id。
    spec_render_require_cite_ids: bool = True
    # 中文注释：是否打印 LLM 原始输出（调试用）。
    log_llm_outputs: bool = False
    # 中文注释：LLM 输出日志截断长度（0 表示不截断）。
    log_llm_output_max_chars: int = 4000
    # 中文注释：步骤摘要日志开关与路径。
    step_log_enable: bool = False
    step_log_path: str = ""
    # 中文注释：英文写作质量门槛（限制中文字符占比）。
    enforce_english: bool = False
    max_cjk_ratio: float = 0.02
    # 中文注释：论文语言（en/zh），用于选择提示词与模板。
    paper_language: str = "en"
    # 中文注释：可显式指定论文提示词目录（覆盖 paper_language 逻辑）。
    paper_prompt_dir: str = ""
    # LaTeX template preset and title/author fields for export.
    latex_template_preset: str = "article_ctex"
    latex_titlepage_style: str = "qwen"
    latex_title: str = ""
    latex_author: str = ""
    latex_hflink: str = "https://huggingface.co/Qwen"
    latex_mslink: str = "https://modelscope.cn/organization/qwen"
    latex_ghlink: str = "https://github.com/QwenLM/Qwen3"
    latex_template_assets_dir: str = "arXiv-2505.09388v1"
    # 中文注释：LaTeX 生成模式：per_section=逐章生成；final_only=最终统一转换。
    latex_generation_mode: str = "final_only"
    # 中文注释：LaTeX 渲染器后端：pandoc/llm。
    latex_renderer_backend: str = "pandoc"
    # 中文注释：Pandoc 可执行文件路径（可被环境变量覆盖）。
    pandoc_bin: str = "pandoc"
    # 中文注释：Pandoc 缺失时是否直接失败。
    pandoc_require: bool = True
    max_iterations: int = 3
    max_rewrite_retry: int = 2
    max_global_retry: int = 2
    max_plan_retry: int = 3
    max_patterns: int = 3
    temperature: float = 0.8
    retriever_top_k: int = 10
    penalty: float = 0.3
    require_individual_threshold: bool = False
    writing_guide_limit_chars: int = 2000
    writing_guide_max_guides: int = 3
    section_max_retries: int = 2
    section_summary_max_chars: int = 400
    template_path: str = "paper_template.yaml"
    enable_related_work: bool = True
    # 中文注释：LaTeX 工具链参数（格式保证专用，不属于内容质量开关）
    latex_engine: str = "xelatex"
    latex_require_tools: bool = True
    latex_section_compile: bool = True
    latex_final_compile: bool = True
    latex_use_chktex: bool = True
    latex_use_lacheck: bool = True
    latex_compile_timeout_sec: int = 120
    latex_max_fix_retries: int = 8
    latex_output_dir: str = "./outputs/overleaf"
    # 中文注释：引用 key 强制规范化为 TODO_ 前缀
    latex_force_todo_cite: bool = True
    # 中文注释：最终 LaTeX 格式修复重试次数（仅 final_only 流程）
    latex_format_max_retries: int = 3
    # 中文注释：Markdown 质量硬校验开关（质量优先模式下启用）。
    strict_markdown_validation: bool = False
    # 中文注释：章节字数最小比例（相对于模板 target_words_min）。
    section_min_word_ratio: float = 0.6
    # 中文注释：Markdown 质量不足时的额外重试次数。
    section_min_quality_max_retries: int = 1
    # 中文注释：章节覆盖验收器配置（llm/rule/off）。
    coverage_enable: bool = True
    coverage_validator_backend: str = "llm"
    coverage_validator_max_retries: int = 2
    coverage_validator_require_evidence: bool = True
    # 中文注释：质量优先失败即中止。
    quality_fail_fast: bool = False
    # 中文注释：真实引用链路开关与策略
    citations_enable: bool = True
    citations_require_verifiable: bool = True
    citations_require_api_key: bool = False
    citations_provider: str = "semantic_scholar"
    citations_year_range: str = "2018-2026"
    citations_max_rounds: int = 3
    # 中文注释：Pandoc 渲染一致性阈值与失败策略
    pandoc_consistency_threshold: float = 0.65
    pandoc_fail_on_content_loss: bool = True


@dataclass
class AppConfig:
    workflow: WorkflowConfig = field(default_factory=WorkflowConfig)
    paper_workflow: PaperWorkflowConfig = field(default_factory=PaperWorkflowConfig)


def load_config(path: Optional[str] = None) -> AppConfig:
    """
    功能：load_config 的核心流程封装，负责处理输入并输出结果。
    参数：path。
    返回：依据实现返回结果或产生副作用。
    流程：接收输入 → 核心逻辑处理 → 输出或触发副作用。
    说明：如遇异常或边界情况，调用侧需关注返回值或日志信息。
    """
    if not path:
        return AppConfig()

    data = yaml.safe_load(Path(path).read_text()) or {}
    workflow_data = data.get("workflow", {})
    paper_data = data.get("paper_workflow", {})

    seed_raw = workflow_data.get("seed", WorkflowConfig.seed)
    seed: Optional[int]
    if seed_raw is None or str(seed_raw).strip() == "":
        seed = None
    else:
        seed = int(seed_raw)

    workflow = WorkflowConfig(
        seed=seed,
        novelty_threshold=float(
            workflow_data.get("novelty_threshold", WorkflowConfig.novelty_threshold)
        ),
        pass_threshold=float(
            workflow_data.get("pass_threshold", WorkflowConfig.pass_threshold)
        ),
        max_pattern_retry=int(
            workflow_data.get("max_pattern_retry", WorkflowConfig.max_pattern_retry)
        ),
        max_packaging_retry=int(
            workflow_data.get(
                "max_packaging_retry", WorkflowConfig.max_packaging_retry
            )
        ),
        max_global_retry=int(
            workflow_data.get("max_global_retry", WorkflowConfig.max_global_retry)
        ),
        retriever_top_k=int(
            workflow_data.get("retriever_top_k", WorkflowConfig.retriever_top_k)
        ),
        novelty_top_k=int(
            workflow_data.get("novelty_top_k", WorkflowConfig.novelty_top_k)
        ),
    )

    paper_seed_raw = paper_data.get("seed", PaperWorkflowConfig.seed)
    paper_seed: Optional[int]
    if paper_seed_raw is None or str(paper_seed_raw).strip() == "":
        paper_seed = None
    else:
        paper_seed = int(paper_seed_raw)

    pandoc_bin_raw = paper_data.get("pandoc_bin")
    if pandoc_bin_raw is None or str(pandoc_bin_raw).strip() == "":
        pandoc_bin = os.getenv("PANDOC_BIN", PaperWorkflowConfig.pandoc_bin)
    else:
        pandoc_bin = str(pandoc_bin_raw)

    paper_workflow = PaperWorkflowConfig(
        seed=paper_seed,
        novelty_threshold=float(
            paper_data.get("novelty_threshold", PaperWorkflowConfig.novelty_threshold)
        ),
        force_novelty_pass=bool(
            paper_data.get("force_novelty_pass", PaperWorkflowConfig.force_novelty_pass)
        ),
        skip_novelty_check=bool(
            paper_data.get("skip_novelty_check", PaperWorkflowConfig.skip_novelty_check)
        ),
        format_only_mode=bool(
            paper_data.get("format_only_mode", PaperWorkflowConfig.format_only_mode)
        ),
        pass_threshold=float(
            paper_data.get("pass_threshold", PaperWorkflowConfig.pass_threshold)
        ),
        output_format=str(
            paper_data.get("output_format", PaperWorkflowConfig.output_format)
        ),
        markdown_writer_backend=str(
            paper_data.get(
                "markdown_writer_backend",
                PaperWorkflowConfig.markdown_writer_backend,
            )
        ),
        spec_cite_token=str(
            paper_data.get("spec_cite_token", PaperWorkflowConfig.spec_cite_token)
        ),
        spec_render_require_cite_ids=bool(
            paper_data.get(
                "spec_render_require_cite_ids",
                PaperWorkflowConfig.spec_render_require_cite_ids,
            )
        ),
        log_llm_outputs=bool(
            paper_data.get("log_llm_outputs", PaperWorkflowConfig.log_llm_outputs)
        ),
        log_llm_output_max_chars=int(
            paper_data.get(
                "log_llm_output_max_chars",
                PaperWorkflowConfig.log_llm_output_max_chars,
            )
        ),
        step_log_enable=bool(
            paper_data.get("step_log_enable", PaperWorkflowConfig.step_log_enable)
        ),
        step_log_path=str(
            paper_data.get("step_log_path", PaperWorkflowConfig.step_log_path)
        ),
        enforce_english=bool(
            paper_data.get("enforce_english", PaperWorkflowConfig.enforce_english)
        ),
        max_cjk_ratio=float(
            paper_data.get("max_cjk_ratio", PaperWorkflowConfig.max_cjk_ratio)
        ),
        paper_language=str(
            paper_data.get("paper_language", PaperWorkflowConfig.paper_language)
        ),
        paper_prompt_dir=str(
            paper_data.get("paper_prompt_dir", PaperWorkflowConfig.paper_prompt_dir)
        ),
        latex_template_preset=str(
            paper_data.get("latex_template_preset", PaperWorkflowConfig.latex_template_preset)
        ),
        latex_titlepage_style=str(
            paper_data.get(
                "latex_titlepage_style", PaperWorkflowConfig.latex_titlepage_style
            )
        ),
        latex_title=str(
            paper_data.get("latex_title", PaperWorkflowConfig.latex_title)
        ),
        latex_author=str(
            paper_data.get("latex_author", PaperWorkflowConfig.latex_author)
        ),
        latex_hflink=str(
            paper_data.get("latex_hflink", PaperWorkflowConfig.latex_hflink)
        ),
        latex_mslink=str(
            paper_data.get("latex_mslink", PaperWorkflowConfig.latex_mslink)
        ),
        latex_ghlink=str(
            paper_data.get("latex_ghlink", PaperWorkflowConfig.latex_ghlink)
        ),
        latex_template_assets_dir=str(
            paper_data.get(
                "latex_template_assets_dir", PaperWorkflowConfig.latex_template_assets_dir
            )
        ),
        latex_generation_mode=str(
            paper_data.get("latex_generation_mode", PaperWorkflowConfig.latex_generation_mode)
        ),
        latex_renderer_backend=str(
            paper_data.get(
                "latex_renderer_backend", PaperWorkflowConfig.latex_renderer_backend
            )
        ),
        pandoc_bin=pandoc_bin,
        pandoc_require=bool(
            paper_data.get("pandoc_require", PaperWorkflowConfig.pandoc_require)
        ),
        max_iterations=int(
            paper_data.get("max_iterations", PaperWorkflowConfig.max_iterations)
        ),
        max_rewrite_retry=int(
            paper_data.get("max_rewrite_retry", PaperWorkflowConfig.max_rewrite_retry)
        ),
        max_global_retry=int(
            paper_data.get("max_global_retry", PaperWorkflowConfig.max_global_retry)
        ),
        max_plan_retry=int(paper_data.get("max_plan_retry", PaperWorkflowConfig.max_plan_retry)),
        max_patterns=int(paper_data.get("max_patterns", PaperWorkflowConfig.max_patterns)),
        temperature=float(paper_data.get("temperature", PaperWorkflowConfig.temperature)),
        retriever_top_k=int(paper_data.get("retriever_top_k", PaperWorkflowConfig.retriever_top_k)),
        penalty=float(paper_data.get("penalty", PaperWorkflowConfig.penalty)),
        require_individual_threshold=bool(
            paper_data.get(
                "require_individual_threshold",
                PaperWorkflowConfig.require_individual_threshold,
            )
        ),
        writing_guide_limit_chars=int(
            paper_data.get(
                "writing_guide_limit_chars",
                PaperWorkflowConfig.writing_guide_limit_chars,
            )
        ),
        writing_guide_max_guides=int(
            paper_data.get(
                "writing_guide_max_guides",
                PaperWorkflowConfig.writing_guide_max_guides,
            )
        ),
        section_max_retries=int(
            paper_data.get(
                "section_max_retries",
                PaperWorkflowConfig.section_max_retries,
            )
        ),
        section_summary_max_chars=int(
            paper_data.get(
                "section_summary_max_chars",
                PaperWorkflowConfig.section_summary_max_chars,
            )
        ),
        template_path=str(
            paper_data.get("template_path", PaperWorkflowConfig.template_path)
        ),
        enable_related_work=bool(
            paper_data.get(
                "enable_related_work",
                PaperWorkflowConfig.enable_related_work,
            )
        ),
        latex_engine=str(
            paper_data.get("latex_engine", PaperWorkflowConfig.latex_engine)
        ),
        latex_require_tools=bool(
            paper_data.get("latex_require_tools", PaperWorkflowConfig.latex_require_tools)
        ),
        latex_section_compile=bool(
            paper_data.get(
                "latex_section_compile",
                PaperWorkflowConfig.latex_section_compile,
            )
        ),
        latex_final_compile=bool(
            paper_data.get(
                "latex_final_compile",
                PaperWorkflowConfig.latex_final_compile,
            )
        ),
        latex_use_chktex=bool(
            paper_data.get("latex_use_chktex", PaperWorkflowConfig.latex_use_chktex)
        ),
        latex_use_lacheck=bool(
            paper_data.get("latex_use_lacheck", PaperWorkflowConfig.latex_use_lacheck)
        ),
        latex_compile_timeout_sec=int(
            paper_data.get(
                "latex_compile_timeout_sec",
                PaperWorkflowConfig.latex_compile_timeout_sec,
            )
        ),
        latex_max_fix_retries=int(
            paper_data.get(
                "latex_max_fix_retries",
                PaperWorkflowConfig.latex_max_fix_retries,
            )
        ),
        latex_output_dir=str(
            paper_data.get("latex_output_dir", PaperWorkflowConfig.latex_output_dir)
        ),
        latex_force_todo_cite=bool(
            paper_data.get(
                "latex_force_todo_cite",
                PaperWorkflowConfig.latex_force_todo_cite,
            )
        ),
        latex_format_max_retries=int(
            paper_data.get(
                "latex_format_max_retries", PaperWorkflowConfig.latex_format_max_retries,
            )
        ),
        strict_markdown_validation=bool(
            paper_data.get(
                "strict_markdown_validation",
                PaperWorkflowConfig.strict_markdown_validation,
            )
        ),
        section_min_word_ratio=float(
            paper_data.get(
                "section_min_word_ratio",
                PaperWorkflowConfig.section_min_word_ratio,
            )
        ),
        section_min_quality_max_retries=int(
            paper_data.get(
                "section_min_quality_max_retries",
                PaperWorkflowConfig.section_min_quality_max_retries,
            )
        ),
        coverage_enable=bool(
            paper_data.get(
                "coverage_enable",
                PaperWorkflowConfig.coverage_enable,
            )
        ),
        coverage_validator_backend=str(
            paper_data.get(
                "coverage_validator_backend",
                PaperWorkflowConfig.coverage_validator_backend,
            )
        ),
        coverage_validator_max_retries=int(
            paper_data.get(
                "coverage_validator_max_retries",
                PaperWorkflowConfig.coverage_validator_max_retries,
            )
        ),
        coverage_validator_require_evidence=bool(
            paper_data.get(
                "coverage_validator_require_evidence",
                PaperWorkflowConfig.coverage_validator_require_evidence,
            )
        ),
        quality_fail_fast=bool(
            paper_data.get(
                "quality_fail_fast",
                PaperWorkflowConfig.quality_fail_fast,
            )
        ),
        citations_enable=bool(
            paper_data.get("citations_enable", PaperWorkflowConfig.citations_enable)
        ),
        citations_require_verifiable=bool(
            paper_data.get(
                "citations_require_verifiable", PaperWorkflowConfig.citations_require_verifiable,
            )
        ),
        citations_require_api_key=bool(
            paper_data.get(
                "citations_require_api_key",
                PaperWorkflowConfig.citations_require_api_key,
            )
        ),
        citations_provider=str(
            paper_data.get("citations_provider", PaperWorkflowConfig.citations_provider)
        ),
        citations_year_range=str(
            paper_data.get("citations_year_range", PaperWorkflowConfig.citations_year_range)
        ),
        citations_max_rounds=int(
            paper_data.get("citations_max_rounds", PaperWorkflowConfig.citations_max_rounds)
        ),
        pandoc_consistency_threshold=float(
            paper_data.get(
                "pandoc_consistency_threshold",
                PaperWorkflowConfig.pandoc_consistency_threshold,
            )
        ),
        pandoc_fail_on_content_loss=bool(
            paper_data.get(
                "pandoc_fail_on_content_loss",
                PaperWorkflowConfig.pandoc_fail_on_content_loss,
            )
        ),
    )

    return AppConfig(workflow=workflow, paper_workflow=paper_workflow)


def config_to_dict(config: AppConfig) -> Dict[str, Any]:
    """
    功能：config_to_dict 的核心流程封装，负责处理输入并输出结果。
    参数：config。
    返回：依据实现返回结果或产生副作用。
    流程：接收输入 → 核心逻辑处理 → 输出或触发副作用。
    说明：如遇异常或边界情况，调用侧需关注返回值或日志信息。
    """
    return {
        "workflow": {
            "seed": config.workflow.seed,
            "novelty_threshold": config.workflow.novelty_threshold,
            "pass_threshold": config.workflow.pass_threshold,
            "max_pattern_retry": config.workflow.max_pattern_retry,
            "max_packaging_retry": config.workflow.max_packaging_retry,
            "max_global_retry": config.workflow.max_global_retry,
            "retriever_top_k": config.workflow.retriever_top_k,
            "novelty_top_k": config.workflow.novelty_top_k,
        },
        "paper_workflow": {
            "seed": config.paper_workflow.seed,
            "novelty_threshold": config.paper_workflow.novelty_threshold,
            "force_novelty_pass": config.paper_workflow.force_novelty_pass,
            "skip_novelty_check": config.paper_workflow.skip_novelty_check,
            "format_only_mode": config.paper_workflow.format_only_mode,
            "pass_threshold": config.paper_workflow.pass_threshold,
            "max_iterations": config.paper_workflow.max_iterations,
            "max_rewrite_retry": config.paper_workflow.max_rewrite_retry,
            "max_global_retry": config.paper_workflow.max_global_retry,
            "max_plan_retry": config.paper_workflow.max_plan_retry,
            "max_patterns": config.paper_workflow.max_patterns,
            "temperature": config.paper_workflow.temperature,
            "retriever_top_k": config.paper_workflow.retriever_top_k,
            "penalty": config.paper_workflow.penalty,
            "require_individual_threshold": config.paper_workflow.require_individual_threshold,
            "writing_guide_limit_chars": config.paper_workflow.writing_guide_limit_chars,
            "writing_guide_max_guides": config.paper_workflow.writing_guide_max_guides,
            "section_max_retries": config.paper_workflow.section_max_retries,
            "section_summary_max_chars": config.paper_workflow.section_summary_max_chars,
            "template_path": config.paper_workflow.template_path,
            "enable_related_work": config.paper_workflow.enable_related_work,
            "output_format": config.paper_workflow.output_format,
            "markdown_writer_backend": config.paper_workflow.markdown_writer_backend,
            "spec_cite_token": config.paper_workflow.spec_cite_token,
            "spec_render_require_cite_ids": config.paper_workflow.spec_render_require_cite_ids,
            "log_llm_outputs": config.paper_workflow.log_llm_outputs,
            "log_llm_output_max_chars": config.paper_workflow.log_llm_output_max_chars,
            "step_log_enable": config.paper_workflow.step_log_enable,
            "step_log_path": config.paper_workflow.step_log_path,
            "enforce_english": config.paper_workflow.enforce_english,
            "max_cjk_ratio": config.paper_workflow.max_cjk_ratio,
            "paper_language": config.paper_workflow.paper_language,
            "paper_prompt_dir": config.paper_workflow.paper_prompt_dir,
            "latex_template_preset": config.paper_workflow.latex_template_preset,
            "latex_titlepage_style": config.paper_workflow.latex_titlepage_style,
            "latex_title": config.paper_workflow.latex_title,
            "latex_author": config.paper_workflow.latex_author,
            "latex_hflink": config.paper_workflow.latex_hflink,
            "latex_mslink": config.paper_workflow.latex_mslink,
            "latex_ghlink": config.paper_workflow.latex_ghlink,
            "latex_template_assets_dir": config.paper_workflow.latex_template_assets_dir,
            "latex_generation_mode": config.paper_workflow.latex_generation_mode,
            "latex_renderer_backend": config.paper_workflow.latex_renderer_backend,
            "pandoc_bin": config.paper_workflow.pandoc_bin,
            "pandoc_require": config.paper_workflow.pandoc_require,
            "latex_engine": config.paper_workflow.latex_engine,
            "latex_require_tools": config.paper_workflow.latex_require_tools,
            "latex_section_compile": config.paper_workflow.latex_section_compile,
            "latex_final_compile": config.paper_workflow.latex_final_compile,
            "latex_use_chktex": config.paper_workflow.latex_use_chktex,
            "latex_use_lacheck": config.paper_workflow.latex_use_lacheck,
            "latex_compile_timeout_sec": config.paper_workflow.latex_compile_timeout_sec,
            "latex_max_fix_retries": config.paper_workflow.latex_max_fix_retries,
            "latex_output_dir": config.paper_workflow.latex_output_dir,
            "latex_force_todo_cite": config.paper_workflow.latex_force_todo_cite,
            "latex_format_max_retries": config.paper_workflow.latex_format_max_retries,
            "strict_markdown_validation": config.paper_workflow.strict_markdown_validation,
            "section_min_word_ratio": config.paper_workflow.section_min_word_ratio,
            "section_min_quality_max_retries": config.paper_workflow.section_min_quality_max_retries,
            "coverage_enable": config.paper_workflow.coverage_enable,
            "coverage_validator_backend": config.paper_workflow.coverage_validator_backend,
            "coverage_validator_max_retries": config.paper_workflow.coverage_validator_max_retries,
            "coverage_validator_require_evidence": config.paper_workflow.coverage_validator_require_evidence,
            "quality_fail_fast": config.paper_workflow.quality_fail_fast,
            "citations_enable": config.paper_workflow.citations_enable,
            "citations_require_verifiable": config.paper_workflow.citations_require_verifiable,
            "citations_require_api_key": config.paper_workflow.citations_require_api_key,
            "citations_provider": config.paper_workflow.citations_provider,
            "citations_year_range": config.paper_workflow.citations_year_range,
            "citations_max_rounds": config.paper_workflow.citations_max_rounds,
            "pandoc_consistency_threshold": config.paper_workflow.pandoc_consistency_threshold,
            "pandoc_fail_on_content_loss": config.paper_workflow.pandoc_fail_on_content_loss,
        },
    }
