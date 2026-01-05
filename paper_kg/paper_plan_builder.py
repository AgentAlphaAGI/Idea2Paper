# 中文注释: 本文件实现 PaperPlan（论文蓝图）的构建逻辑。
from __future__ import annotations

from typing import Dict, List

from paper_kg.models import (
    PaperPlan,
    PaperSectionPlan,
    Pattern,
    RequiredPoint,
    StorySkeleton,
    Trick,
)
from paper_kg.trick_placement import place_tricks


def build_paper_plan(
    core_idea: str,
    domain: str,
    story_skeleton: StorySkeleton,
    patterns: List[Pattern],
    tricks: List[Trick],
    template: PaperPlan,
    paper_language: str = "zh",
) -> PaperPlan:
    """
    功能：基于故事框架与模板构建完整 PaperPlan。
    参数：core_idea、domain、story_skeleton、patterns、tricks、template。
    返回：PaperPlan。
    流程：规则映射 required_points + trick 落位 + 生成术语/主张/假设。
    说明：
    - 该函数不依赖 LLM，确保离线可用。
    - required_points 来自 story_skeleton 的映射规则。
    """
    trick_map = place_tricks(tricks)
    sections: List[PaperSectionPlan] = []
    for section in template.sections:
        required_points = _map_points(
            section.section_id,
            story_skeleton,
            core_idea,
            domain,
            paper_language,
        )
        required_point_items = _build_required_point_items(
            section.section_id,
            required_points,
        )
        required_tricks = trick_map.get(section.section_id, [])
        required_trick_names = _extract_trick_names(required_tricks)
        goal = _augment_goal_with_story_guidance(
            section_id=section.section_id,
            base_goal=section.goal,
            skeleton=story_skeleton,
            paper_language=paper_language,
        )
        sections.append(
            PaperSectionPlan(
                section_id=section.section_id,
                title=section.title,
                goal=goal,
                target_words_min=section.target_words_min,
                target_words_max=section.target_words_max,
                required_points=required_points,
                required_tricks=required_tricks,
                required_point_items=required_point_items,
                required_trick_names=required_trick_names,
                style_constraints=section.style_constraints,
                dependencies=section.dependencies,
            )
        )

    return PaperPlan(
        template_name=template.template_name,
        sections=sections,
        terms=_build_terms(core_idea, domain, patterns, tricks, paper_language),
        claims=_build_claims(core_idea, domain, paper_language),
        assumptions=_build_assumptions(paper_language),
    )


def _map_points(
    section_id: str,
    skeleton: StorySkeleton,
    core_idea: str,
    domain: str,
    paper_language: str,
) -> List[str]:
    """
    功能：把故事框架要点映射到对应章节的 required_points。
    参数：section_id、skeleton、core_idea、domain。
    返回：要点列表。
    流程：根据章节 id 选择对应的 skeleton 字段。
    说明：该映射规则是可解释的、可配置的默认策略。
    """
    language = (paper_language or "zh").lower()
    if language.startswith("en"):
        mapping: Dict[str, List[str]] = {
            "abstract": [
                f"Research problem: {core_idea}",
                "Method overview",
                "Key results (placeholder)",
            ],
            "introduction": ["Problem framing", "Research gap", "Contributions"],
            "related_work": [
                "Overview of related directions",
                "Positioning vs. prior work (use citation markers only)",
            ],
            "method": [
                "Method overview",
                "Algorithm details",
                "Theoretical analysis (assumptions/complexity)",
            ],
            "experiments": [
                "Experimental setup",
                "Baselines",
                "Main results",
                "Ablation and analysis",
            ],
            "conclusion": ["Summary of contributions", "Future work"],
            "limitations": ["Scope and limitations", "Failure modes"],
            "ethics": ["Ethical considerations and privacy", "Bias, safety, and compliance"],
        }
    else:
        mapping = {
            "abstract": [f"研究问题：{core_idea}", "方法概述", "实验结论占位"],
            "introduction": [skeleton.problem_framing, skeleton.gap_pattern],
            "related_work": ["相关方向概述", "与本文差异对比（允许 TODO 引用占位）"],
            "method": [skeleton.method_story],
            "experiments": [skeleton.experiments_story],
            "conclusion": ["贡献总结", "未来工作方向"],
            "limitations": ["适用范围与限制", "潜在失败模式"],
            "ethics": ["伦理风险与隐私说明", "潜在偏差与合规提示"],
        }
    points = mapping.get(section_id, [])
    return [p for p in points if str(p).strip()]


def _build_terms(
    core_idea: str,
    domain: str,
    patterns: List[Pattern],
    tricks: List[Trick],
    paper_language: str,
) -> Dict[str, str]:
    """
    功能：构建术语表（terms）。
    参数：core_idea、domain、patterns、tricks。
    返回：术语字典。
    流程：抽取核心术语 → 填充“待定义”占位。
    说明：避免空 terms 影响结构化输出稳定性。
    """
    language = (paper_language or "zh").lower()
    if language.startswith("en"):
        domain_label = _normalize_domain_for_english(domain)
        terms = {
            "Core idea": core_idea,
            "Domain": domain_label,
        }
        default_definition = "Definition to be clarified in Method/Introduction."
    else:
        terms = {
            "核心想法": core_idea,
            "领域": domain,
        }
        default_definition = "该术语的定义需在 Method/Introduction 中补充"
    for pattern in patterns[:3]:
        terms[pattern.pattern_name] = default_definition
    for trick in tricks[:2]:
        terms[trick.name] = (
            "Definition should be clarified in the corresponding section."
            if language.startswith("en")
            else "该技巧的定义需在对应章节中补充"
        )
    return terms


def _build_claims(core_idea: str, domain: str, paper_language: str) -> List[str]:
    """
    功能：构建核心主张列表。
    参数：core_idea、domain。
    返回：主张列表。
    流程：规则化模板生成。
    说明：用于帮助 Abstract/Conclusion 的结构完整性。
    """
    language = (paper_language or "zh").lower()
    if language.startswith("en"):
        domain_label = _normalize_domain_for_english(domain)
        return [
            f"We propose {core_idea} to improve {domain_label} tasks.",
            "Empirical results indicate stable improvements on standard benchmarks.",
        ]
    return [
        f"本文提出{core_idea}，用于提升{domain}任务的性能。",
        "实验结果显示该方法在标准基准上具有稳定优势。",
    ]


def _build_assumptions(paper_language: str) -> List[str]:
    """
    功能：构建默认假设/限制列表。
    参数：无。
    返回：假设列表。
    流程：返回通用占位。
    说明：可在后续章节中进一步展开或修改。
    """
    language = (paper_language or "zh").lower()
    if language.startswith("en"):
        return [
            "We assume publicly available and reproducible datasets/benchmarks.",
            "Experimental environments and hardware satisfy the described setup.",
        ]
    return [
        "假设可获得公开可复现的数据集与基准。",
        "实验环境与硬件配置满足论文描述要求。",
    ]


def _normalize_domain_for_english(domain: str) -> str:
    domain = (domain or "").strip()
    if not domain:
        return domain
    lower = domain.lower()
    if "自然语言" in domain or lower in {"nlp", "natural language processing"}:
        return "Natural Language Processing"
    if "计算机视觉" in domain or lower in {"cv", "computer vision"}:
        return "Computer Vision"
    if "语音" in domain or "speech" in lower:
        return "Speech Processing"
    if "推荐" in domain or "recommend" in lower:
        return "Recommender Systems"
    if "图" in domain or "graph" in lower:
        return "Graph Machine Learning"
    return domain


def _augment_goal_with_story_guidance(
    section_id: str,
    base_goal: str,
    skeleton: StorySkeleton,
    paper_language: str,
    max_chars: int = 1200,
) -> str:
    """
    功能：将 story_skeleton 的关键叙事注入到章节 goal 中（写作指导）。
    说明：避免把长段落塞进 required_points，导致字面匹配失败。
    """
    language = (paper_language or "zh").lower()
    prefix = "Story guidance:" if language.startswith("en") else "叙事指导："
    guidance_map: Dict[str, str] = {
        "abstract": "\n".join(
            [skeleton.problem_framing, skeleton.method_story, skeleton.experiments_story]
        ),
        "introduction": "\n".join([skeleton.problem_framing, skeleton.gap_pattern]),
        "method": skeleton.method_story,
        "experiments": skeleton.experiments_story,
    }
    guidance = (guidance_map.get(section_id, "") or "").strip()
    base_goal = (base_goal or "").strip()
    if not guidance:
        return base_goal
    combined = f"{base_goal}\n\n{prefix}\n{guidance}".strip()
    if max_chars > 0 and len(combined) > max_chars:
        combined = combined[:max_chars].rstrip() + "..."
    return combined


def _build_required_point_items(
    section_id: str,
    required_points: List[str],
) -> List[RequiredPoint]:
    items: List[RequiredPoint] = []
    for idx, text in enumerate(required_points or [], start=1):
        point_id = f"{section_id}_p{idx}"
        items.append(RequiredPoint(id=point_id, text=text))
    return items


def _extract_trick_names(required_tricks: List[str]) -> List[str]:
    names: List[str] = []
    for instruction in required_tricks or []:
        if not instruction:
            continue
        if "：" in instruction:
            name = instruction.split("：", 1)[0].strip()
        elif ":" in instruction:
            name = instruction.split(":", 1)[0].strip()
        else:
            name = instruction.strip()
        if name:
            names.append(name)
    return names
