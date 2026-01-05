# 中文注释: 本文件负责加载与解析论文模板配置（Paper Template）。
from __future__ import annotations

from pathlib import Path
from typing import List

import yaml

from paper_kg.models import PaperPlan, PaperSectionPlan


def load_paper_template(path: str) -> PaperPlan:
    """
    功能：加载论文模板配置并生成 PaperPlan 骨架。
    参数：path（模板 YAML 路径）。
    返回：PaperPlan（仅包含章节框架与全局模板信息）。
    流程：读取 YAML → 解析 sections → 构建 PaperPlan。
    说明：
    - 本函数只生成“模板骨架”，不填充 required_points/required_tricks 等动态字段。
    - 若模板文件缺失或格式错误，会抛出异常，由上层处理。
    """
    template_path = Path(path)
    if not template_path.exists():
        raise FileNotFoundError(f"未找到论文模板文件: {template_path}")

    data = yaml.safe_load(template_path.read_text(encoding="utf-8")) or {}
    template_name = str(data.get("template_name", "")).strip() or "default_template"
    sections_data = data.get("sections", []) or []

    sections: List[PaperSectionPlan] = []
    for section in sections_data:
        # 中文注释：required_points/required_tricks 将在 plan_builder 中补齐。
        sections.append(
            PaperSectionPlan(
                section_id=section.get("section_id"),
                title=str(section.get("title", "")),
                goal=str(section.get("goal", "")),
                target_words_min=int(section.get("target_words_min", 0)),
                target_words_max=int(section.get("target_words_max", 0)),
                required_points=list(section.get("required_points", []) or []),
                required_tricks=list(section.get("required_tricks", []) or []),
                style_constraints=list(section.get("style_constraints", []) or []),
                dependencies=list(section.get("dependencies", []) or []),
            )
        )

    return PaperPlan(
        template_name=template_name,
        sections=sections,
        terms={},
        claims=[],
        assumptions=[],
    )
