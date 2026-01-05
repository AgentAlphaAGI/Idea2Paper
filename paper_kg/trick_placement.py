# 中文注释: 本文件实现 Trick 到章节的规则落位与写作指令生成。
from __future__ import annotations

import re
from typing import Dict, List

from paper_kg.models import PaperSectionId, Trick


def place_tricks(tricks: List[Trick]) -> Dict[PaperSectionId, List[str]]:
    """
    功能：根据 Trick 属性与关键词将技巧落位到具体章节，并生成写作指令。
    参数：tricks（Trick 列表）。
    返回：章节 -> 写作指令列表的字典。
    流程：遍历 tricks → 规则分类 → 生成指令 → 汇总。
    说明：
    - 纯规则，不调用 LLM，保证离线可用。
    - 同一个 trick 可能落位到多个章节（如 toy example）。
    """
    placements: Dict[PaperSectionId, List[str]] = {
        "abstract": [],
        "introduction": [],
        "related_work": [],
        "method": [],
        "experiments": [],
        "conclusion": [],
        "limitations": [],
        "ethics": [],
    }

    for trick in tricks:
        text = _normalize_text(trick)
        primary = _infer_primary_section(text, trick.trick_type)
        instruction = _build_instruction(trick, primary, text)
        placements[primary].append(instruction)

        # 中文注释：toy example 同时适用于 Introduction/Method。
        if _contains(text, ["toy", "示例", "玩具"]):
            if instruction not in placements["introduction"]:
                placements["introduction"].append(instruction)
            if instruction not in placements["method"]:
                placements["method"].append(instruction)

    return placements


def _normalize_text(trick: Trick) -> str:
    """
    功能：把 Trick 的名称/描述/用途拼接成归一化文本。
    参数：trick。
    返回：小写文本。
    流程：拼接字段 → 小写。
    说明：用于关键词匹配。
    """
    parts = [trick.name or "", trick.description or "", trick.purpose or "", trick.trick_type or ""]
    return " ".join(parts).lower()


def _contains(text: str, keywords: List[str]) -> bool:
    """
    功能：判断文本是否包含关键词列表中的任意项。
    参数：text、keywords。
    返回：bool。
    流程：遍历关键词 → 子串匹配。
    说明：用于规则分配。
    """
    return any(k.lower() in text for k in keywords)


def _infer_primary_section(text: str, trick_type: str) -> PaperSectionId:
    """
    功能：根据 trick_type 与关键词推断主要章节。
    参数：text、trick_type。
    返回：PaperSectionId。
    流程：优先 trick_type → 关键词 → 默认 method。
    说明：按需求把实验/理论/写作类分别落位。
    """
    type_lower = (trick_type or "").lower()
    if "experiment" in type_lower:
        return "experiments"
    if "method" in type_lower:
        return "method"
    if "writing" in type_lower:
        return "introduction"

    if _contains(text, ["ablation", "消融", "对比", "sota", "baseline", "显著性", "对照", "评测"]):
        return "experiments"
    if _contains(text, ["理论", "证明", "复杂度", "收敛", "算法", "推导"]):
        return "method"
    if _contains(text, ["相关工作", "综述", "引用", "对比方法"]):
        return "related_work"
    if _contains(text, ["动机", "缺口", "问题定义", "贡献", "背景"]):
        return "introduction"
    return "method"


def _build_instruction(trick: Trick, section_id: PaperSectionId, text: str) -> str:
    """
    功能：根据章节与关键词生成具体写作指令。
    参数：trick、section_id、text。
    返回：指令字符串。
    流程：关键词匹配 → 模板拼接。
    说明：指令以短句为主，便于 LLM 执行。
    """
    name = trick.name or "技巧"
    if section_id == "experiments":
        if _contains(text, ["ablation", "消融"]):
            return f"{name}：在 Experiments 加入消融小节，提供表格占位并总结消融结论。"
        if _contains(text, ["sota", "baseline", "对比"]):
            return f"{name}：在 Experiments 加入与 SOTA/基线对比的主表与结论段。"
        return f"{name}：在 Experiments 描述实验设置与验证结果。"
    if section_id == "method":
        if _contains(text, ["理论", "证明", "收敛", "复杂度"]):
            return f"{name}：在 Method 增加理论/复杂度分析小节并解释推导逻辑。"
        if _contains(text, ["算法", "流程", "结构"]):
            return f"{name}：在 Method 详细描述算法流程并给出步骤说明。"
        return f"{name}：在 Method 明确使用该技巧的实现方式。"
    if section_id == "introduction":
        if _contains(text, ["动机", "缺口", "问题定义"]):
            return f"{name}：在 Introduction 强化问题动机与研究缺口。"
        return f"{name}：在 Introduction 提及该技巧的背景与贡献。"
    if section_id == "related_work":
        return f"{name}：在 Related Work 对比相关方法并说明差异。"
    if section_id == "abstract":
        return f"{name}：在 Abstract 简要提及该技巧的贡献。"
    if section_id == "conclusion":
        return f"{name}：在 Conclusion 总结该技巧的贡献与影响。"
    if section_id == "limitations":
        return f"{name}：在 Limitations 说明该技巧的局限性。"
    if section_id == "ethics":
        return f"{name}：在 Ethics 提示该技巧可能的风险或注意事项。"
    return f"{name}：在 {section_id} 中体现该技巧。"
