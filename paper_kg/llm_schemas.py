# 中文注释: 本文件集中定义论文链路使用的 LLM 结构化输出 Schema（Pydantic 模型）。
from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, ConfigDict, Field


class PaperParseOutput(BaseModel):
    """
    功能：用户自然语言输入的解析输出（抽取核心 idea 与领域）。
    参数：core_idea、domain。
    返回：PaperParseOutput。
    流程：字段校验 → 序列化。
    说明：
    - 该结构用于 LangChain `with_structured_output`，确保输出格式稳定。
    - domain 允许输出多个候选领域，上层会择一并做归一化。
    """

    model_config = ConfigDict(extra="ignore")

    core_idea: str = Field(default="", description="中文一句话核心创新/研究想法")
    domain: List[str] = Field(default_factory=list, description="领域候选列表，如 [\"NLP\",\"CV\"]")


class PaperReviewScores(BaseModel):
    """
    功能：论文审稿四维评分结构。
    参数：novelty、soundness、clarity、significance。
    返回：PaperReviewScores。
    流程：字段校验 → 序列化。
    说明：所有分值范围建议为 0-10。
    """

    model_config = ConfigDict(extra="ignore")

    novelty: float = 0.0
    soundness: float = 0.0
    clarity: float = 0.0
    significance: float = 0.0


class PaperReviewDefect(BaseModel):
    """
    功能：审稿过程中识别出的缺陷结构。
    参数：type/location/detail。
    返回：PaperReviewDefect。
    流程：字段校验 → 序列化。
    说明：
    - type 推荐使用可枚举缺陷类型：novelty_low/experiments_weak/theory_missing/clarity_low/significance_low。
    - location 推荐为 introduction/method/experiments/related_work/writing 等。
    """

    model_config = ConfigDict(extra="ignore")

    type: str = ""
    location: str = ""
    detail: str = ""


class PaperReviewerOutput(BaseModel):
    """
    功能：单个 Reviewer 的结构化输出 Schema。
    参数：overall_score、scores、strengths、weaknesses、defects、decision、issue_type。
    返回：PaperReviewerOutput。
    流程：字段校验 → 序列化。
    说明：
    - issue_type 用于反馈回退决策（PASS/CLARITY_ISSUE/SOUNDNESS_RISK/NOVELTY_LOW/SIGNIFICANCE_LOW）。
    - decision 为审稿结论字符串（accept/weak_accept/borderline/weak_reject/reject）。
    """

    model_config = ConfigDict(extra="ignore")

    overall_score: float = 0.0
    scores: PaperReviewScores = Field(default_factory=PaperReviewScores)
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    defects: List[PaperReviewDefect] = Field(default_factory=list)
    decision: str = ""
    issue_type: str = ""


PaperIssueType = Literal[
    "PASS",
    "CLARITY_ISSUE",
    "SOUNDNESS_RISK",
    "NOVELTY_LOW",
    "SIGNIFICANCE_LOW",
]

