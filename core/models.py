# 中文注释: 本文件包含可运行代码与流程说明。
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class IntentResult:
    industry_id: str
    industry_name: str
    intent_label: str
    constraints: Dict[str, Optional[str]]


@dataclass
class CreativePattern:
    id: str
    name: str
    template: str
    tags: List[str]
    applicable_industries: List[str]
    example_copy: str


@dataclass
class AdCase:
    id: str
    industry: str
    title: str
    copy: str
    strategy_tags: List[str]
    created_at: str


@dataclass
class RetrievedAd:
    ad: AdCase
    score: float


@dataclass
class NoveltyItem:
    ad_id: str
    score: float
    reason: str


@dataclass
class NoveltyReport:
    is_novel: bool
    max_similarity: float
    top_k: List[NoveltyItem]
    warning: Optional[str] = None


@dataclass
class AdDraft:
    headline: str
    body: str
    strategy_rationale: str
    audience: str
    channels: List[str]
    risk_notes: List[str]


@dataclass
class ReviewScores:
    creativity: float
    clarity: float
    strategy_fit: float
    compliance_risk: float
    conversion_potential: float
    total: float

    @classmethod
    def from_components(
        cls,
        creativity: float,
        clarity: float,
        strategy_fit: float,
        compliance_risk: float,
        conversion_potential: float,
    ) -> "ReviewScores":
        """
        功能：from_components 的核心流程封装，负责处理输入并输出结果。
        参数：cls、creativity、clarity、strategy_fit、compliance_risk、conversion_potential。
        返回：依据实现返回结果或产生副作用。
        流程：接收输入 → 核心逻辑处理 → 输出或触发副作用。
        说明：如遇异常或边界情况，调用侧需关注返回值或日志信息。
        """
        total = (
            creativity
            + clarity
            + strategy_fit
            + compliance_risk
            + conversion_potential
        ) / 5.0
        return cls(
            creativity=creativity,
            clarity=clarity,
            strategy_fit=strategy_fit,
            compliance_risk=compliance_risk,
            conversion_potential=conversion_potential,
            total=total,
        )


@dataclass
class ReviewResult:
    reviewer_id: str
    persona: str
    scores: ReviewScores
    comment: str
    issue_type: str


@dataclass
class WorkflowReport:
    status: str
    intent: IntentResult
    creative_pattern: CreativePattern
    retrieved_ads: List[RetrievedAd]
    novelty_report: NoveltyReport
    ad_draft: AdDraft
    reviews: List[ReviewResult]
    metrics: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """
        功能：to_dict 的核心流程封装，负责处理输入并输出结果。
        参数：无。
        返回：依据实现返回结果或产生副作用。
        流程：接收输入 → 核心逻辑处理 → 输出或触发副作用。
        说明：如遇异常或边界情况，调用侧需关注返回值或日志信息。
        """
        return asdict(self)
