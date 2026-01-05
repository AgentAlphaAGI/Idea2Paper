# 中文注释: 本文件定义论文 LangGraph 工作流的最终报告结构（用于 CLI 输出）。
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from paper_kg.models import NoveltyCheckResult, Pattern, StoryDraft, Trick


@dataclass
class PaperWorkflowReport:
    """
    功能：论文工作流最终输出报告。
    参数：status/core_idea/domain/patterns/tricks/novelty/draft/metrics/warnings。
    返回：PaperWorkflowReport。
    流程：作为纯数据载体被 controller/CLI 生成与序列化。
    说明：to_dict 会将 Pydantic 对象转换为可 JSON 序列化结构。
    """

    status: str
    core_idea: str
    domain: str
    patterns: List[Pattern] = field(default_factory=list)
    tricks: List[Trick] = field(default_factory=list)
    novelty_report: Optional[NoveltyCheckResult] = None
    draft: Optional[StoryDraft] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """
        功能：将报告转换为可 JSON 序列化的字典。
        参数：无。
        返回：dict。
        流程：逐字段转换 → Pydantic 使用 model_dump → 返回。
        说明：CLI 会直接 json.dumps 本结果。
        """
        return {
            "status": self.status,
            "core_idea": self.core_idea,
            "domain": self.domain,
            "patterns": [pattern.model_dump() for pattern in self.patterns],
            "tricks": [trick.model_dump() for trick in self.tricks],
            "novelty_report": self.novelty_report.model_dump() if self.novelty_report else None,
            "draft": self.draft.model_dump() if self.draft else None,
            "metrics": self.metrics,
            "warnings": list(self.warnings),
        }
