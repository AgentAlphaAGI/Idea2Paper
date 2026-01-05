# 中文注释: 本文件定义故事生成的状态结构。
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from paper_kg.models import Defect, Domain, Idea, Pattern, Trick


@dataclass
class StoryState:
    """
    功能：故事生成过程的状态容器。
    参数：idea/domain/patterns/tricks/defects/estimated_score。
    返回：StoryState。
    流程：保存当前构建状态。
    说明：用于动作的不可变迁移。
    """

    idea: Idea | None = None
    domain: Domain | None = None
    patterns: List[Pattern] = field(default_factory=list)
    tricks: List[Trick] = field(default_factory=list)
    defects: List[Defect] = field(default_factory=list)
    estimated_score: float = 0.0

    def to_dict(self) -> dict:
        """
        功能：将状态转换为字典。
        参数：无。
        返回：字典结构。
        流程：提取关键字段。
        说明：用于日志与调试。
        """
        return {
            "idea": self.idea.id if self.idea else None,
            "domain": self.domain.id if self.domain else None,
            "patterns": [pattern.id for pattern in self.patterns],
            "tricks": [trick.id for trick in self.tricks],
            "defects": [defect.defect_id for defect in self.defects],
            "score": self.estimated_score,
        }
