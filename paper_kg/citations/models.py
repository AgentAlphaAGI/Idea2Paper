# 中文注释: 引用链路的数据模型（候选/需求/检索结果/状态）。
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

NeedType = Literal["HARD", "SOFT", "NONE"]
RetrieveStatus = Literal["FOUND", "NOT_FOUND"]


class CitationCandidate(BaseModel):
    """引用候选点。"""

    model_config = ConfigDict(extra="ignore")

    candidate_id: str
    section_id: str
    claim_text: str
    context_text: str
    keywords_suggested: List[str] = Field(default_factory=list)


class CitationNeed(BaseModel):
    """引用必要性判断结果。"""

    model_config = ConfigDict(extra="ignore")

    candidate_id: str
    need_type: NeedType
    keywords: List[str] = Field(default_factory=list)
    year_range: str = ""


class RetrievedCitation(BaseModel):
    """引用检索结果。"""

    model_config = ConfigDict(extra="ignore")

    candidate_id: str
    status: RetrieveStatus
    paper_id: Optional[str] = None
    bibtex: str = ""
    bibkey: str = ""
    doi: Optional[str] = None
    arxiv: Optional[str] = None
    url: Optional[str] = None
    title: str = ""
    authors: List[str] = Field(default_factory=list)
    year: Optional[int] = None
    venue: str = ""
    evidence: List[str] = Field(default_factory=list)


class CitationState(BaseModel):
    """引用流程状态（便于调试）。"""

    model_config = ConfigDict(extra="ignore")

    candidates: List[CitationCandidate] = Field(default_factory=list)
    needs: List[CitationNeed] = Field(default_factory=list)
    retrieved: List[RetrievedCitation] = Field(default_factory=list)
    rounds_used: int = 0
    unresolved: List[str] = Field(default_factory=list)
