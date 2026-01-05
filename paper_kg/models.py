# 中文注释: 本文件定义知识图谱的核心数据模型与序列化结构。
from __future__ import annotations

from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

NodeID = str
NodeType = str
EdgeType = str
PaperSectionId = Literal[
    "abstract",
    "introduction",
    "related_work",
    "method",
    "experiments",
    "conclusion",
    "limitations",
    "ethics",
]


class Paper(BaseModel):
    """
    功能：Paper 节点模型。
    参数：id/title/conference。
    返回：Paper 节点结构。
    流程：字段校验 → 序列化。
    说明：字段命名与 knowledge_graph.json 完全一致。
    """

    model_config = ConfigDict(extra="ignore")

    id: NodeID
    title: str = ""
    conference: str = ""

    def to_dict(self) -> Dict:
        """
        功能：将 Paper 转为可序列化字典。
        参数：无。
        返回：字典结构。
        流程：model_dump → 返回。
        说明：用于 JSON 存储。
        """
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict) -> "Paper":
        """
        功能：从字典构建 Paper 实例。
        参数：data。
        返回：Paper。
        流程：model_validate → 返回。
        说明：用于读取 JSON。
        """
        return cls.model_validate(data)


class Idea(BaseModel):
    """
    功能：Idea 节点模型。
    参数：核心创新描述与输入输出结构。
    返回：Idea 节点。
    流程：字段校验 → 序列化。
    说明：字段命名与 knowledge_graph.json 一致。
    """

    model_config = ConfigDict(extra="ignore")

    id: NodeID
    core_idea: str = ""
    tech_stack: List[str] = Field(default_factory=list)
    input_type: str = ""
    output_type: str = ""

    def to_dict(self) -> Dict:
        """
        功能：将 Idea 转为字典。
        参数：无。
        返回：字典。
        流程：model_dump → 返回。
        说明：用于 JSON 输出。
        """
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict) -> "Idea":
        """
        功能：从字典构建 Idea 实例。
        参数：data。
        返回：Idea。
        流程：model_validate → 返回。
        说明：用于加载 JSON。
        """
        return cls.model_validate(data)


class Domain(BaseModel):
    """
    功能：Domain 节点模型。
    参数：name/paper_count。
    返回：Domain 节点。
    流程：字段校验 → 序列化。
    说明：字段命名与 knowledge_graph.json 一致。
    """

    model_config = ConfigDict(extra="ignore")

    id: NodeID
    name: str = ""
    paper_count: int = 0

    def to_dict(self) -> Dict:
        """
        功能：将 Domain 转为字典。
        参数：无。
        返回：字典。
        流程：model_dump → 返回。
        说明：用于 JSON 输出。
        """
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict) -> "Domain":
        """
        功能：从字典构建 Domain 实例。
        参数：data。
        返回：Domain。
        流程：model_validate → 返回。
        说明：用于加载 JSON。
        """
        return cls.model_validate(data)


class Review(BaseModel):
    """
    功能：Review 节点模型。
    参数：paper_id/reviewer/overall_score/confidence/strengths/weaknesses。
    返回：Review 节点。
    流程：字段校验 → 序列化。
    说明：字段命名与 knowledge_graph.json 一致。
    """

    model_config = ConfigDict(extra="ignore")

    id: NodeID
    paper_id: NodeID
    reviewer: str = ""
    overall_score: str = ""
    confidence: str = ""
    strengths: str = ""
    weaknesses: str = ""

    def to_dict(self) -> Dict:
        """
        功能：将 Review 转为字典。
        参数：无。
        返回：字典。
        流程：model_dump → 返回。
        说明：用于 JSON 输出。
        """
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict) -> "Review":
        """
        功能：从字典构建 Review 实例。
        参数：data。
        返回：Review。
        流程：model_validate → 返回。
        说明：用于加载 JSON。
        """
        return cls.model_validate(data)


class Skeleton(BaseModel):
    """
    功能：Skeleton 节点模型。
    参数：skeleton_type/content。
    返回：Skeleton 节点。
    流程：字段校验 → 序列化。
    说明：字段命名与 knowledge_graph.json 一致。
    """

    model_config = ConfigDict(extra="ignore")

    id: NodeID
    skeleton_type: str = ""
    content: str = ""

    def to_dict(self) -> Dict:
        """
        功能：将 Skeleton 转为字典。
        参数：无。
        返回：字典。
        流程：model_dump → 返回。
        说明：用于 JSON 输出。
        """
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict) -> "Skeleton":
        """
        功能：从字典构建 Skeleton 实例。
        参数：data。
        返回：Skeleton。
        流程：model_validate → 返回。
        说明：用于加载 JSON。
        """
        return cls.model_validate(data)


class Pattern(BaseModel):
    """
    功能：Pattern 节点模型。
    参数：pattern_name/pattern_summary/writing_guide 等。
    返回：Pattern 节点。
    流程：字段校验 → 序列化。
    说明：字段命名与 knowledge_graph.json 一致。
    """

    model_config = ConfigDict(extra="ignore")

    id: NodeID
    pattern_name: str = ""
    pattern_summary: str = ""
    writing_guide: str = ""
    cluster_size: int = 0
    coherence_score: float = 0.0
    skeleton_count: int = 0
    trick_count: int = 0

    def to_dict(self) -> Dict:
        """
        功能：将 Pattern 转为字典。
        参数：无。
        返回：字典。
        流程：model_dump → 返回。
        说明：用于 JSON 输出。
        """
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict) -> "Pattern":
        """
        功能：从字典构建 Pattern 实例。
        参数：data。
        返回：Pattern。
        流程：model_validate → 返回。
        说明：用于加载 JSON。
        """
        return cls.model_validate(data)


class Trick(BaseModel):
    """
    功能：Trick 节点模型。
    参数：name/trick_type/purpose/description。
    返回：Trick 节点。
    流程：字段校验 → 序列化。
    说明：字段命名与 knowledge_graph.json 一致。
    """

    model_config = ConfigDict(extra="ignore")

    id: NodeID
    name: str = ""
    trick_type: str = ""
    purpose: str = ""
    description: str = ""

    def to_dict(self) -> Dict:
        """
        功能：将 Trick 转为字典。
        参数：无。
        返回：字典。
        流程：model_dump → 返回。
        说明：用于 JSON 输出。
        """
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict) -> "Trick":
        """
        功能：从字典构建 Trick 实例。
        参数：data。
        返回：Trick。
        流程：model_validate → 返回。
        说明：用于加载 JSON。
        """
        return cls.model_validate(data)


class Defect(BaseModel):
    """
    功能：Defect 节点模型。
    参数：缺陷类型与频次。
    返回：Defect 节点。
    流程：字段校验 → 序列化。
    说明：该结构用于内部审稿回退逻辑，不依赖外部 KG。
    """

    model_config = ConfigDict(extra="ignore")

    defect_id: NodeID
    type: str
    description: str
    frequency: float = 0.0

    def to_dict(self) -> Dict:
        """
        功能：将 Defect 转为字典。
        参数：无。
        返回：字典。
        流程：model_dump → 返回。
        说明：用于 JSON 输出。
        """
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict) -> "Defect":
        """
        功能：从字典构建 Defect 实例。
        参数：data。
        返回：Defect。
        流程：model_validate → 返回。
        说明：用于加载 JSON。
        """
        return cls.model_validate(data)


class WorksWellInProps(BaseModel):
    """
    功能：Pattern-IN_DOMAIN 等边属性的可选结构。
    参数：effectiveness/confidence/sample_count。
    返回：边属性结构。
    流程：字段校验 → 序列化。
    说明：该结构仅供内部统计使用。
    """

    model_config = ConfigDict(extra="ignore")

    effectiveness: float = 0.0
    confidence: float = 0.0
    sample_count: int = 0


class ComposedOfProps(BaseModel):
    """
    功能：Pattern-CONTAINS_TRICK 边属性。
    参数：order。
    返回：边属性结构。
    流程：字段校验 → 序列化。
    说明：KG 默认没有 order，可选补充。
    """

    model_config = ConfigDict(extra="ignore")

    order: int = 0


class MutationProps(BaseModel):
    """
    功能：Pattern 变异边属性（预留）。
    参数：描述/成功率/难度/缺陷类型等。
    返回：边属性结构。
    流程：字段校验 → 序列化。
    说明：当前 KG 未提供，保留以兼容后续扩展。
    """

    model_config = ConfigDict(extra="ignore")

    description: str = ""
    success_rate: Optional[float] = None
    difficulty: Optional[str] = None
    defect_type: Optional[str] = None
    effectiveness: Optional[float] = None


class StorySkeleton(BaseModel):
    """
    功能：故事框架结构。
    参数：problem_framing/gap_pattern/method_story/experiments_story。
    返回：StorySkeleton。
    流程：字段校验 → 序列化。
    说明：用于故事初版结构化输出。
    """

    model_config = ConfigDict(extra="ignore")

    problem_framing: str
    gap_pattern: str
    method_story: str
    experiments_story: str


class RequiredPoint(BaseModel):
    """
    功能：章节必覆盖要点（带稳定 ID）。
    参数：id/text。
    返回：RequiredPoint。
    流程：字段校验 → 序列化。
    说明：用于语义覆盖验收，避免死板字面匹配。
    """

    model_config = ConfigDict(extra="ignore")

    id: str
    text: str = ""


class PaperSectionPlan(BaseModel):
    """
    功能：单个章节的写作计划结构（蓝图中的章节配置）。
    参数：section_id/title/goal/字数范围/必覆盖要点/必用技巧/风格约束/依赖。
    返回：PaperSectionPlan。
    流程：字段校验 → 序列化。
    说明：section_id 使用固定枚举，方便后续章节写作与回退定位。
    """

    model_config = ConfigDict(extra="ignore")

    section_id: PaperSectionId
    title: str = ""
    goal: str = ""
    target_words_min: int = 0
    target_words_max: int = 0
    required_points: List[str] = Field(default_factory=list)
    required_tricks: List[str] = Field(default_factory=list)
    required_point_items: List[RequiredPoint] = Field(default_factory=list)
    required_trick_names: List[str] = Field(default_factory=list)
    style_constraints: List[str] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list)


class PaperPlan(BaseModel):
    """
    功能：论文蓝图（Paper Blueprint）结构。
    参数：模板名/章节计划/术语表/核心主张/假设限制。
    返回：PaperPlan。
    流程：字段校验 → 序列化。
    说明：作为分章节写作的总输入结构。
    """

    model_config = ConfigDict(extra="ignore")

    template_name: str = ""
    sections: List[PaperSectionPlan] = Field(default_factory=list)
    terms: Dict[str, str] = Field(default_factory=dict)
    claims: List[str] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)


class MarkdownParagraph(BaseModel):
    """Markdown paragraph block."""

    model_config = ConfigDict(extra="ignore")

    type: Literal["paragraph"] = "paragraph"
    text: str = ""


class MarkdownHeading(BaseModel):
    """Markdown heading block (level 2/3)."""

    model_config = ConfigDict(extra="ignore")

    type: Literal["heading"] = "heading"
    level: int = 2
    text: str = ""


class MarkdownBullets(BaseModel):
    """Markdown bullet list block."""

    model_config = ConfigDict(extra="ignore")

    type: Literal["bullets"] = "bullets"
    items: List[str] = Field(default_factory=list)


class MarkdownEquation(BaseModel):
    """Markdown equation block (LaTeX content only, no $$)."""

    model_config = ConfigDict(extra="ignore")

    type: Literal["equation"] = "equation"
    latex: str = ""


class MarkdownCode(BaseModel):
    """Markdown code block."""

    model_config = ConfigDict(extra="ignore")

    type: Literal["code"] = "code"
    language: str = ""
    code: str = ""


MarkdownBlock = Annotated[
    Union[
        MarkdownParagraph,
        MarkdownHeading,
        MarkdownBullets,
        MarkdownEquation,
        MarkdownCode,
    ],
    Field(discriminator="type"),
]


class SectionContentSpec(BaseModel):
    """Structured content spec for deterministic Markdown rendering."""

    model_config = ConfigDict(extra="ignore")

    blocks: List[MarkdownBlock] = Field(default_factory=list)


class PaperSectionOutput(BaseModel):
    """
    功能：单个章节的写作输出结构。
    参数：section_id/title/内容/使用技巧/覆盖要点/todo/字数/摘要。
    返回：PaperSectionOutput。
    流程：字段校验 → 序列化。
    说明：summary 用于后续章节输入（摘要传递，避免上下文过长）。
    """

    model_config = ConfigDict(extra="ignore")

    section_id: PaperSectionId
    title: str = ""
    content_markdown: str = ""
    # 中文注释：结构化内容 spec（用于确定性渲染 Markdown）
    content_spec: Optional[SectionContentSpec] = None
    # 中文注释：LaTeX 正文（输出格式为 latex 时使用）
    content_latex: str = ""
    used_tricks: List[str] = Field(default_factory=list)
    covered_points: List[str] = Field(default_factory=list)
    todo: List[str] = Field(default_factory=list)
    word_count: int = 0
    summary: str = ""
    # 中文注释：LaTeX 检查/编译摘要（格式保证专用）
    latex_check: Dict[str, object] = Field(default_factory=dict)


class SimilarItem(BaseModel):
    """
    功能：新颖性检索返回的相似项。
    参数：item_id/score/metadata。
    返回：SimilarItem。
    流程：字段校验 → 序列化。
    说明：metadata 可用于降权策略。
    """

    model_config = ConfigDict(extra="ignore")

    item_id: str
    score: float
    metadata: Dict = Field(default_factory=dict)


class NoveltyCheckResult(BaseModel):
    """
    功能：新颖性检查结果结构。
    参数：是否新颖、相似度、相似项列表。
    返回：NoveltyCheckResult。
    流程：字段校验 → 序列化。
    说明：suggestion 使用中文。
    """

    model_config = ConfigDict(extra="ignore")

    is_novel: bool
    max_similarity: float
    top_similar: List[SimilarItem] = Field(default_factory=list)
    suggestion: Optional[str] = None


class StoryDraft(BaseModel):
    """
    功能：故事初版输出结构。
    参数：idea/domain/selected patterns/tricks/故事框架/新颖性/最终文案/论文蓝图/章节输出。
    返回：StoryDraft。
    流程：字段校验 → 序列化。
    说明：final_story 为中文整合稿。
    """

    model_config = ConfigDict(extra="ignore")

    idea: str
    domain: str
    patterns: List[Pattern] = Field(default_factory=list)
    tricks: List[Trick] = Field(default_factory=list)
    story_skeleton: StorySkeleton
    novelty: NoveltyCheckResult
    final_story: str
    paper_plan: Optional[PaperPlan] = None
    paper_sections: List[PaperSectionOutput] = Field(default_factory=list)
    paper_markdown: str = ""
    section_summaries: Dict[str, str] = Field(default_factory=dict)
    # 中文注释：整篇 LaTeX 输出（可选，主要用于展示/调试）
    paper_latex: str = ""
    # 中文注释：Overleaf 工程目录与 zip 路径
    overleaf_project_dir: str = ""
    overleaf_zip_path: str = ""
    # 中文注释：LaTeX 编译报告摘要
    latex_compile_report: Dict[str, object] = Field(default_factory=dict)
