# 中文注释: 本文件实现论文链路的输入解析（从自然语言提取 core_idea + domain）。
from __future__ import annotations

import logging
import re
from typing import Optional, Tuple

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from core.llm_utils import coerce_model
from paper_kg.llm_schemas import PaperParseOutput
from paper_kg.prompts import load_paper_prompt


logger = logging.getLogger(__name__)


class PaperInputParser:
    """
    功能：解析用户的一句话想法，抽取“核心 idea + 领域”。
    参数：llm（可选）。
    返回： (core_idea, domain_name)。
    流程：优先使用 LLM 结构化抽取；失败则用规则兜底。
    说明：
    - domain_name 只在 NLP/CV 两者中归一化（不做 Router/多任务判断）。
    - 若 LLM 输出的 domain 不在候选中，默认回退到 NLP。
    """

    def __init__(self, llm: Optional[BaseChatModel] = None) -> None:
        """
        功能：初始化解析器。
        参数：llm。
        返回：无。
        流程：保存 llm，并在存在时构建 structured_llm + prompt 模板。
        说明：llm=None 时仅启用规则兜底。
        """
        self.llm = llm
        self.prompt = None
        if llm is None:
            return

        system_prompt = load_paper_prompt("extract_idea_system.txt")
        self.structured_llm = llm.with_structured_output(PaperParseOutput, method="json_mode")
        self.prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=system_prompt),
                ("human", "用户输入：{text}\n请输出 JSON。"),
            ]
        )

    def parse(self, text: str) -> Tuple[str, str]:
        """
        功能：解析用户输入并返回 (core_idea, domain_name)。
        参数：text。
        返回：(core_idea, domain_name)。
        流程：LLM 抽取 → 归一化 domain → 异常回退规则。
        说明：domain_name 为 \"NLP\" 或 \"CV\"。
        """
        cleaned = str(text or "").strip()
        if not cleaned:
            return "", "NLP"

        if self.llm is not None and self.prompt is not None:
            try:
                logger.info("PAPER_PARSE_LLM_START")
                response = self.structured_llm.invoke(self.prompt.format_messages(text=cleaned))
                output = coerce_model(response, PaperParseOutput)
                core_idea = str(output.core_idea or "").strip() or cleaned
                domain = _normalize_domain(output.domain)
                logger.info("PAPER_PARSE_LLM_OK: domain=%s", domain)
                return core_idea, domain
            except Exception as exc:  # noqa: BLE001
                logger.warning("PAPER_PARSE_LLM_FAIL: fallback to rules: %s", exc)

        logger.info("PAPER_PARSE_RULE_FALLBACK")
        return _fallback_parse(cleaned)


def _normalize_domain(domains: list[str]) -> str:
    """
    功能：把 LLM 输出的领域候选归一化为 NLP/CV。
    参数：domains。
    返回：归一化后的领域名（NLP/CV）。
    流程：遍历候选 → 命中映射 → 回退 NLP。
    说明：允许出现 \"LLM\"/\"Vision\"/\"图像\" 等同义词。
    """
    for item in domains or []:
        name = str(item or "").strip().upper()
        if not name:
            continue
        if name in {"NLP", "LLM", "TEXT", "LANGUAGE"}:
            return "NLP"
        if name in {"CV", "VISION", "IMAGE", "VIDEO"}:
            return "CV"
        if "图像" in item or "视觉" in item:
            return "CV"
        if "语言" in item or "文本" in item or "大模型" in item:
            return "NLP"
    return "NLP"


def _fallback_parse(text: str) -> Tuple[str, str]:
    """
    功能：在无 LLM 或解析失败时的规则兜底解析。
    参数：text。
    返回：(core_idea, domain_name)。
    流程：根据关键词推断 domain，并尽量截取更像“idea”的子串。
    说明：该策略为原型保证可用性，不追求精确。
    """
    lowered = text.lower()
    domain = "NLP"
    if any(k in lowered for k in ["cv", "vision", "image", "video", "图像", "视觉", "检测", "分割"]):
        domain = "CV"

    # 中文注释：尝试从“我想/我们提出/目标是”等句式中截取核心想法。
    match = re.search(r"(?:我想|我们|本文|目标是|希望)(.*)$", text)
    core = match.group(1).strip() if match else text.strip()
    core = re.sub(r"^[，,:：\\s]+", "", core).strip()
    return core or text.strip(), domain
