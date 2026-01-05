# 中文注释: 本文件提供 LaTeX 章节的确定性修复与 LLM 修复能力。
from __future__ import annotations

import hashlib
import logging
import re
from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from core.llm_utils import safe_content
from paper_kg.prompts import load_paper_prompt


logger = logging.getLogger(__name__)


class LatexFixer:
    """
    功能：LaTeX 修复器（确定性修复 + LLM 修复）。
    参数：llm/force_todo_cite。
    返回：修复后的 LaTeX 正文。
    说明：用于保证可编译性，优先做确定性修复。
    """

    def __init__(self, llm: Optional[BaseChatModel], force_todo_cite: bool = True) -> None:
        self.llm = llm
        self.force_todo_cite = force_todo_cite
        self.fix_prompt: Optional[ChatPromptTemplate] = None

        if llm is not None:
            fix_system = load_paper_prompt("latex_fix_from_log_system.txt")
            self.fix_prompt = ChatPromptTemplate.from_messages(
                [
                    SystemMessage(content=fix_system),
                    ("human", "编译错误日志：{log_excerpt}\n待修复章节：{tex}"),
                ]
            )

    def deterministic_fix(self, text: str) -> str:
        """
        功能：确定性修复（无需 LLM）。
        参数：text。
        返回：修复后的 LaTeX 文本。
        流程：移除 Markdown 痕迹 → 转义特殊字符 → 规范化 cite key。
        说明：此步骤确保最基础的可编译性。
        """
        text = _strip_markdown(text)
        text = _escape_specials(text)
        if self.force_todo_cite:
            text = _normalize_cite_keys(text)
        return text

    def llm_fix_from_log(self, tex: str, log_excerpt: str) -> str:
        """
        功能：使用 LLM 根据编译日志修复 LaTeX 正文。
        参数：tex、log_excerpt。
        返回：修复后的 LaTeX 文本。
        说明：若无 LLM，直接返回原 tex。
        """
        if self.llm is None or self.fix_prompt is None:
            logger.info("LATEX_LLM_FIX_SKIP: reason=llm_or_prompt_missing")
            return tex

        logger.info("LATEX_LLM_FIX_START: log_excerpt=%s", (log_excerpt or "")[:300])
        response = self.llm.invoke(self.fix_prompt.format_messages(tex=tex, log_excerpt=log_excerpt))
        return safe_content(response)


def _strip_markdown(text: str) -> str:
    """
    功能：移除常见 Markdown 痕迹，避免影响 LaTeX 编译。
    参数：text。
    返回：清理后的文本。
    说明：该函数不追求完美，只做最常见的安全清理。
    """
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    # 移除行首标题与列表标记
    text = re.sub(r"^\\s*#.+$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\\s*-\\s+", "", text, flags=re.MULTILINE)
    return text


def _escape_specials(text: str) -> str:
    """
    功能：转义 LaTeX 常见特殊字符。
    参数：text。
    返回：转义后的文本。
    说明：仅对未转义字符做处理，避免重复转义。
    """
    for char, escaped in [
        ("_", r"\\_"),
        ("%", r"\\%"),
        ("&", r"\\&"),
        ("#", r"\\#"),
        ("$", r"\\$"),
    ]:
        pattern = re.compile(rf"(?<!\\){re.escape(char)}")
        text = pattern.sub(escaped, text)
    return text


def _normalize_cite_keys(text: str) -> str:
    """
    功能：强制引用 key 规范化为 TODO_ 前缀。
    参数：text。
    返回：规范化后的文本。
    说明：避免 bibkey 不存在导致编译失败。
    """
    cite_pattern = re.compile(r"\\cite[a-zA-Z]*\{([^}]*)\}")

    def _replace(match: re.Match) -> str:
        raw_keys = match.group(1)
        keys = [k.strip() for k in raw_keys.split(",") if k.strip()]
        fixed_keys = [_to_todo_key(k) for k in keys]
        return match.group(0).replace(raw_keys, ",".join(fixed_keys))

    return cite_pattern.sub(_replace, text)


def _to_todo_key(key: str) -> str:
    """
    功能：将任意 cite key 转为 TODO_ 前缀的安全 key。
    参数：key。
    返回：安全 key。
    说明：仅保留 ASCII 字符；为空则用 hash。
    """
    if key.startswith("TODO_"):
        suffix = _sanitize_key(key[len("TODO_") :])
        if suffix:
            return f"TODO_{suffix}"
    sanitized = _sanitize_key(key)
    if not sanitized:
        digest = hashlib.md5(key.encode("utf-8")).hexdigest()[:8]
        sanitized = digest
    return f"TODO_{sanitized}"


def _sanitize_key(key: str) -> str:
    """
    功能：清理 cite key 中的非法字符。
    参数：key。
    返回：仅包含 [A-Za-z0-9:_-] 的字符串。
    """
    return re.sub(r"[^A-Za-z0-9:_-]+", "", key)
