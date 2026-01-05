# 中文注释: 本文件包含可运行代码与流程说明。
from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from pydantic import BaseModel


def extract_json(text: str) -> Dict[str, Any]:
    """
    功能：从文本中提取 JSON 对象并解析为字典。
    参数：text。
    返回：解析后的字典。
    流程：正则定位 JSON → json.loads 解析。
    说明：找不到 JSON 或解析失败会抛出异常。
    """
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError("LLM output does not contain JSON object")
    return json.loads(match.group(0))


def safe_content(message: Any) -> str:
    """
    功能：从 LangChain 消息对象中提取文本内容。
    参数：message。
    返回：消息文本。
    流程：读取 content 属性 → 兜底转字符串。
    说明：兼容不同消息对象类型。
    """
    content = getattr(message, "content", None)
    if content is None:
        return str(message)
    return str(content)


def log_llm_output(logger, label: str, response: Any, max_chars: int = 4000) -> None:
    """
    功能：打印 LLM 输出到日志（可截断）。
    参数：logger/label/response/max_chars。
    返回：无。
    说明：仅用于调试；避免输出过长导致日志噪声。
    """
    if response is None:
        return
    text = safe_content(response)
    if max_chars and len(text) > max_chars:
        text = text[:max_chars].rstrip() + "...(truncated)"
    logger.info("LLM_OUTPUT: %s\n%s", label, text)


def ensure_list(value: Any) -> List[str]:
    """
    功能：将输入值统一为字符串列表。
    参数：value。
    返回：字符串列表。
    流程：判断类型 → 拆分或过滤 → 清洗。
    说明：用于兼容 LLM 返回字符串或数组。
    """
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        parts = re.split(r"[、,;|\n]+", value)
        return [part.strip() for part in parts if part.strip()]
    if value is None:
        return []
    return [str(value).strip()]


def ensure_dict(value: Any) -> Dict[str, Any]:
    """
    功能：将输入值统一为字典对象。
    参数：value。
    返回：字典。
    流程：判断类型 → 解析或回退为空字典。
    说明：字符串类型会尝试解析为 JSON。
    """
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return {}
    return {}


def require_text(data: Dict[str, Any], key: str) -> str:
    """
    功能：获取指定字段并确保为非空文本。
    参数：data、key。
    返回：清洗后的文本。
    流程：读取字段 → 转字符串 → 校验非空。
    说明：缺失或为空会抛出异常。
    """
    value = str(data.get(key, "")).strip()
    if not value:
        raise ValueError(f"Missing required field: {key}")
    return value


def coerce_float(value: Any, default: float = 0.0, low: float = 0.0, high: float = 10.0) -> float:
    """
    功能：将输入值转换为浮点数并限定范围。
    参数：value、default、low、high。
    返回：裁剪后的浮点数。
    流程：尝试 float 转换 → 范围裁剪 → 返回。
    说明：转换失败时返回 default。
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, number))


def coerce_model(response: Any, model_cls: type[BaseModel]) -> BaseModel:
    """
    功能：将 LLM 输出转换为指定的 Pydantic 模型实例。
    参数：response、model_cls。
    返回：模型实例。
    流程：判断类型 → model_validate 转换 → 返回。
    说明：支持 dict 或模型实例输入。
    """
    if isinstance(response, model_cls):
        return response
    return model_cls.model_validate(response)
