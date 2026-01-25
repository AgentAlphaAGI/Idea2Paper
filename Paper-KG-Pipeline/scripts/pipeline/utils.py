import json
import re
import warnings
from typing import Dict, Any, Optional

import requests

# 抑制 urllib3 的 OpenSSL 警告
warnings.filterwarnings("ignore", category=UserWarning, module='urllib3')

from .config import LLM_API_KEY, LLM_API_URL, LLM_MODEL

def call_llm(prompt: str, temperature: float = 0.7, max_tokens: int = 2000) -> str:
    """调用 LLM API"""
    if not LLM_API_KEY:
        print("⚠️  警告: LLM_API_KEY 未配置，使用模拟输出")
        return f"[模拟LLM输出] Prompt: {prompt[:100]}..."

    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    try:
        response = requests.post(LLM_API_URL, headers=headers, json=data, timeout=60)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"❌ LLM 调用失败: {e}")
        return ""

def clean_json_text(text: str) -> str:
    """清理 JSON 文本中的 Markdown 标记和非法字符"""
    clean_text = text.strip()
    if clean_text.startswith("```json"):
        clean_text = clean_text[7:]
    if clean_text.startswith("```"):
        clean_text = clean_text[3:]
    if clean_text.endswith("```"):
        clean_text = clean_text[:-3]
    return clean_text.strip()

def parse_json_from_llm(response: str) -> Optional[Dict[str, Any]]:
    """从 LLM 响应中解析 JSON，包含自动修复逻辑"""
    try:
        # 1. 基础清理
        clean_response = clean_json_text(response)

        # 2. 提取 JSON 部分 (寻找最外层的 {})
        start = clean_response.find('{')
        end = clean_response.rfind('}') + 1

        if start >= 0 and end > start:
            json_str = clean_response[start:end]

            # 2.1 预处理：处理非法控制字符
            def replace_control_chars(match):
                s = match.group(0)
                return s.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')

            # 匹配双引号包裹的内容
            json_str = re.sub(r'"((?:[^"\\]|\\.)*)"', replace_control_chars, json_str, flags=re.DOTALL)

            # 2.2 尝试直接解析
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass

            # 2.3 尝试修复常见的 JSON 错误
            repaired = json_str
            # 移除尾部逗号
            repaired = re.sub(r',(\s*[}\]])', r'\1', repaired)
            # 修复字段间缺失逗号 (如 "val" "key")
            repaired = re.sub(r'("\s*)\n?\s*"', r'\1,\n"', repaired)
            # 修复结构间缺失逗号 (如 } "key" 或 ] "key")
            repaired = re.sub(r'(}|])\s*\n?\s*"', r'\1,\n"', repaired)

            try:
                return json.loads(repaired)
            except:
                pass

        return None

    except Exception as e:
        print(f"   ⚠️  JSON 解析工具内部错误: {e}")
        return None

def compute_jaccard_similarity(text1: str, text2: str) -> float:
    """计算文本相似度（Jaccard）"""
    tokens1 = set(text1.lower().split())
    tokens2 = set(text2.lower().split())

    if not tokens1 or not tokens2:
        return 0.0

    intersection = tokens1 & tokens2
    union = tokens1 | tokens2

    return len(intersection) / len(union)

