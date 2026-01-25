import json
import re
import time
import warnings
from typing import Dict, Any, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# 抑制 urllib3 的 OpenSSL 警告
warnings.filterwarnings("ignore", category=UserWarning, module='urllib3')

from .config import LLM_API_KEY, LLM_API_URL, LLM_MODEL

def _create_session_with_retries():
    """创建带有重试机制的 requests Session"""
    session = requests.Session()

    # 定义重试策略
    retry_strategy = Retry(
        total=3,  # 总共重试 3 次
        backoff_factor=2,  # 指数退避: 1s, 2s, 4s
        status_forcelist=[429, 500, 502, 503, 504],  # 在这些状态码上重试
        allowed_methods=["POST", "GET"]
    )

    # 为 HTTP 和 HTTPS 适配器应用重试策略
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    return session

def call_llm(prompt: str, temperature: float = 0.7, max_tokens: int = 2000, timeout: int = 120) -> str:
    """
    调用 LLM API（支持重试和延长超时）

    Args:
        prompt: 提示文本
        temperature: 温度参数
        max_tokens: 最大 token 数
        timeout: 请求超时时间（秒），默认 120s
    """
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

    max_retries = 3
    retry_delay = 2

    for attempt in range(max_retries):
        try:
            session = _create_session_with_retries()

            if attempt > 0:
                print(f"   ⏳ 重试 LLM 调用 (尝试 {attempt + 1}/{max_retries})...")
                time.sleep(retry_delay)

            response = session.post(
                LLM_API_URL,
                headers=headers,
                json=data,
                timeout=timeout
            )
            response.raise_for_status()
            session.close()
            return response.json()["choices"][0]["message"]["content"]

        except requests.exceptions.Timeout as e:
            print(f"   ⚠️  超时异常 (尝试 {attempt + 1}/{max_retries}): 读取超时")
            if attempt < max_retries - 1:
                print(f"   ⏳ {retry_delay} 秒后重试...")
            else:
                print(f"❌ LLM 调用失败（超时）: {e}")
                return ""

        except requests.exceptions.ConnectionError as e:
            print(f"   ⚠️  连接异常 (尝试 {attempt + 1}/{max_retries}): {str(e)[:60]}")
            if attempt < max_retries - 1:
                print(f"   ⏳ {retry_delay} 秒后重试...")
            else:
                print(f"❌ LLM 调用失败（连接错误）: {e}")
                return ""

        except Exception as e:
            print(f"❌ LLM 调用失败: {e}")
            return ""

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

