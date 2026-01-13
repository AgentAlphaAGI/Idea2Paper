import json
import re
from typing import Any
from ai_scientist.utils.token_tracker import track_token_usage

import backoff
import openai
from ai_scientist.openai_compat import (
    parse_model_spec,
    make_openai_client,
    safe_chat_completions_create,
)

MAX_NUM_TOKENS = 4096


def _model_id(model: str) -> str:
    _, model_id = parse_model_spec(model)
    return model_id


def _is_reasoning_model(model_id: str) -> bool:
    return model_id.startswith("o1") or model_id.startswith("o3")


def _build_messages(
    model_id: str,
    system_message: str | None,
    msg_history: list[dict[str, Any]],
    user_message: str | None,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if _is_reasoning_model(model_id):
        if system_message:
            messages.append({"role": "user", "content": system_message})
    else:
        if system_message:
            messages.append({"role": "system", "content": system_message})
    messages.extend(msg_history)
    if user_message is not None:
        messages.append({"role": "user", "content": user_message})
    return messages

AVAILABLE_LLMS = [
    "claude-3-5-sonnet-20240620",
    "claude-3-5-sonnet-20241022",
    # OpenAI models
    "gpt-4o-mini",
    "gpt-4o-mini-2024-07-18",
    "gpt-4o",
    "gpt-4o-2024-05-13",
    "gpt-4o-2024-08-06",
    "gpt-4.1",
    "gpt-4.1-2025-04-14",
    "gpt-4.1-mini",
    "gpt-4.1-mini-2025-04-14",
    "o1",
    "o1-2024-12-17",
    "o1-preview-2024-09-12",
    "o1-mini",
    "o1-mini-2024-09-12",
    "o3-mini",
    "o3-mini-2025-01-31",
    # DeepSeek Models
    "deepseek-coder-v2-0724",
    "deepcoder-14b",
    # Llama 3 models
    "llama3.1-405b",
    # Anthropic Claude models via Amazon Bedrock
    "bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
    "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
    "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0",
    "bedrock/anthropic.claude-3-haiku-20240307-v1:0",
    "bedrock/anthropic.claude-3-opus-20240229-v1:0",
    # Anthropic Claude models Vertex AI
    "vertex_ai/claude-3-opus@20240229",
    "vertex_ai/claude-3-5-sonnet@20240620",
    "vertex_ai/claude-3-5-sonnet@20241022",
    "vertex_ai/claude-3-sonnet@20240229",
    "vertex_ai/claude-3-haiku@20240307",
    # Google Gemini models
    "gemini-2.0-flash",
    "gemini-2.5-flash-preview-04-17",
    "gemini-2.5-pro-preview-03-25",
    # GPT-OSS models via Ollama
    "ollama/gpt-oss:20b",
    "ollama/gpt-oss:120b",
    # Qwen models via Ollama
    "ollama/qwen3:8b",
    "ollama/qwen3:32b",
    "ollama/qwen3:235b",

    "ollama/qwen2.5vl:8b",
    "ollama/qwen2.5vl:32b",

    "ollama/qwen3-coder:70b",
    "ollama/qwen3-coder:480b",

    # Deepseek models via Ollama
    "ollama/deepseek-r1:8b",
    "ollama/deepseek-r1:32b",
    "ollama/deepseek-r1:70b",
    "ollama/deepseek-r1:671b",
]


# Get N responses from a single message, used for ensembling.
@backoff.on_exception(
    backoff.expo,
    (
        openai.RateLimitError,
        openai.APITimeoutError,
        openai.InternalServerError,
    ),
)
@track_token_usage
def get_batch_responses_from_llm(
    prompt,
    client,
    model,
    system_message,
    print_debug=False,
    msg_history=None,
    temperature=0.7,
    n_responses=1,
) -> tuple[list[str], list[list[dict[str, Any]]]]:
    msg = prompt
    if msg_history is None:
        msg_history = []

    model_id = _model_id(model)
    messages = _build_messages(model_id, system_message, msg_history, msg)
    temp = 1 if _is_reasoning_model(model_id) else temperature
    max_tokens = None if _is_reasoning_model(model_id) else MAX_NUM_TOKENS
    response = safe_chat_completions_create(
        client,
        model=model_id,
        messages=messages,
        temperature=temp,
        max_tokens=max_tokens,
        seed=0,
        n=n_responses,
    )
    content = [r.message.content for r in response.choices]
    base_history = msg_history + [{"role": "user", "content": msg}]
    new_msg_history = [
        base_history + [{"role": "assistant", "content": c}] for c in content
    ]

    if print_debug:
        # Just print the first one.
        print()
        print("*" * 20 + " LLM START " + "*" * 20)
        for j, msg in enumerate(new_msg_history[0]):
            print(f'{j}, {msg["role"]}: {msg["content"]}')
        print(content)
        print("*" * 21 + " LLM END " + "*" * 21)
        print()

    return content, new_msg_history


@track_token_usage
def make_llm_call(client, model, temperature, system_message, prompt):
    model_id = _model_id(model)
    temp = 1 if _is_reasoning_model(model_id) else temperature
    max_tokens = None if _is_reasoning_model(model_id) else MAX_NUM_TOKENS
    messages = _build_messages(model_id, system_message, prompt, None)
    return safe_chat_completions_create(
        client,
        model=model_id,
        messages=messages,
        temperature=temp,
        max_tokens=max_tokens,
        seed=0,
        n=1,
    )


@backoff.on_exception(
    backoff.expo,
    (
        openai.RateLimitError,
        openai.APITimeoutError,
        openai.InternalServerError,
    ),
)
def get_response_from_llm(
    prompt,
    client,
    model,
    system_message,
    print_debug=False,
    msg_history=None,
    temperature=0.7,
) -> tuple[str, list[dict[str, Any]]]:
    msg = prompt
    if msg_history is None:
        msg_history = []

    model_id = _model_id(model)
    temp = 1 if _is_reasoning_model(model_id) else temperature
    max_tokens = None if _is_reasoning_model(model_id) else MAX_NUM_TOKENS
    messages = _build_messages(model_id, system_message, msg_history, msg)
    response = safe_chat_completions_create(
        client,
        model=model_id,
        messages=messages,
        temperature=temp,
        max_tokens=max_tokens,
        seed=0,
        n=1,
    )
    content = response.choices[0].message.content
    new_msg_history = msg_history + [{"role": "user", "content": msg}]
    new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]

    if print_debug:
        print()
        print("*" * 20 + " LLM START " + "*" * 20)
        for j, msg in enumerate(new_msg_history):
            print(f'{j}, {msg["role"]}: {msg["content"]}')
        print(content)
        print("*" * 21 + " LLM END " + "*" * 21)
        print()

    return content, new_msg_history


def extract_json_between_markers(llm_output: str) -> dict | None: 
    # Regular expression pattern to find JSON content between ```json and ```
    json_pattern = r"```json(.*?)```"
    matches = re.findall(json_pattern, llm_output, re.DOTALL)

    if not matches:
        # Fallback: Try to find any JSON-like content in the output
        json_pattern = r"\{.*?\}"
        matches = re.findall(json_pattern, llm_output, re.DOTALL)

    for json_string in matches:
        json_string = json_string.strip()
        try:
            parsed_json = json.loads(json_string)
            return parsed_json
        except json.JSONDecodeError:
            # Attempt to fix common JSON issues
            try:
                # Remove invalid control characters
                json_string_clean = re.sub(r"[\x00-\x1F\x7F]", "", json_string)
                parsed_json = json.loads(json_string_clean)
                return parsed_json
            except json.JSONDecodeError:
                continue  # Try next match

    return None  # No valid JSON found


def create_client(model) -> tuple[Any, str]:
    provider, model_id = parse_model_spec(model)
    client = make_openai_client(provider, max_retries=2)
    return client, model_id
