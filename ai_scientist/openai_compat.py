import json
import logging
import os
import re
from copy import deepcopy
from typing import Any

import openai

logger = logging.getLogger("ai-scientist")


def _env_flag(name: str) -> bool:
    value = os.environ.get(name, "").strip().lower()
    return value in {"1", "true", "yes", "y", "on"}


def parse_model_spec(model: str) -> tuple[str, str]:
    if model.startswith("ollama/"):
        return "ollama", model[len("ollama/") :]
    if "::" in model:
        provider, model_id = model.split("::", 1)
        return provider, model_id
    return "default", model


def normalize_provider_env_key(provider: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", provider).upper()


def get_provider_credentials(provider: str) -> tuple[str | None, str | None]:
    if provider in {"", "default"}:
        return (
            os.environ.get("OPENAI_BASE_URL"),
            os.environ.get("OPENAI_API_KEY"),
        )

    if provider == "ollama":
        base_url = os.environ.get(
            "OAIC_OLLAMA_BASE_URL", "http://localhost:11434/v1"
        )
        api_key = os.environ.get("OAIC_OLLAMA_API_KEY", os.environ.get("OLLAMA_API_KEY"))
        return base_url, api_key

    key = normalize_provider_env_key(provider)
    base_url = os.environ.get(f"OAIC_{key}_BASE_URL")
    api_key = os.environ.get(f"OAIC_{key}_API_KEY")
    return base_url, api_key


def make_openai_client(provider: str, max_retries: int) -> openai.OpenAI:
    base_url, api_key = get_provider_credentials(provider)
    if provider in {"", "default"}:
        if not api_key:
            raise ValueError("Missing OPENAI_API_KEY for default provider.")
        return openai.OpenAI(api_key=api_key, base_url=base_url, max_retries=max_retries)

    if provider == "ollama":
        return openai.OpenAI(
            api_key=api_key or "",
            base_url=base_url,
            max_retries=max_retries,
        )

    key = normalize_provider_env_key(provider)
    missing = []
    if not base_url:
        missing.append(f"OAIC_{key}_BASE_URL")
    if not api_key:
        missing.append(f"OAIC_{key}_API_KEY")
    if missing:
        raise ValueError(f"Missing {', '.join(missing)} for provider '{provider}'.")
    return openai.OpenAI(api_key=api_key, base_url=base_url, max_retries=max_retries)


def _strip_image_content(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stripped: list[dict[str, Any]] = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            new_content = [
                item
                for item in content
                if not (isinstance(item, dict) and item.get("type") == "image_url")
            ]
            stripped.append({**msg, "content": new_content})
        else:
            stripped.append(msg)
    return stripped


def _is_missing_user_role_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return (
        "do not contain elements with the role of user" in msg
        or "role of user" in msg
    )


def _ensure_user_role(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if any(msg.get("role") == "user" for msg in messages):
        return messages
    return [*messages, {"role": "user", "content": "OK"}]


def _append_json_instruction(
    messages: list[dict[str, Any]], json_schema: dict
) -> list[dict[str, Any]]:
    instruction = (
        "\nReturn ONLY valid JSON (no markdown) that matches this JSON schema: "
        + json.dumps(json_schema, ensure_ascii=True)
    )
    updated = deepcopy(messages)
    if not updated:
        updated.append({"role": "user", "content": instruction.strip()})
        return updated

    last = updated[-1]
    content = last.get("content")
    if isinstance(content, list):
        content.append({"type": "text", "text": instruction})
        last["content"] = content
    elif isinstance(content, str):
        last["content"] = content + instruction
    else:
        last["content"] = instruction.strip()
    updated[-1] = last
    return updated


def safe_chat_completions_create(
    client: openai.OpenAI,
    model: str,
    messages: list[dict[str, Any]],
    temperature: float | None = None,
    max_tokens: int | None = None,
    seed: int | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: dict[str, Any] | None = None,
    functions: list[dict[str, Any]] | None = None,
    function_call: dict[str, Any] | str | None = None,
    response_format: dict[str, Any] | None = None,
    reasoning_effort: str | None = None,
    max_completion_tokens: int | None = None,
    stop: list[str] | str | None = None,
    json_schema: dict | None = None,
    force_json: bool = False,
    **kwargs,
):
    disable_tools = _env_flag("OAIC_DISABLE_TOOLS")
    disable_vision = _env_flag("OAIC_DISABLE_VISION")
    debug = _env_flag("OAIC_DEBUG")
    if disable_tools and json_schema:
        force_json = True

    base_messages = deepcopy(messages)
    if disable_vision:
        base_messages = _strip_image_content(base_messages)

    base_kwargs: dict[str, Any] = {
        "model": model,
        "messages": base_messages,
    }
    if temperature is not None:
        base_kwargs["temperature"] = temperature
    if max_tokens is not None:
        base_kwargs["max_tokens"] = max_tokens
    if max_completion_tokens is not None:
        base_kwargs["max_completion_tokens"] = max_completion_tokens
    if seed is not None:
        base_kwargs["seed"] = seed
    if response_format is not None:
        base_kwargs["response_format"] = response_format
    if reasoning_effort is not None:
        base_kwargs["reasoning_effort"] = reasoning_effort
    if stop is not None:
        base_kwargs["stop"] = stop
    if tools is not None and not disable_tools:
        base_kwargs["tools"] = tools
        if tool_choice is not None:
            base_kwargs["tool_choice"] = tool_choice
    if functions is not None and not disable_tools:
        base_kwargs["functions"] = functions
        if function_call is not None:
            base_kwargs["function_call"] = function_call
    base_kwargs.update(kwargs)

    attempts: list[tuple[str, list[dict[str, Any]], dict[str, Any]]] = []
    current_kwargs = dict(base_kwargs)
    if not force_json:
        attempts.append(("base", base_messages, current_kwargs))

    if "seed" in current_kwargs:
        current_kwargs = {k: v for k, v in current_kwargs.items() if k != "seed"}
        if not force_json:
            attempts.append(("drop seed", base_messages, current_kwargs))
    if "reasoning_effort" in current_kwargs:
        current_kwargs = {
            k: v for k, v in current_kwargs.items() if k != "reasoning_effort"
        }
        if not force_json:
            attempts.append(("drop reasoning_effort", base_messages, current_kwargs))
    if "max_completion_tokens" in current_kwargs and "max_tokens" not in current_kwargs:
        swapped = dict(current_kwargs)
        swapped["max_tokens"] = swapped.pop("max_completion_tokens")
        if not force_json:
            attempts.append(
                ("swap max_completion_tokens->max_tokens", base_messages, swapped)
            )
        current_kwargs = swapped
    elif "max_tokens" in current_kwargs and "max_completion_tokens" not in current_kwargs:
        swapped = dict(current_kwargs)
        swapped["max_completion_tokens"] = swapped.pop("max_tokens")
        if not force_json:
            attempts.append(
                ("swap max_tokens->max_completion_tokens", base_messages, swapped)
            )
        current_kwargs = swapped
    if "response_format" in current_kwargs:
        current_kwargs = {
            k: v for k, v in current_kwargs.items() if k != "response_format"
        }
        if not force_json:
            attempts.append(("drop response_format", base_messages, current_kwargs))
    if "stop" in current_kwargs and current_kwargs.get("stop") is None:
        current_kwargs = {k: v for k, v in current_kwargs.items() if k != "stop"}
        if not force_json:
            attempts.append(("drop stop None", base_messages, current_kwargs))

    if (disable_tools and json_schema) or (force_json and json_schema):
        json_messages = _append_json_instruction(base_messages, json_schema)
        no_tools_kwargs = {
            k: v
            for k, v in current_kwargs.items()
            if k not in {"tools", "tool_choice", "functions", "function_call"}
        }
        attempts.append(("json-only", json_messages, no_tools_kwargs))
    elif not disable_tools and tools and json_schema:
        functions_payload = [tool.get("function") for tool in tools]
        tool_choice_dict = tool_choice or {}
        function_call_payload = None
        if isinstance(tool_choice_dict, dict):
            func = tool_choice_dict.get("function", {})
            if isinstance(func, dict) and "name" in func:
                function_call_payload = {"name": func["name"]}
        fn_kwargs = dict(current_kwargs)
        fn_kwargs.pop("tools", None)
        fn_kwargs.pop("tool_choice", None)
        fn_kwargs["functions"] = functions_payload
        if function_call_payload is not None:
            fn_kwargs["function_call"] = function_call_payload
        attempts.append(("tools->functions", base_messages, fn_kwargs))

        json_messages = _append_json_instruction(base_messages, json_schema)
        json_kwargs = {
            k: v
            for k, v in current_kwargs.items()
            if k not in {"tools", "tool_choice", "functions", "function_call"}
        }
        attempts.append(("json-fallback", json_messages, json_kwargs))

    has_images = any(
        isinstance(msg.get("content"), list)
        and any(
            isinstance(item, dict) and item.get("type") == "image_url"
            for item in msg.get("content", [])
        )
        for msg in base_messages
    )
    if has_images and not disable_vision:
        for label, msg_list, kwargs_dict in list(attempts):
            stripped = _strip_image_content(msg_list)
            attempts.append((label + " (no-vision)", stripped, dict(kwargs_dict)))

    last_exc: Exception | None = None
    require_user_role = False
    for label, msg_list, kwargs_dict in attempts:
        try:
            kwargs_dict = dict(kwargs_dict)
            kwargs_dict["messages"] = (
                _ensure_user_role(msg_list) if require_user_role else msg_list
            )
            return client.chat.completions.create(**kwargs_dict)
        except Exception as exc:  # noqa: BLE001
            if not require_user_role and _is_missing_user_role_error(exc):
                require_user_role = True
                if debug:
                    logger.warning(
                        "OAIC user-role retry (%s): %s", label, str(exc)
                    )
                try:
                    retry_kwargs = dict(kwargs_dict)
                    retry_kwargs["messages"] = _ensure_user_role(msg_list)
                    return client.chat.completions.create(**retry_kwargs)
                except Exception as retry_exc:  # noqa: BLE001
                    last_exc = retry_exc
                    if debug:
                        logger.warning(
                            "OAIC retry after user-role (%s): %s",
                            label,
                            str(retry_exc),
                        )
                    continue
            last_exc = exc
            if debug:
                logger.warning("OAIC retry (%s): %s", label, exc)
            continue
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("No attempts were made for chat completion.")
