import json
import logging
import time
import re

from .utils import FunctionSpec, OutputType, opt_messages_to_list, backoff_create
from funcy import notnone, once, select_values
import openai
from rich import print
from ai_scientist.openai_compat import (
    parse_model_spec,
    make_openai_client,
    safe_chat_completions_create,
)

logger = logging.getLogger("ai-scientist")


OPENAI_TIMEOUT_EXCEPTIONS = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)

def get_ai_client(model: str, max_retries=2) -> openai.OpenAI:
    provider, _ = parse_model_spec(model)
    return make_openai_client(provider, max_retries=max_retries)


def _extract_json_from_text(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for pattern in (r"```json(.*?)```", r"\{.*\}"):
        matches = re.findall(pattern, text, flags=re.DOTALL)
        for match in matches:
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                continue
    raise ValueError("Failed to parse JSON from model output.")


def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    raw_model = model_kwargs.get("model", "")
    _provider, model_id = parse_model_spec(raw_model)
    client = get_ai_client(raw_model, max_retries=0)
    filtered_kwargs: dict = select_values(notnone, model_kwargs)  # type: ignore

    messages = opt_messages_to_list(system_message, user_message)

    filtered_kwargs["model"] = model_id

    tools = None
    tool_choice = None
    if func_spec is not None:
        tools = [func_spec.as_openai_tool_dict]
        tool_choice = func_spec.openai_tool_choice_dict

    t0 = time.time()
    call_kwargs = dict(filtered_kwargs)
    call_kwargs.pop("model", None)
    completion = backoff_create(
        safe_chat_completions_create,
        OPENAI_TIMEOUT_EXCEPTIONS,
        client=client,
        model=model_id,
        messages=messages,
        tools=tools,
        tool_choice=tool_choice,
        json_schema=func_spec.json_schema if func_spec else None,
        **call_kwargs,
    )
    req_time = time.time() - t0

    choice = completion.choices[0]

    if func_spec is None:
        output = choice.message.content
    else:
        try:
            if choice.message.tool_calls:
                assert (
                    choice.message.tool_calls[0].function.name == func_spec.name
                ), "Function name mismatch"
                try:
                    print(f"[cyan]Raw func call response: {choice}[/cyan]")
                    output = json.loads(choice.message.tool_calls[0].function.arguments)
                except json.JSONDecodeError as e:
                    logger.error(
                        "Error decoding the function arguments: %s",
                        choice.message.tool_calls[0].function.arguments,
                    )
                    raise e
            elif getattr(choice.message, "function_call", None):
                func_call = choice.message.function_call
                if func_call.name != func_spec.name:
                    raise ValueError("Function name mismatch")
                output = json.loads(func_call.arguments)
            else:
                output = _extract_json_from_text(choice.message.content or "")
        except Exception:
            fallback = safe_chat_completions_create(
                client=client,
                model=model_id,
                messages=messages,
                tools=None,
                tool_choice=None,
                json_schema=func_spec.json_schema,
                force_json=True,
                **call_kwargs,
            )
            fallback_choice = fallback.choices[0]
            output = _extract_json_from_text(fallback_choice.message.content or "")

    usage = getattr(completion, "usage", None)
    in_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
    out_tokens = getattr(usage, "completion_tokens", 0) if usage else 0

    info = {
        "system_fingerprint": getattr(completion, "system_fingerprint", None),
        "model": getattr(completion, "model", None),
        "created": getattr(completion, "created", None),
    }

    return output, req_time, in_tokens, out_tokens, info
