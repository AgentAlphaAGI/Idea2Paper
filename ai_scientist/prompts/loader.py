import json
import importlib.util
from pathlib import Path
from typing import Any
from rich import print

_TEXT_CACHE: dict[str, str] = {}
_JSON_CACHE: dict[str, Any] = {}
_PY_CACHE: dict[str, Any] = {}
_REGISTRY_CACHE: dict[str, dict[str, str]] | None = None


def prompt_root() -> Path:
    return Path(__file__).resolve().parent


def _warn_and_raise(rel_path: str, exc: Exception) -> None:
    print(f"[yellow]Warning:[/yellow] Failed to load prompt {rel_path}: {exc}")
    raise exc


def load_text(rel_path: str) -> str:
    if rel_path in _TEXT_CACHE:
        return _TEXT_CACHE[rel_path]
    path = prompt_root() / rel_path
    try:
        content = path.read_text(encoding="utf-8")
    except Exception as exc:
        _warn_and_raise(rel_path, exc)
    _TEXT_CACHE[rel_path] = content
    return content


def load_json(rel_path: str) -> Any:
    if rel_path in _JSON_CACHE:
        return _JSON_CACHE[rel_path]
    try:
        content = load_text(rel_path)
        value = json.loads(content)
    except Exception as exc:
        _warn_and_raise(rel_path, exc)
    _JSON_CACHE[rel_path] = value
    return value


def render_text(rel_path: str, **kwargs: Any) -> str:
    return load_text(rel_path).format_map(kwargs)


def load_py(rel_path: str) -> Any:
    if rel_path in _PY_CACHE:
        return _PY_CACHE[rel_path]
    path = prompt_root() / rel_path
    try:
        module_name = "ai_scientist.prompts." + rel_path.replace("/", ".").replace(
            ".py", ""
        )
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load module spec for {rel_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        value = getattr(module, "PROMPTS", None)
        if value is None:
            raise AttributeError(f"PROMPTS not found in {rel_path}")
    except Exception as exc:
        _warn_and_raise(rel_path, exc)
    _PY_CACHE[rel_path] = value
    return value


def deep_render(value: Any, **kwargs: Any) -> Any:
    if isinstance(value, str):
        return value.format_map(kwargs)
    if isinstance(value, list):
        return [deep_render(item, **kwargs) for item in value]
    if isinstance(value, dict):
        return {key: deep_render(val, **kwargs) for key, val in value.items()}
    return value


def _load_registry() -> dict[str, dict[str, str]]:
    global _REGISTRY_CACHE
    if _REGISTRY_CACHE is not None:
        return _REGISTRY_CACHE
    registry_path = prompt_root() / "registry.json"
    try:
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    except Exception as exc:
        _warn_and_raise("registry.json", exc)
    _REGISTRY_CACHE = registry
    return registry


def load_prompt(prompt_id: str) -> Any:
    registry = _load_registry()
    entry = registry.get(prompt_id)
    if entry is None:
        raise KeyError(f"Unknown prompt id: {prompt_id}")
    prompt_type = entry.get("type")
    rel_path = entry.get("path", "")
    if prompt_type == "text" or prompt_type == "template":
        return load_text(rel_path)
    if prompt_type == "json":
        return load_json(rel_path)
    if prompt_type == "py":
        return load_py(rel_path)
    raise ValueError(f"Unsupported prompt type: {prompt_type} for id {prompt_id}")


def render_prompt(prompt_id: str, **kwargs: Any) -> str:
    value = load_prompt(prompt_id)
    if not isinstance(value, str):
        raise TypeError(f"Prompt {prompt_id} is not a text/template prompt")
    return value.format_map(kwargs)
