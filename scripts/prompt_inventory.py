#!/usr/bin/env python3
import ast
import json
import re
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
AI_ROOT = ROOT / "ai_scientist"
OUT_PATH = AI_ROOT / "prompts" / "_inventory.json"

KEYWORDS = ("prompt", "system", "template", "message", "desc")
DICT_KEYS = ("Introduction", "Instructions", "Response format", "IMPORTANT")


def run_rg(pattern: str) -> list[dict[str, Any]]:
    try:
        result = subprocess.run(
            ["rg", "-n", "-g", "*.py", pattern, str(AI_ROOT)],
            check=False,
            text=True,
            capture_output=True,
        )
    except FileNotFoundError:
        return []

    usage_sites: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        parts = line.split(":", 2)
        if len(parts) < 3:
            continue
        path, lineno, snippet = parts[0], parts[1], parts[2].strip()
        usage_sites.append(
            {"file": path, "line": int(lineno), "snippet": snippet}
        )
    return usage_sites


def extract_placeholders(text: str) -> list[str]:
    placeholders = re.findall(r"{([^{}]+)}", text)
    return [p for p in placeholders if p]


class PromptVisitor(ast.NodeVisitor):
    def __init__(self, filename: str, source: str):
        self.filename = filename
        self.source = source
        self.entries: list[dict[str, Any]] = []
        self.symbol_stack: list[str] = []

    def _current_symbol(self) -> str:
        return ".".join(self.symbol_stack) if self.symbol_stack else "module"

    def _make_id(self, lineno: int, name: str) -> str:
        rel = Path(self.filename).relative_to(ROOT)
        rel_id = str(rel).replace("/", ".").replace(".py", "")
        return f"{rel_id}.{name}.L{lineno}"

    def _record_text(self, lineno: int, name: str, text: str, kind: str) -> None:
        entry = {
            "id": self._make_id(lineno, name),
            "source_file": self.filename,
            "source_symbol": self._current_symbol(),
            "kind": kind,
            "placeholders": extract_placeholders(text),
            "usage_sites": [],
        }
        self.entries.append(entry)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        self.symbol_stack.append(node.name)
        self.generic_visit(node)
        self.symbol_stack.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        self.symbol_stack.append(node.name)
        self.generic_visit(node)
        self.symbol_stack.pop()

    def visit_Assign(self, node: ast.Assign) -> Any:
        target_names = []
        for tgt in node.targets:
            if isinstance(tgt, ast.Name):
                target_names.append(tgt.id)
        if not target_names:
            self.generic_visit(node)
            return

        joined = None
        if isinstance(node.value, ast.Constant) and isinstance(
            node.value.value, str
        ):
            joined = node.value.value
            kind = "template" if extract_placeholders(joined) else "text"
        elif isinstance(node.value, ast.JoinedStr):
            src = ast.get_source_segment(self.source, node.value) or ""
            joined = src
            kind = "template"
        elif isinstance(node.value, ast.Dict):
            kind = "prompt_type_json"
            for name in target_names:
                entry = {
                    "id": self._make_id(node.lineno, name),
                    "source_file": self.filename,
                    "source_symbol": self._current_symbol(),
                    "kind": kind,
                    "placeholders": [],
                    "usage_sites": [],
                }
                self.entries.append(entry)
            self.generic_visit(node)
            return
        else:
            self.generic_visit(node)
            return

        for name in target_names:
            if any(k in name.lower() for k in KEYWORDS):
                self._record_text(node.lineno, name, joined or "", kind)

        self.generic_visit(node)

    def visit_Dict(self, node: ast.Dict) -> Any:
        for key in node.keys:
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                if any(k in key.value for k in DICT_KEYS):
                    entry = {
                        "id": self._make_id(node.lineno, f"dict_{key.value}"),
                        "source_file": self.filename,
                        "source_symbol": self._current_symbol(),
                        "kind": "prompt_type_json",
                        "placeholders": [],
                        "usage_sites": [],
                    }
                    self.entries.append(entry)
        self.generic_visit(node)


def build_inventory() -> list[dict[str, Any]]:
    usage_sites = []
    usage_sites.extend(run_rg(r"get_response_from_llm\\("))
    usage_sites.extend(run_rg(r"query\\("))
    usage_sites.extend(run_rg(r"plan_and_code_query\\("))
    usage_sites.extend(run_rg(r"create_client\\("))
    usage_sites.extend(run_rg(r"system_message="))
    usage_sites.extend(run_rg(r"user_message="))
    usage_sites.extend(run_rg(r"prompt="))

    entries: list[dict[str, Any]] = []
    for path in AI_ROOT.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        visitor = PromptVisitor(str(path), source)
        visitor.visit(tree)
        entries.extend(visitor.entries)

    for entry in entries:
        entry["usage_sites"] = [
            site
            for site in usage_sites
            if site["file"] == entry["source_file"]
        ]
    return entries


def main() -> None:
    inventory = build_inventory()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    print(f"Wrote inventory to {OUT_PATH}")


if __name__ == "__main__":
    main()
