# Prompts

This directory stores all LLM/VLM prompts used by the project. Prompt files must be
treated as immutable content: do not strip, dedent, or reformat text when loading.
All files are UTF-8 encoded and should preserve exact whitespace and newlines.

Guidelines:
- Keep each prompt in a dedicated file (or a small module JSON file) referenced by `registry.json`.
- Use `ai_scientist.prompts.loader` to load prompt text/JSON.
- If you introduce placeholders, use Python `str.format` and keep braces consistent.
- Do not change prompt content unless you are intentionally revising model instructions.
