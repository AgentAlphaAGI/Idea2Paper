# 中文注释: 从 Markdown 中提取 CITE_CANDIDATE，并补齐稳定的候选 ID。
from __future__ import annotations

import re
from typing import List, Tuple

from paper_kg.citations.models import CitationCandidate


_MARKER_RE = re.compile(r"<!--\s*CITE_CANDIDATE(?::([A-Za-z0-9:_-]+))?\s*-->")
_LIST_ITEM_RE = re.compile(r"^\s*(?:[-*]|\d+\.)\s+")


def extract_cite_candidates_from_markdown(
    markdown: str,
    section_id: str,
    start_index: int = 1,
) -> Tuple[str, List[CitationCandidate], int]:
    """
    功能：提取单个章节 Markdown 内的引用候选点。
    参数：markdown/section_id/start_index。
    返回：更新后的 markdown、候选列表、下一个 index。
    说明：
    - 忽略 code fence 内的 marker。
    - 支持段落/列表项/行内 marker。
    - 若 marker 无 id，会补齐为 sec_{section_id}_{idx:03d}（稳定）。
    """
    if not markdown:
        return markdown, [], start_index

    lines = markdown.splitlines()
    existing_ids, next_index = _collect_existing_ids(lines, section_id, start_index)

    candidates: List[CitationCandidate] = []
    updated_lines: List[str] = []
    in_code = False

    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            updated_lines.append(line)
            continue
        if in_code or "<!-- CITE_CANDIDATE" not in line:
            updated_lines.append(line)
            continue

        new_line = line
        cursor = 0
        pieces: List[str] = []
        for match in _MARKER_RE.finditer(line):
            pieces.append(new_line[cursor:match.start()])
            raw_id = (match.group(1) or "").strip()
            if raw_id:
                candidate_id = raw_id
            else:
                candidate_id = _next_available_id(section_id, existing_ids, next_index)
                next_index += 1
                existing_ids.add(candidate_id)

            claim_text, context_text = _extract_claim_context_from_lines(
                lines, idx, match.start()
            )
            candidates.append(
                CitationCandidate(
                    candidate_id=candidate_id,
                    section_id=section_id,
                    claim_text=claim_text,
                    context_text=context_text,
                    keywords_suggested=[],
                )
            )
            pieces.append(f"<!-- CITE_CANDIDATE:{candidate_id} -->")
            cursor = match.end()

        pieces.append(new_line[cursor:])
        updated_lines.append("".join(pieces))

    updated = "\n".join(updated_lines)
    return updated, candidates, next_index


def extract_cite_candidates_from_sections(
    sections: List[Tuple[str, str]],
) -> Tuple[List[Tuple[str, str]], List[CitationCandidate]]:
    """
    功能：按章节批量提取候选点。
    参数：sections（(section_id, markdown)）。
    返回：更新后的 sections + candidates。
    """
    updated_sections: List[Tuple[str, str]] = []
    candidates: List[CitationCandidate] = []

    for section_id, markdown in sections:
        updated, items, _ = extract_cite_candidates_from_markdown(
            markdown, section_id, start_index=1
        )
        updated_sections.append((section_id, updated))
        candidates.extend(items)

    return updated_sections, candidates


def _collect_existing_ids(
    lines: List[str], section_id: str, start_index: int
) -> Tuple[set[str], int]:
    existing_ids: set[str] = set()
    next_index = start_index
    in_code = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        for match in _MARKER_RE.finditer(line):
            cand_id = (match.group(1) or "").strip()
            if not cand_id:
                continue
            existing_ids.add(cand_id)
            match_idx = _match_section_index(section_id, cand_id)
            if match_idx is not None:
                next_index = max(next_index, match_idx + 1)
    return existing_ids, next_index


def _match_section_index(section_id: str, cand_id: str) -> int | None:
    prefix = f"sec_{section_id}_"
    if not cand_id.startswith(prefix):
        return None
    suffix = cand_id[len(prefix) :]
    if suffix.isdigit():
        return int(suffix)
    return None


def _next_available_id(section_id: str, existing_ids: set[str], index: int) -> str:
    while True:
        candidate_id = f"sec_{section_id}_{index:03d}"
        if candidate_id not in existing_ids:
            return candidate_id
        index += 1


def _extract_claim_context_from_lines(
    lines: List[str], line_index: int, marker_pos: int
) -> Tuple[str, str]:
    block_start, block_end = _find_block_bounds(lines, line_index)
    block_lines = [_strip_list_marker(l) for l in lines[block_start : block_end + 1]]
    block_text = "\n".join(block_lines)

    prefix_lines = [_strip_list_marker(l) for l in lines[block_start:line_index]]
    current_prefix = _strip_list_marker(lines[line_index][:marker_pos])
    block_before_marker = "\n".join(prefix_lines + [current_prefix])

    block_text = _MARKER_RE.sub("", block_text).strip()
    block_before_marker = _MARKER_RE.sub("", block_before_marker).strip()

    sentences = _split_sentences(block_before_marker)
    claim = sentences[-1].strip() if sentences else block_before_marker
    return claim, block_text


def _find_block_bounds(lines: List[str], line_index: int) -> Tuple[int, int]:
    if _is_list_item(lines[line_index]):
        start = line_index
        end = line_index
        for j in range(line_index + 1, len(lines)):
            if not lines[j].strip():
                break
            if _is_list_item(lines[j]):
                break
            if lines[j].lstrip().startswith("```"):
                break
            if not _is_list_continuation(lines[j]):
                break
            end = j
        return start, end

    start = line_index
    while start > 0:
        prev = lines[start - 1]
        if not prev.strip() or _is_list_item(prev) or prev.lstrip().startswith("```"):
            break
        start -= 1

    end = line_index
    while end + 1 < len(lines):
        nxt = lines[end + 1]
        if not nxt.strip() or _is_list_item(nxt) or nxt.lstrip().startswith("```"):
            break
        end += 1
    return start, end


def _is_list_item(line: str) -> bool:
    return bool(_LIST_ITEM_RE.match(line))


def _is_list_continuation(line: str) -> bool:
    if not line.strip():
        return False
    if _is_list_item(line):
        return False
    return line.startswith(" ") or line.startswith("\t")


def _strip_list_marker(line: str) -> str:
    return _LIST_ITEM_RE.sub("", line)


def _split_sentences(text: str) -> List[str]:
    parts = re.split(r"([。！？.!?])", text)
    sentences: List[str] = []
    buf = ""
    for part in parts:
        if part in {"。", "！", "？", ".", "!", "?"}:
            buf += part
            sentences.append(buf)
            buf = ""
        else:
            buf += part
    if buf.strip():
        sentences.append(buf)
    return sentences
