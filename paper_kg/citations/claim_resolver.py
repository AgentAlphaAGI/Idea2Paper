# 中文注释: 对无法检索的引用做合法降级/改写。
from __future__ import annotations

import re
from typing import Dict, List, Tuple

from paper_kg.citations.models import CitationCandidate


def resolve_claims_in_sections(
    sections: List[Tuple[str, str]],
    candidates: List[CitationCandidate],
    unresolved_ids: List[str],
) -> List[Tuple[str, str]]:
    """
    功能：对未找到引用的候选点做降级改写。
    参数：sections（(section_id, markdown)）、candidates、unresolved_ids。
    返回：更新后的 sections。
    说明：只改对应段落，且移除 CITE_CANDIDATE 标记。
    """
    unresolved = set(unresolved_ids)
    if not unresolved:
        return sections

    cand_map: Dict[str, CitationCandidate] = {c.candidate_id: c for c in candidates}
    updated_sections: List[Tuple[str, str]] = []

    for section_id, markdown in sections:
        text = markdown
        for cand_id in list(unresolved):
            cand = cand_map.get(cand_id)
            if cand is None or cand.section_id != section_id:
                continue
            text = _resolve_candidate_in_text(text, cand_id)
        updated_sections.append((section_id, text))
    return updated_sections


def _resolve_candidate_in_text(text: str, candidate_id: str) -> str:
    marker = f"<!-- CITE_CANDIDATE:{candidate_id} -->"
    pos = text.find(marker)
    if pos == -1:
        return text

    before = text[:pos]
    after = text[pos + len(marker) :]

    para_start = before.rfind("\n\n")
    if para_start == -1:
        para_start = 0
    else:
        para_start += 2

    paragraph = before[para_start:]
    new_paragraph = _weaken_paragraph(paragraph)
    return text[:para_start] + new_paragraph + after


def _weaken_paragraph(paragraph: str) -> str:
    paragraph = paragraph.rstrip()
    sentences = _split_sentences(paragraph)
    if not sentences:
        return paragraph
    sentences[-1] = _rewrite_sentence(sentences[-1])
    return "".join(sentences)


def _rewrite_sentence(sentence: str) -> str:
    rules = [
        (r"已有研究表明", "我们观察到"),
        (r"研究表明", "我们观察到"),
        (r"文献表明", "我们观察到"),
        (r"显著优于", "在我们的实验中表现更好"),
        (r"首次提出", "我们提出"),
        (r"公认", "广泛认为"),
    ]
    for pattern, replacement in rules:
        if re.search(pattern, sentence):
            return re.sub(pattern, replacement, sentence)

    # 默认加上主观表述，避免需要外部引用
    stripped = sentence.strip()
    if stripped:
        return f"我们认为，{stripped}"
    return sentence


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
