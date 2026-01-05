# 中文注释: BibTeX 生成与 bibkey 规则（来自检索元数据）。
from __future__ import annotations

import hashlib
import re
from typing import List, Optional


def make_bibkey(doi: Optional[str], arxiv: Optional[str], title: str, year: Optional[int]) -> str:
    if doi:
        return sanitize_bibkey(doi.replace("/", "_"))
    if arxiv:
        return sanitize_bibkey(f"arxiv_{arxiv}")
    base = f"{title}_{year or ''}".strip() or "ref"
    digest = hashlib.md5(base.encode("utf-8")).hexdigest()[:8]
    return sanitize_bibkey(f"ref_{digest}")


def sanitize_bibkey(key: str) -> str:
    return re.sub(r"[^A-Za-z0-9:_-]+", "", key)


def build_bibtex(
    bibkey: str,
    title: str,
    authors: List[str],
    venue: str,
    year: Optional[int],
    doi: Optional[str],
    url: Optional[str],
) -> str:
    fields: List[str] = []
    entry_type = _infer_entry_type(venue)
    if title:
        fields.append(f"  title = {{{title}}}")
    if authors:
        fields.append(f"  author = {{{' and '.join(authors)}}}")
    if venue:
        venue_field = "booktitle" if entry_type == "inproceedings" else "journal"
        fields.append(f"  {venue_field} = {{{venue}}}")
    if year:
        fields.append(f"  year = {{{year}}}")
    if doi:
        fields.append(f"  doi = {{{doi}}}")
    if url:
        fields.append(f"  url = {{{url}}}")
    body = ",\n".join(fields)
    return f"@{entry_type}{{{bibkey},\n{body}\n}}\n"


def _infer_entry_type(venue: str) -> str:
    if not venue:
        return "article"
    lowered = venue.lower()
    if any(
        keyword in lowered
        for keyword in (
            "conference",
            "proceedings",
            "workshop",
            "symposium",
        )
    ):
        return "inproceedings"
    if re.search(
        r"\\b(acl|emnlp|naacl|icml|neurips|nips|iclr|aaai|ijcai|kdd|sigir|www|cvpr|eccv|iccv)\\b",
        lowered,
    ):
        return "inproceedings"
    return "article"
