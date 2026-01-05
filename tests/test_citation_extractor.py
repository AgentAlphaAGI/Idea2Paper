# 中文注释: 测试 CITE_CANDIDATE 提取与补 ID 逻辑。

from paper_kg.citations.extractor import extract_cite_candidates_from_markdown


def test_extract_cite_candidates_assigns_ids() -> None:
    markdown = (
        "# 引言\n\n我们提出一个方法。<!-- CITE_CANDIDATE -->\n\n"
        "```python\n# <!-- CITE_CANDIDATE -->\n```\n"
        "- 列表项说明。<!-- CITE_CANDIDATE -->\n\n"
        "另一段落。<!-- CITE_CANDIDATE:custom_1 -->"
    )
    updated, candidates, _ = extract_cite_candidates_from_markdown(markdown, "introduction", start_index=1)

    assert "<!-- CITE_CANDIDATE:sec_introduction_001 -->" in updated
    assert "<!-- CITE_CANDIDATE:sec_introduction_002 -->" in updated
    assert "<!-- CITE_CANDIDATE:custom_1 -->" in updated
    assert "```python\n# <!-- CITE_CANDIDATE -->\n```" in updated

    ids = [c.candidate_id for c in candidates]
    assert "sec_introduction_001" in ids
    assert "sec_introduction_002" in ids
    assert "custom_1" in ids
    assert any("列表项" in c.claim_text for c in candidates)

    updated_again, candidates_again, _ = extract_cite_candidates_from_markdown(
        updated, "introduction", start_index=1
    )
    assert [c.candidate_id for c in candidates_again] == [c.candidate_id for c in candidates]
