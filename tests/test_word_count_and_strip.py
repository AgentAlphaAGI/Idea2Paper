# Word count and markdown strip tests.

from paper_kg.section_writer import _count_words, _strip_markdown_for_count


def test_word_count_english() -> None:
    assert _count_words("This is a test") == 4


def test_strip_markdown_for_count_removes_fence_and_markers() -> None:
    md = (
        "# Title\n\n"
        "This is a sentence. <!-- CITE_CANDIDATE:intro_001 -->\n\n"
        "```python\nprint('x')\n```\n"
    )
    stripped = _strip_markdown_for_count(md)
    assert "CITE_CANDIDATE" not in stripped
    assert "print('x')" not in stripped
    assert "Title" in stripped
    assert _count_words(stripped) > 1
