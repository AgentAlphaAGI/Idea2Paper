# Title canonicalization tests.

from paper_kg.section_writer import _canonicalize_title


def test_canonicalize_title_case_insensitive() -> None:
    assert _canonicalize_title("Introduction") == _canonicalize_title("INTRODUCTION")


def test_canonicalize_title_punctuation() -> None:
    assert _canonicalize_title("Method: Overview") == _canonicalize_title("Method Overview")
