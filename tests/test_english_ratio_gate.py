# English ratio gate tests.

from core.config import PaperWorkflowConfig
from paper_kg.models import PaperSectionOutput, PaperSectionPlan
from paper_kg.section_writer import PaperSectionWriter


def test_english_ratio_gate_flags_cjk() -> None:
    config = PaperWorkflowConfig(
        output_format="latex",
        latex_generation_mode="final_only",
        enforce_english=True,
        max_cjk_ratio=0.01,
        section_min_word_ratio=0.0,
    )
    writer = PaperSectionWriter(llm=None, config=config, pattern_guides="")
    section_plan = PaperSectionPlan(
        section_id="introduction",
        title="Introduction",
        goal="",
        target_words_min=10,
        target_words_max=50,
        required_points=[],
        required_tricks=[],
        style_constraints=[],
        dependencies=[],
    )
    output = PaperSectionOutput(
        section_id="introduction",
        title="Introduction",
        content_markdown="# Introduction\n\nThis is English. 这里有中文。",
        used_tricks=[],
        covered_points=[],
        todo=[],
        word_count=0,
        summary="",
    )

    issues = writer._validate_markdown_output(section_plan, output)
    assert any("non_english_ratio_high" in issue for issue in issues)
