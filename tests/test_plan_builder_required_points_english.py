# Ensure English plans generate English required_point_items and goal guidance.

import re

from paper_kg.models import StorySkeleton
from paper_kg.paper_plan_builder import build_paper_plan
from paper_kg.paper_template import load_paper_template


def test_required_points_english_and_goal_guidance() -> None:
    skeleton = StorySkeleton(
        problem_framing="We study a robust retrieval pipeline for repositories.",
        gap_pattern="Prior systems lack reliability under noisy inputs.",
        method_story="We propose a two-stage retrieval and verification approach.",
        experiments_story="We evaluate on multiple datasets with ablations.",
    )
    template = load_paper_template("paper_template_en.yaml")
    plan = build_paper_plan(
        core_idea="automated repository retrieval",
        domain="NLP",
        story_skeleton=skeleton,
        patterns=[],
        tricks=[],
        template=template,
        paper_language="en",
    )
    intro = next(section for section in plan.sections if section.section_id == "introduction")
    assert intro.required_point_items
    assert all(
        not re.search(r"[\u4e00-\u9fff]", item.text)
        for item in intro.required_point_items
    )
    assert skeleton.problem_framing in intro.goal
