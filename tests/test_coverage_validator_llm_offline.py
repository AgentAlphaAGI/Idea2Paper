# Offline test for LLM coverage validator integration.

from core.config import PaperWorkflowConfig
from paper_kg.models import PaperPlan, PaperSectionOutput, PaperSectionPlan, RequiredPoint
from paper_kg.section_writer import PaperSectionWriter
from paper_kg.coverage_validator import CoverageCheckResult


class _FakeStructured:
    def __init__(self, responses):
        self._responses = list(responses)
        self._idx = 0

    def invoke(self, _messages):  # noqa: ANN001
        if not self._responses:
            raise AssertionError("No fake responses configured")
        if self._idx >= len(self._responses):
            return self._responses[-1]
        response = self._responses[self._idx]
        self._idx += 1
        return response


class _FakeLLM:
    def __init__(self, section_outputs, coverage_results):
        self._section_outputs = _FakeStructured(section_outputs)
        self._coverage_results = _FakeStructured(coverage_results)

    def with_structured_output(self, model_cls, **_kwargs):  # noqa: ANN001
        name = getattr(model_cls, "__name__", "")
        if name == "PaperSectionOutput":
            return self._section_outputs
        if name == "CoverageCheckResult":
            return self._coverage_results
        return _FakeStructured([])

    def invoke(self, _messages):  # noqa: ANN001
        class _Resp:
            def __init__(self, content):
                self.content = content

        return _Resp("{}")


def test_coverage_validator_llm_rewrite() -> None:
    initial = {
        "section_id": "introduction",
        "title": "Introduction",
        "content_markdown": "# Introduction\n\nWe provide background only.",
        "used_tricks": [],
        "covered_points": [],
        "todo": [],
        "word_count": 20,
        "summary": "Short summary.",
    }
    rewritten = {
        "section_id": "introduction",
        "title": "Introduction",
        "content_markdown": "# Introduction\n\nWe address the research gap and provide contributions.",
        "used_tricks": ["Ablation Study"],
        "covered_points": ["introduction_p1"],
        "todo": [],
        "word_count": 30,
        "summary": "Updated summary.",
    }
    missing = CoverageCheckResult(
        ok=False,
        missing_point_ids=["introduction_p1"],
        missing_trick_names=["Ablation Study"],
        evidence={},
        notes="first",
    )
    passed = CoverageCheckResult(
        ok=True,
        missing_point_ids=[],
        missing_trick_names=[],
        evidence={
            "introduction_p1": "research gap",
            "Ablation Study": "Ablation Study",
        },
        notes="second",
    )
    llm = _FakeLLM([initial, rewritten], [missing, passed])

    config = PaperWorkflowConfig(
        output_format="latex",
        latex_generation_mode="final_only",
        format_only_mode=False,
        strict_markdown_validation=True,
        section_min_word_ratio=0.0,
        coverage_validator_backend="llm",
        coverage_validator_max_retries=2,
        coverage_validator_require_evidence=True,
    )
    writer = PaperSectionWriter(llm=llm, config=config, pattern_guides="")
    plan = PaperPlan(template_name="t", sections=[], terms={}, claims=[], assumptions=[])
    section_plan = PaperSectionPlan(
        section_id="introduction",
        title="Introduction",
        goal="",
        target_words_min=10,
        target_words_max=200,
        required_points=["Research gap"],
        required_tricks=["Ablation Study: add ablation discussion"],
        required_point_items=[RequiredPoint(id="introduction_p1", text="Research gap")],
        required_trick_names=["Ablation Study"],
        style_constraints=[],
        dependencies=[],
    )

    output = writer.write_section(plan=plan, section_plan=section_plan, prior_summaries={})
    assert "Fallback output" not in (output.content_markdown or "")
    assert "research gap" in output.content_markdown.lower()
    assert "Ablation Study" in output.used_tricks
