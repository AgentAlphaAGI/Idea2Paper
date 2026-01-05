# Ensure markdown quality issues use quality rewrite path.

from core.config import PaperWorkflowConfig
from paper_kg.models import PaperPlan, PaperSectionPlan
from paper_kg.section_writer import PaperSectionWriter


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
    def __init__(self, responses):
        self._structured = _FakeStructured(responses)
        self.raw_invoke_count = 0

    def with_structured_output(self, *_args, **_kwargs):  # noqa: ANN001
        return self._structured

    def invoke(self, _messages):  # noqa: ANN001
        self.raw_invoke_count += 1

        class _Resp:
            def __init__(self):
                self.content = "{}"

        return _Resp()


def test_quality_rewrite_path_skips_structured_repair() -> None:
    initial = {
        "section_id": "introduction",
        "title": "Introduction",
        "content_markdown": "",
        "content_spec": {
            "blocks": [
                {"type": "paragraph", "text": "Too short. [[CITE]]"}
            ]
        },
        "used_tricks": [],
        "covered_points": [],
        "todo": [],
        "word_count": 5,
        "summary": "Short summary.",
    }
    rewritten = {
        "section_id": "introduction",
        "title": "Introduction",
        "content_markdown": "",
        "content_spec": {
            "blocks": [
                {
                    "type": "paragraph",
                    "text": (
                        "We expand the background and motivation to satisfy the length "
                        "requirement for this section. The discussion clarifies the "
                        "problem setting, highlights the research gap, and motivates "
                        "the proposed direction. It also previews the core contributions "
                        "and evaluation plan. [[CITE]]"
                    ),
                }
            ]
        },
        "used_tricks": [],
        "covered_points": [],
        "todo": [],
        "word_count": 200,
        "summary": "Expanded summary.",
    }
    llm = _FakeLLM([initial, rewritten])

    config = PaperWorkflowConfig(
        output_format="latex",
        latex_generation_mode="final_only",
        markdown_writer_backend="spec",
        spec_cite_token="[[CITE]]",
        spec_render_require_cite_ids=True,
        strict_markdown_validation=True,
        section_min_word_ratio=1.0,
        section_min_quality_max_retries=1,
        section_max_retries=1,
        format_only_mode=True,
    )
    writer = PaperSectionWriter(llm=llm, config=config, pattern_guides="")
    plan = PaperPlan(template_name="t", sections=[])
    section_plan = PaperSectionPlan(
        section_id="introduction",
        title="Introduction",
        goal="",
        target_words_min=20,
        target_words_max=200,
        required_points=[],
        required_tricks=[],
        style_constraints=[],
        dependencies=[],
    )

    output = writer.write_section(plan=plan, section_plan=section_plan, prior_summaries={})
    assert output.content_markdown.startswith("# Introduction")
    assert llm.raw_invoke_count == 0
