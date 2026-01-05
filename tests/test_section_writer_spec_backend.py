# Validate spec backend rendering inside SectionWriter.

from core.config import PaperWorkflowConfig
from paper_kg.models import PaperPlan, PaperSectionPlan
from paper_kg.section_writer import PaperSectionWriter


class _FakeStructuredLLM:
    def __init__(self, response):
        self.response = response

    def invoke(self, _messages):  # noqa: ANN001
        return self.response


class _FakeLLM:
    def __init__(self, structured_response):
        self.structured_response = structured_response

    def with_structured_output(self, *_args, **_kwargs):  # noqa: ANN001
        return _FakeStructuredLLM(self.structured_response)

    def invoke(self, _messages):  # noqa: ANN001
        class _Resp:
            def __init__(self, content):
                self.content = content

        return _Resp("{}")


def test_section_writer_spec_backend_renders_markdown() -> None:
    structured = {
        "section_id": "introduction",
        "title": "Introduction",
        "content_markdown": "",
        "content_spec": {
            "blocks": [
                {
                    "type": "paragraph",
                    "text": "We outline the research gap and motivation. [[CITE]]",
                }
            ]
        },
        "used_tricks": [],
        "covered_points": [],
        "todo": [],
        "word_count": 999,
        "summary": "",
    }
    llm = _FakeLLM(structured)

    config = PaperWorkflowConfig(
        output_format="latex",
        markdown_writer_backend="spec",
        spec_cite_token="[[CITE]]",
        spec_render_require_cite_ids=True,
        strict_markdown_validation=True,
        section_min_word_ratio=0.0,
    )
    writer = PaperSectionWriter(llm=llm, config=config, pattern_guides="")
    plan = PaperPlan(template_name="t", sections=[])
    section_plan = PaperSectionPlan(
        section_id="introduction",
        title="Introduction",
        goal="",
        target_words_min=10,
        target_words_max=200,
        required_points=[],
        required_tricks=[],
        style_constraints=[],
        dependencies=[],
    )

    output = writer.write_section(plan=plan, section_plan=section_plan, prior_summaries={})
    assert output.content_markdown.startswith("# Introduction")
    assert "<!-- CITE_CANDIDATE:introduction_001 -->" in output.content_markdown
    assert output.word_count != 999
