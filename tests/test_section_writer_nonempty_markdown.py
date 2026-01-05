# 中文注释: 验证章节写作会强制校验 content_markdown 非空。

import json

from core.config import PaperWorkflowConfig
from paper_kg.models import PaperPlan, PaperSectionPlan
from paper_kg.section_writer import PaperSectionWriter


class _FakeStructuredLLM:
    def __init__(self, response):
        self.response = response

    def invoke(self, _messages):  # noqa: ANN001
        return self.response


class _FakeLLM:
    def __init__(self, structured_response, raw_response):
        self.structured_response = structured_response
        self.raw_response = raw_response

    def with_structured_output(self, *_args, **_kwargs):  # noqa: ANN001
        return _FakeStructuredLLM(self.structured_response)

    def invoke(self, _messages):  # noqa: ANN001
        class _Resp:
            def __init__(self, content):
                self.content = content

        return _Resp(self.raw_response)


def test_section_writer_repairs_empty_markdown() -> None:
    structured = {
        "section_id": "introduction",
        "title": "Introduction",
        "content_markdown": "",
        "used_tricks": [],
        "covered_points": [],
        "todo": [],
        "word_count": 500,
        "summary": "",
    }
    repaired = {
        "section_id": "introduction",
        "title": "Introduction",
        "content_markdown": "# Introduction\n\nThis is valid content.",
        "used_tricks": [],
        "covered_points": [],
        "todo": [],
        "word_count": 10,
        "summary": "A short summary.",
    }
    llm = _FakeLLM(structured, json.dumps(repaired))

    config = PaperWorkflowConfig(
        output_format="latex",
        strict_markdown_validation=True,
        section_min_word_ratio=0.0,
        section_min_quality_max_retries=1,
    )
    writer = PaperSectionWriter(llm=llm, config=config, pattern_guides="")
    plan = PaperPlan(template_name="t", sections=[])
    section_plan = PaperSectionPlan(
        section_id="introduction",
        title="Introduction",
        goal="",
        target_words_min=10,
        target_words_max=20,
        required_points=[],
        required_tricks=[],
        style_constraints=[],
        dependencies=[],
    )

    output = writer.write_section(plan=plan, section_plan=section_plan, prior_summaries={})
    assert output.content_markdown.strip()
    assert output.word_count != 500
