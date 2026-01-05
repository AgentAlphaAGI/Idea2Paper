# 中文注释: 测试 LLM retrieval agent 的强校验逻辑。

from paper_kg.citations.models import CitationCandidate, CitationNeed
from paper_kg.citations.retrieval_agent import CitationRetrievalAgent
from paper_kg.citations.semantic_scholar_tool import PaperCandidate, PaperDetails


class _FakeLLM:
    def __init__(self, responses):
        self._responses = list(responses)

    def invoke(self, _messages):  # noqa: ANN001
        class _Resp:
            def __init__(self, content):
                self.content = content

        return _Resp(self._responses.pop(0))


class _FakeTool:
    def search(self, _query, limit=5, fields=None):  # noqa: ANN001
        return [
            PaperCandidate(
                paper_id="paper123",
                title="Test Paper",
                year=2020,
                venue="TestConf",
                authors=["Alice"],
                external_ids={"DOI": "10.1/abc"},
                url="https://example.com",
            )
        ]

    def fetch_details(self, paper_id, fields=None):  # noqa: ANN001
        return PaperDetails(
            paper_id=paper_id,
            title="Test Paper",
            year=2020,
            venue="TestConf",
            authors=["Alice"],
            external_ids={"DOI": "10.1/abc"},
            url="https://example.com",
            citation_styles={"bibtex": "@article{test, title={Test}}"},
        )


def test_retrieval_agent_rejects_llm_facts() -> None:
    llm = _FakeLLM(
        [
            '{"query":"神经机器翻译 注意力机制"}',
            '{"status":"FOUND","selected_paperId":"paper123","doi":"10.1/abc"}',
        ]
    )
    agent = CitationRetrievalAgent(llm=llm, tool=_FakeTool(), require_verifiable=True)
    candidate = CitationCandidate(
        candidate_id="sec_intro_001",
        section_id="introduction",
        claim_text="已有研究表明该方法有效。",
        context_text="已有研究表明该方法有效。",
        keywords_suggested=[],
    )
    need = CitationNeed(
        candidate_id="sec_intro_001",
        need_type="HARD",
        keywords=["神经机器翻译", "注意力机制"],
        year_range="2018-2026",
    )

    results = agent.retrieve([need], [candidate])
    assert results[0].status == "NOT_FOUND"
