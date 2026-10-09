import pytest

from app.agents.summarize import SummarizeAgent
from app.llm import LLMService
from app.models.research import ResearchState, SearchResult


@pytest.mark.integration
@pytest.mark.anyio
async def test_summarize_agent_with_llm():
    agent = SummarizeAgent(LLMService())

    state = ResearchState(
        query="What is CRISPR?",
        search_results=[
            SearchResult(
                id="S1",
                title="CRISPR Overview",
                url="https://example.com/crispr",
                content=(
                    "CRISPR-Cas9 is a genome editing technology "
                    "derived from a bacterial adaptive immune system."
                ),
                score=0.95,
                source_domain="example.com",
            ),
            SearchResult(
                id="S2",
                title="CRISPR Mechanism",
                url="https://example.com/mechanism",
                content=(
                    "Cas9 uses guide RNA to target a specific DNA "
                    "sequence for modification."
                ),
                score=0.90,
                source_domain="example.com",
            ),
        ],
    )

    result = await agent.run(state)

    summary = result["summary"]

    assert summary.text
    assert len(summary.text) > 0

    assert summary.source_ids
    assert set(summary.source_ids).issubset({"S1", "S2"})