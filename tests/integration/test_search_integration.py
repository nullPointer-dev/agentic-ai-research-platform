import pytest

from app.agents.search import SearchAgent
from app.models.research import ResearchState
from app.services.search import SearchService


@pytest.mark.integration
@pytest.mark.anyio
async def test_search_agent_with_tavily():
    agent = SearchAgent(SearchService())

    state = ResearchState(
        query="What is CRISPR?"
    )

    result = await agent.run(state)

    assert "search_results" in result
    assert len(result["search_results"]) > 0

    for search_result in result["search_results"]:
        assert search_result.title
        assert search_result.url
        assert search_result.content
        assert search_result.source_domain