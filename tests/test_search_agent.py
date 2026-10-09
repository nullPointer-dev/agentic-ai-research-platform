import pytest
from app.agents.search import SearchAgent
from app.models.research import ResearchState, SearchResult

class FakeSearchService:
    def __init__(self):
        self.received_query = None

    async def search(self, query: str) -> list[SearchResult]:
        self.received_query = query

        return [
            SearchResult(
                id="S1",
                title="Test Article",
                url="https://example.com/article",
                content="Test content",
                score=0.95,
                source_domain="example.com",
            )
        ]


@pytest.mark.anyio
async def test_search_agent():
    search_service = FakeSearchService()
    agent = SearchAgent(search_service)

    state = ResearchState(
        query="test query",
    )

    result = await agent.run(state)
    assert search_service.received_query == "test query"
    assert len(result["search_results"]) == 1
    assert result["search_results"][0].id == "S1"
    assert result["search_results"][0].title == "Test Article"
    assert result["search_results"][0].url == "https://example.com/article"
    assert result["search_results"][0].content == "Test content"
    assert result["search_results"][0].score == 0.95