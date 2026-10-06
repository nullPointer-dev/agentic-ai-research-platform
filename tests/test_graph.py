import pytest

from app.agents.search import SearchAgent
from app.graph.research import build_research_graph
from app.models.research import ResearchState, SearchResult


class FakeSearchAgent:
    async def run(self, state: ResearchState) -> ResearchState:
        state.search_results = [
            SearchResult(
                title="Test Research",
                url="https://example.com/research",
                content="This is test research content.",
                score=0.95,
                source_domain="example.com",
            )
        ]
        return state


@pytest.mark.anyio
async def test_research_graph():
    graph = build_research_graph(FakeSearchAgent())
    initial_state = ResearchState(query="What is CRISPR?")
    result = await graph.ainvoke(initial_state)
    print(type(result))
    print(type(result["search_results"]))
    print(type(result["search_results"][0]))
    assert result["query"] == "What is CRISPR?"
    assert len(result["search_results"]) == 1
    assert result["search_results"][0].title == "Test Research"