import pytest

from app.agents.search import SearchAgent
from app.graph.research import build_research_graph
from app.models.research import ResearchState, SearchResult, Summary


class FakeSearchAgent:
    async def run(self, state: ResearchState) -> ResearchState:
        state.search_results = [
            SearchResult(
                id="S1",
                title="Test Research",
                url="https://example.com/research",
                content="This is test research content.",
                score=0.95,
                source_domain="example.com",
            )
        ]
        return state

class FakeSummarizeAgent:
    async def run(self, state: ResearchState) -> dict:
        return {
            "summary": Summary(
                text="This is a test summary.",
                source_ids=["S1"],
            )
        }

@pytest.mark.anyio
async def test_research_graph():
    graph = build_research_graph(FakeSearchAgent(), FakeSummarizeAgent())
    initial_state = ResearchState(query="What is CRISPR?")
    result = await graph.ainvoke(initial_state)
    print(type(result))
    print(type(result["search_results"]))
    print(type(result["search_results"][0]))
    assert result["query"] == "What is CRISPR?"
    assert len(result["search_results"]) == 1
    assert result["search_results"][0].title == "Test Research"
    assert result["summary"].text == "This is a test summary."
    assert result["summary"].source_ids == ["S1"]