import pytest

from app.agents.write import WriteAgent
from app.models.research import Draft
from app.agents.search import SearchAgent
from app.models.research import VerificationResult
from app.graph.research import build_research_graph
from app.models.research import ResearchState, SearchResult, Summary


class FakeSearchAgent:
    async def run(self, state: ResearchState) -> dict:
        return {
            "search_results": [
                SearchResult(
                    id="S1",
                    title="Test Research",
                    url="https://example.com/research",
                    content="This is test research content.",
                    score=0.95,
                    source_domain="example.com",
                )
            ]
        }

class FakeSummarizeAgent:
    async def run(self, state: ResearchState) -> dict:
        return {
            "summary": Summary(
                text="This is a test summary.",
                source_ids=["S1"],
            )
        }

class FakeWriteAgent:
    async def run(self, state: ResearchState) -> dict:
        assert state.summary is not None

        return {
            "draft": Draft(
                text="CRISPR is a genome-editing technology. [S1]"
            )
        }

class FakeVerifyAgent:
    async def run(self, state: ResearchState) -> dict:
        assert state.draft is not None

        return {
            "verification": VerificationResult(
                passed=True,
                issues=[],
                feedback="",
            )
        }

@pytest.mark.anyio
async def test_research_graph():
    graph = build_research_graph(FakeSearchAgent(), FakeSummarizeAgent(), FakeWriteAgent(), FakeVerifyAgent())
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
    assert result["draft"].text == "CRISPR is a genome-editing technology. [S1]"
    assert result["verification"].passed is True
    assert result["verification"].issues == []
    assert result["verification"].feedback == ""