import pytest
from app.agents.summarize import build_evidence, SummarizeAgent
from app.models.research import SearchResult, ResearchState

class FakeLLMService:
    def __init__(self):
        self.received_prompt = None

    async def generate(self, prompt: str) -> str:
        self.received_prompt = prompt

        return """
        {
            "text": "CRISPR is a genome-editing technology.",
            "source_ids": ["S1", "S2"]
        }
        """

def test_build_evidence():
    results = [
        SearchResult(
            title="First Article",
            url="https://example.com/first",
            content="First article content.",
            score=0.95,
            source_domain="example.com",
        ),
        SearchResult(
            title="Second Article",
            url="https://example.com/second",
            content="Second article content.",
            score=0.87,
            source_domain="example.com",
        ),
    ]

    evidence = build_evidence(results)

    assert "[S1]" in evidence
    assert "[S2]" in evidence

    assert "Title: First Article" in evidence
    assert "Content:\nFirst article content." in evidence

    assert "Title: Second Article" in evidence
    assert "Content:\nSecond article content." in evidence

    assert "https://example.com/first" in evidence
    assert "https://example.com/second" in evidence

    assert "0.95" not in evidence
    assert "0.87" not in evidence

@pytest.mark.anyio
async def test_summarize_agent():
    llm_service = FakeLLMService()
    agent = SummarizeAgent(llm_service)

    state = ResearchState(
        query="What is CRISPR?",
        search_results=[
            SearchResult(
                title="First Article",
                url="https://example.com/first",
                content="CRISPR is a genome-editing technology.",
                score=0.95,
                source_domain="example.com",
            ),
            SearchResult(
                title="Second Article",
                url="https://example.com/second",
                content="CRISPR can target specific DNA sequences.",
                score=0.87,
                source_domain="example.com",
            ),
        ],
    )

    result = await agent.run(state)

    assert result["summary"].text == (
        "CRISPR is a genome-editing technology."
    )

    assert result["summary"].source_ids == ["S1", "S2"]

    assert "What is CRISPR?" in llm_service.received_prompt
    assert "[S1]" in llm_service.received_prompt
    assert "[S2]" in llm_service.received_prompt

    assert "First Article" in llm_service.received_prompt
    assert "Second Article" in llm_service.received_prompt

    assert "0.95" not in llm_service.received_prompt
    assert "0.87" not in llm_service.received_prompt