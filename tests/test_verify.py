import pytest

from app.agents.verify import VerifyAgent, build_verification_prompt
from app.models.research import (
    Draft,
    ResearchState,
    SearchResult,
    VerificationResult
)

class FakeLLMService:
    def __init__(self, response: str) -> None:
        self.response = response
        self.received_prompt = ""

    async def generate(self, prompt: str, response_format=None) -> str:
        self.received_prompt = prompt
        return self.response

def test_build_verification_prompt_includes_draft_and_evidence():
    state = ResearchState(
        query="What is CRISPR?",
        search_results=[
            SearchResult(
                id="S1",
                title="CRISPR overview",
                url="https://example.com/crispr",
                content="CRISPR-Cas9 can cut targeted DNA sequences.",
                score=0.95,
                source_domain="example.com",
            )
        ],
        draft=Draft(
            text="CRISPR-Cas9 can cut targeted DNA sequences. [S1]"
        ),
    )

    prompt = build_verification_prompt(state)

    assert "What is CRISPR?" in prompt
    assert state.draft.text in prompt
    assert "[S1]" in prompt
    assert "CRISPR-Cas9 can cut targeted DNA sequences." in prompt
    assert "unsupported" in prompt.lower()


def test_build_verification_prompt_requires_draft():
    state = ResearchState(query="What is CRISPR?")

    with pytest.raises(ValueError, match="without a draft"):
        build_verification_prompt(state)

@pytest.mark.anyio
async def test_verify_agent():
    llm = FakeLLMService(
        """
        {
            "passed": true,
            "issues": [],
            "feedback": ""
        }
        """
    )
    agent = VerifyAgent(llm)

    state = ResearchState(
        query="What is CRISPR?",
        search_results=[
            SearchResult(
                id="S1",
                title="CRISPR overview",
                url="https://example.com/crispr",
                content="CRISPR-Cas9 can cut targeted DNA sequences.",
                score=0.95,
                source_domain="example.com",
            )
        ],
        draft=Draft(
            text="CRISPR-Cas9 can cut targeted DNA sequences. [S1]"
        ),
    )

    result = await agent.run(state)

    assert isinstance(result["verification"], VerificationResult)
    assert result["verification"].passed is True
    assert result["verification"].issues == []
    assert "CRISPR overview" in llm.received_prompt


@pytest.mark.anyio
async def test_verify_agent_rejects_invalid_source_id():
    llm = FakeLLMService(
        """
        {
            "passed": false,
            "issues": [
                {
                    "category": "unsupported_claim",
                    "description": "The cited source is unavailable.",
                    "source_ids": ["S99"]
                }
            ],
            "feedback": "Use a valid source."
        }
        """
    )
    agent = VerifyAgent(llm)

    state = ResearchState(
        query="What is CRISPR?",
        search_results=[
            SearchResult(
                id="S1",
                title="CRISPR overview",
                url="https://example.com/crispr",
                content="CRISPR-Cas9 can cut targeted DNA sequences.",
                score=0.95,
                source_domain="example.com",
            )
        ],
        draft=Draft(text="A claim. [S1]"),
    )

    with pytest.raises(ValueError, match="invalid source ID"):
        await agent.run(state)
