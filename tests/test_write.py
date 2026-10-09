import pytest

from app.agents.write import WriteAgent
from app.models.research import ResearchState, Summary


class FakeLLMService:
    def __init__(self, response=None):
        self.received_prompt = None
        self.call_count = 0
        self.response = response or (
            "## CRISPR\n\n"
            "CRISPR-Cas9 is a genome-editing technology. [S1] "
            "It uses a guide RNA to direct the Cas9 enzyme to a "
            "specific DNA sequence. The enzyme cuts the target DNA, "
            "allowing researchers to modify genetic material. "
            "This technology is used in biomedical research and "
            "the study of gene function."
        )

    async def generate(self, prompt: str) -> str:
        self.received_prompt = prompt
        self.call_count += 1
        return self.response


@pytest.mark.anyio
async def test_write_agent():
    llm_service = FakeLLMService()
    agent = WriteAgent(llm_service)

    state = ResearchState(
        query="What is CRISPR?",
        summary=Summary(
            text="CRISPR-Cas9 is a genome-editing technology.",
            source_ids=["S1"],
        ),
    )

    result = await agent.run(state)

    assert result["draft"].text.startswith("## CRISPR")
    assert "[S1]" in result["draft"].text

    assert "What is CRISPR?" in llm_service.received_prompt
    assert "genome-editing technology" in llm_service.received_prompt
    assert "S1" in llm_service.received_prompt

    assert llm_service.call_count == 1

@pytest.mark.anyio
async def test_write_agent_requires_summary():
    agent = WriteAgent(FakeLLMService())
    state = ResearchState(query="What is CRISPR?")

    with pytest.raises(ValueError, match="without a summary"):
        await agent.run(state)

@pytest.mark.anyio
async def test_write_agent_rejects_safety_message():
    llm = FakeLLMService("User Safety: safe")
    agent = WriteAgent(llm)

    state = ResearchState(
        query="What is CRISPR?",
        summary=Summary(
            text="CRISPR-Cas9 is a genome-editing technology.",
            source_ids=["S1"],
        ),
    )

    with pytest.raises(ValueError, match="safety message"):
        await agent.run(state)