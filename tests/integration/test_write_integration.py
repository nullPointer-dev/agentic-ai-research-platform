import pytest

from app.agents.write import WriteAgent
from app.llm import LLMService
from app.models.research import ResearchState, Summary


@pytest.mark.integration
@pytest.mark.anyio
async def test_write_agent_with_real_llm():
    state = ResearchState(
        query="What is CRISPR-Cas9?",
        summary=Summary(
            text=(
                "CRISPR-Cas9 is a genome-editing system. "
                "A guide RNA directs Cas9 to a target DNA sequence, "
                "where Cas9 can cut the DNA."
            ),
            source_ids=["S1", "S2"],
        ),
    )

    agent = WriteAgent(LLMService())
    result = await agent.run(state)

    assert result["draft"].text.strip()
    assert len(result["draft"].text) > 100
    print("\nGenerated draft:\n", result["draft"].text)
