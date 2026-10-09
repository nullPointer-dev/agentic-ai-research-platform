import pytest

from app.graph.research import research_graph
from app.models.research import (
    Draft,
    Summary,
    VerificationResult,
)

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    force = True,
)


@pytest.mark.integration
@pytest.mark.anyio
async def test_complete_research_pipeline():
    result = await research_graph.ainvoke(
        {
            "query": (
                "What is CRISPR-Cas9, how does it work, and what are its main applications?"
            )
        }
    )

    # Search
    assert len(result["search_results"]) > 0

    # Summarize
    assert isinstance(result["summary"], Summary)
    assert result["summary"].text.strip()
    assert result["summary"].source_ids

    # Write
    assert isinstance(result["draft"], Draft)
    assert result["draft"].text.strip()

    # Verify
    assert isinstance(
        result["verification"],
        VerificationResult,
    )
    assert result["verification"].passed, (
        f"Verification failed: {result['verification'].feedback}\n"
        f"Issues: {result['verification'].model_dump_json(indent=2)}"
    )