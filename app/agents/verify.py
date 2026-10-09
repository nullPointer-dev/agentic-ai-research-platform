import json
from app.llm import LLMService
from app.models.research import ResearchState, VerificationResult

class VerifyAgent:
    def __init__(self, llm_service: LLMService) -> None:
        self.llm_service = llm_service

    async def run(self, state: ResearchState) -> dict:
        prompt = build_verification_prompt(state)
        response = await self.llm_service.generate(
            prompt,
            response_format={"type": "json_object"},
        )

        data = json.loads(response)
        verification = VerificationResult.model_validate(data)

        valid_source_ids = {
            result.id for result in state.search_results
        }

        for issue in verification.issues:
            if not set(issue.source_ids).issubset(valid_source_ids):
                raise ValueError(
                    "Verification contains an invalid source ID"
                )

        return {"verification": verification}


def build_verification_prompt(state: ResearchState) -> str:
    if state.draft is None:
        raise ValueError("Cannot verify a report without a draft")

    evidence = "\n\n".join(
        f"[{result.id}]\n"
        f"Title: {result.title}\n"
        f"Source: {result.source_domain}\n"
        f"URL: {result.url}\n"
        f"Content:\n{result.content}"
        for result in state.search_results
    )

    return f"""
You are the verification stage of a research pipeline.

Research question:
{state.query}

Draft report:
{state.draft.text}

Available source evidence:
{evidence}

Evaluate the draft against the available evidence.

Check for:
1. Factual claims that are unsupported or contradicted by the evidence.
2. Citations that reference unavailable source IDs.
3. Claims that overstate what the evidence establishes.
4. Important uncertainty that the draft fails to communicate.

Do not assume that a claim is correct merely because it has a citation.
Use only the supplied evidence.

Return ONLY valid JSON matching this structure:
{{
  "passed": true,
  "issues": [
    {{
      "category": "unsupported_claim",
      "description": "Description of the problem",
      "source_ids": ["S1"]
    }}
  ],
  "feedback": "Actionable revision instructions"
}}

Set passed to false if material issues are found.
If no material issues are found, return an empty issues list.
""".strip()
