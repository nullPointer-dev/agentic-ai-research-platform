import json
from app.llm import LLMService
from app.models.research import ResearchState, Summary, SearchResult


def build_evidence(results: list[SearchResult]) -> str:
    sections = []

    for result in results:
        sections.append(
            f"[{result.id}]\n"
            f"Title: {result.title}\n"
            f"Source: {result.source_domain}\n"
            f"URL: {result.url}\n"
            f"Content:\n{result.content}"
        )

    return "\n\n".join(sections)


class SummarizeAgent:
    def __init__(self, llm_service: LLMService) -> None:
        self.llm_service = llm_service

    async def run(self, state: ResearchState) -> dict:
        evidence = build_evidence(state.search_results)

        prompt = f"""
You are the summarization stage of a research pipeline.

Research question:
{state.query}

Below are retrieved sources. Use only information supported by these
sources. Do not invent facts.

{evidence}

Produce a concise synthesis that directly answers the research question.

Every source you rely on should be included in source_ids using the
provided source identifiers.

Return ONLY valid JSON in this exact format:

{{
  "text": "your synthesis",
  "source_ids": ["S1", "S2"]
}}
""".strip()

        response = await self.llm_service.generate(prompt)

        data = json.loads(response)
        summary = Summary.model_validate(data)

        valid_source_ids = {
            result.id
            for result in state.search_results
        }

        if not set(summary.source_ids).issubset(valid_source_ids):
            raise ValueError("Summary contains an invalid source ID")
        
        return {"summary": summary}