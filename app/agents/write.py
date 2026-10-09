from app.llm import LLMService
from app.models.research import Draft, ResearchState


class WriteAgent:
    def __init__(self, llm_service: LLMService) -> None:
        self.llm_service = llm_service

    async def run(self, state: ResearchState) -> dict:
        if state.summary is None:
            raise ValueError("Cannot write a report without a summary")

        prompt = f"""
                You are the writing stage of a research pipeline.

                Research question:
                {state.query}

                Synthesized findings:
                {state.summary.text}

                Source IDs available in the synthesis:
                {", ".join(state.summary.source_ids)}

                Write a coherent research report that directly answers the research question.

                Requirements:
                - Base the report on the synthesized findings.
                - Preserve source IDs next to the claims they support.
                - Only use source IDs provided above.
                - Do not invent facts, URLs, or citations.
                - Clearly communicate uncertainty where relevant.
                - Use clear headings if they improve readability.
                - Return only the report text, without commentary about these instructions.
            """.strip()

        response = await self.llm_service.generate(prompt)

        draft = Draft(text=response.strip())

        if not draft.text:
            raise ValueError("LLM returned an empty draft")

        if draft.text.strip().lower() == "user safety: safe":
            raise ValueError("LLM returned a safety message instead of a draft")

        if len(draft.text.split()) < 30:
            raise ValueError("LLM returned an incomplete draft")

        return {"draft": draft}