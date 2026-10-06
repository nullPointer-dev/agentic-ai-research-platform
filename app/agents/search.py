from app.models.research import ResearchState
from app.services.search import SearchService


class SearchAgent:
    def __init__(self, search_service: SearchService) -> None:
        self.search_service = search_service

    async def run(self, state: ResearchState) -> dict:
        results = await self.search_service.search(state.query)
        return {"search_results": results}