import os
from urllib.parse import urlparse
from tavily import AsyncTavilyClient
from app.models.research import SearchResult
from dotenv import load_dotenv

load_dotenv()

class SearchService:
    def __init__(self) -> None:
        self.client = AsyncTavilyClient(
            api_key=os.environ["TAVILY_API_KEY"],
        )
    async def search(self, query: str) -> list[SearchResult]:
        response = await self.client.search(
            query=query,
            max_results=5,
        )
        return [
            SearchResult(
                title=result["title"],
                url=result["url"],
                content=result["content"],
                score=result["score"],
                published_date=result.get("published_date"),
                source_domain=urlparse(result["url"]).netloc,
            ) for result in response["results"]
        ]