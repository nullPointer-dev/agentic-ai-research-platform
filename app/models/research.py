from datetime import datetime

from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    title: str
    url: str
    content: str
    score: float
    published_date: datetime | None = None
    source_domain: str


class ResearchState(BaseModel):
    query: str
    search_results: list[SearchResult] = Field(default_factory=list)