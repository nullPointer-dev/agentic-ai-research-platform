from datetime import datetime
from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    id: str
    title: str
    url: str
    content: str
    score: float
    published_date: datetime | None = None
    source_domain: str

class Summary(BaseModel):
    text: str
    source_ids: list[str] = Field(default_factory=list)

class Draft(BaseModel):
    text: str

class ResearchState(BaseModel):
    query: str
    search_results: list[SearchResult] = Field(default_factory=list)
    summary: Summary | None = None
    draft: Draft | None = None