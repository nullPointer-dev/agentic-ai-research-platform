import pytest
from unittest.mock import AsyncMock, patch

from app.services.search import SearchService


@pytest.mark.anyio
async def test_search():
    tavily_response = {
        "results": [
            {
                "title": "Example Article",
                "url": "https://example.com/article",
                "content": "This is example content.",
                "score": 0.91,
            },
            {
                "title": "Another Article",
                "url": "https://example.org/research",
                "content": "This is more research content.",
                "score": 0.82,
            },
        ]
    }

    with patch(
        "app.services.search.AsyncTavilyClient"
    ) as mock_client:
        mock_client.return_value.search = AsyncMock(
            return_value=tavily_response
        )

        service = SearchService()

        results = await service.search("test query")

    assert len(results) == 2

    assert results[0].title == "Example Article"
    assert results[0].url == "https://example.com/article"
    assert results[0].content == "This is example content."
    assert results[0].score == 0.91
    assert results[0].source_domain == "example.com"
    assert results[0].published_date is None

    assert results[1].title == "Another Article"
    assert results[1].source_domain == "example.org"

    mock_client.return_value.search.assert_awaited_once_with(
        query="test query",
        max_results=5,
    )