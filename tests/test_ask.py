from fastapi.testclient import TestClient
from app.dependencies import get_llm_service
from app.main import app


class FakeLLMService:
    async def generate(self, prompt: str, response_format: dict[str, any] | None = None) -> str:
        return f"Fake response to: {prompt}"


def test_ask():
    app.dependency_overrides[get_llm_service] = lambda: FakeLLMService()
    client = TestClient(app)
    response = client.post(
        "/ask",
        json={"prompt": "Hello"},
    )
    assert response.status_code == 200
    assert response.json() == {
        "response": "Fake response to: Hello",
    }
    app.dependency_overrides.clear()