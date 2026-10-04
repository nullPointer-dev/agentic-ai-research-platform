import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()


class LLMService:
    def __init__(self) -> None:
        self.client = AsyncOpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
        )

    async def generate(self, prompt: str) -> str:
        response = await self.client.chat.completions.create(
            model=os.environ["LLM_MODEL"],
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        return response.choices[0].message.content or ""