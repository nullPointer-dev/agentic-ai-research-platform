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
        response = await self.client.responses.create(
            model=os.environ["LLM_MODEL"],
            input=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )
        print("---Input Tokens Details---")
        print(response.usage.input_tokens)
        print(response.usage.input_tokens_details)
        print("---Output Tokens Details---")
        print(response.usage.output_tokens)
        print(response.usage.output_tokens_details)
        return response.output_text or ""