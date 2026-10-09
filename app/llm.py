import os
from typing import Any
from dotenv import load_dotenv
from openai import AsyncOpenAI

import logging
logger = logging.getLogger(__name__)

load_dotenv()


class LLMService:
    def __init__(self) -> None:
        self.client = AsyncOpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
        )

    async def generate(
        self, 
        prompt: str, 
        response_format: dict[str, Any] | None = None,
    ) -> str:
        request: dict[str, Any] = {
            "model": os.environ["LLM_MODEL"],
            "input": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        }
        if response_format is not None:
            request["text"] = {"format": response_format}

        response = await self.client.responses.create(**request)

        logger.info(
            "LLM response: model=%s status=%s",
            response.model,
            response.status,
        )


        print("---Input Tokens Details---")
        print(response.usage.input_tokens)
        print(response.usage.input_tokens_details)
        print("---Output Tokens Details---")
        print(response.usage.output_tokens)
        print(response.usage.output_tokens_details)

        output_text = response.output_text
        if not output_text or not output_text.strip():
            logger.error(
                "LLM returned no text: model=%s status=%s "
                "incomplete_details=%r output=%r",
                os.environ["LLM_MODEL"],
                response.status,
                response.incomplete_details,
                response.output,
            )
            raise RuntimeError(
                "LLM returned no text output; inspect the preceding log."
            )

        return output_text