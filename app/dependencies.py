from app.llm import LLMService
llm_service = LLMService()

def get_llm_service() -> LLMService:
    return llm_service