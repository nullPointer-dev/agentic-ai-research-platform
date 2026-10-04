from fastapi import Depends, FastAPI
from app.llm import LLMService
from app.models.requests import AskRequest
from app.dependencies import get_llm_service

app = FastAPI()
llm = LLMService()


@app.get("/")
async def root():
    return {
        "message": "Agentic AI Research",
        "status": "running",
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
    }


@app.post("/ask")
async def ask(request: AskRequest, llm: LLMService = Depends(get_llm_service)):
    response = await llm.generate(request.prompt)
    return {
        "response": response,
    }