# Health check router
from fastapi import APIRouter
from shared.schemas import HealthResponse, OllamaTestRequest, OllamaTestResponse
from backend.core.ollama_client import OllamaClient
from shared.config import get_settings

router = APIRouter()
settings = get_settings()
ollama = OllamaClient()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    reachable = await ollama.is_reachable()
    return HealthResponse(
        status="ok",
        version="0.2.0",
        ollama_reachable=reachable,
        model=settings.ollama_model,
    )


@router.post("/test-ollama", response_model=OllamaTestResponse)
async def test_ollama(request: OllamaTestRequest):
    model = request.model or settings.ollama_model
    result = await ollama.generate(prompt=request.prompt, model=model)
    return OllamaTestResponse(response=result, model=model)
