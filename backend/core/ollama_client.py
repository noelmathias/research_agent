import httpx
from shared.config import get_settings
from backend.core.logger import get_logger
from typing import Optional

settings = get_settings()
logger = get_logger(__name__)


class OllamaClient:
    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
    ):
        self.base_url = base_url or settings.ollama_base_url
        self.model = model or settings.ollama_model
        self.timeout_seconds = timeout_seconds or settings.ollama_timeout_seconds

    async def generate(
        self, prompt: str, model: Optional[str] = None
    ) -> str:
        target_model = model or self.model
        payload = {
            "model": target_model,
            "prompt": prompt,
            "stream": False,
        }
        logger.debug(
            "Ollama generate | model=%s | prompt_len=%d | timeout=%.1fs",
            target_model,
            len(prompt),
            self.timeout_seconds,
        )
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(
                f"{self.base_url}/api/generate",
                json=payload,
            )
            response.raise_for_status()
            result = response.json()["response"]
            logger.debug(
                "Ollama response | model=%s | response_len=%d",
                target_model,
                len(result),
            )
            return result

    async def is_reachable(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(f"{self.base_url}/api/tags")
                reachable = r.status_code == 200
                logger.debug("Ollama reachable: %s", reachable)
                return reachable
        except Exception as e:
            logger.warning("Ollama not reachable: %s", e)
            return False
