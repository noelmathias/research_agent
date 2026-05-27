from typing import List

from sentence_transformers import SentenceTransformer

from shared.config import get_settings

settings = get_settings()


class EmbeddingClient:
    """
    Local embedding model wrapper using SentenceTransformers.

    Completely independent from Ollama generation models.
    """

    def __init__(self):
        self._model = SentenceTransformer(
            settings.embedding_model,
            device=settings.embedding_device,
        )

    async def embed_texts(
        self,
        texts: List[str],
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        """

        if not texts:
            return []

        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        return embeddings.tolist()

    async def embed_query(
        self,
        query: str,
    ) -> List[float]:
        """
        Generate embedding for a single query.
        """

        embeddings = await self.embed_texts([query])

        return embeddings[0]