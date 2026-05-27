from backend.core.embedding_client import EmbeddingClient
from backend.core.vector_store import VectorStore
from shared.config import get_settings
from shared.schemas import (
    SearchResult,
    SearchResponse,
    VectorStoreStatusResponse,
)
from typing import Optional

settings = get_settings()
embedding_client = EmbeddingClient()
vector_store = VectorStore()


def get_vector_store_status() -> VectorStoreStatusResponse:
    return VectorStoreStatusResponse(
        collection_name=settings.chroma_collection_name,
        document_count=vector_store.collection_count(),
        embedding_model=settings.embedding_model,
    )


async def search_documents(query: str, top_k: int, collection_name: Optional[str] = None) -> SearchResponse:
    if not query.strip():
        raise ValueError("Search query cannot be empty.")

    query_embedding = await embedding_client.embed_query(query)
    raw_results = vector_store.search(query_embedding=query_embedding, top_k=top_k)

    documents = raw_results.get("documents", [[]])[0]
    metadatas = raw_results.get("metadatas", [[]])[0]
    distances = raw_results.get("distances", [[]])[0]

    matches = []
    for document, metadata, distance in zip(documents, metadatas, distances):
        metadata = metadata or {}
        matches.append(
            SearchResult(
        id=str(metadata.get("chunk_id", "")),
        text=document,
        score=max(0.0, 1.0 - float(distance)),
        metadata=metadata,
    )
)

    return SearchResponse(
        query=query,
        collection=settings.chroma_collection_name,
        results=matches,
        total_results=len(matches),
    )
