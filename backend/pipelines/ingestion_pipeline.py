from typing import List, Dict, Any, Optional
from pathlib import Path
from backend.core.embedding_client import EmbeddingClient
from backend.core.vector_store import VectorStore
from backend.core.logger import get_logger
from shared.config import get_settings

settings = get_settings()
embedder = EmbeddingClient()
store = VectorStore()
logger = get_logger(__name__)


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> List[str]:
    size = chunk_size or settings.chunk_size
    overlap = chunk_overlap or settings.chunk_overlap

    if len(text) <= size:
        return [text.strip()]

    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = start + size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start += size - overlap

    logger.debug("chunk_text | input_len=%d | chunks=%d", len(text), len(chunks))
    return chunks


def ingest_texts(
    texts: List[str],
    metadatas: Optional[List[Dict[str, Any]]] = None,
    collection_name: Optional[str] = None,
    pre_chunked: bool = False,
) -> Dict[str, Any]:
    all_chunks: List[str] = []
    all_metadatas: List[Dict[str, Any]] = []

    for idx, text in enumerate(texts):
        source_meta = metadatas[idx] if metadatas else {}
        chunks = [text] if pre_chunked else chunk_text(text)
        for chunk_idx, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metadatas.append({
                **source_meta,
                "source_index": idx,
                "chunk_index": chunk_idx,
                "chunk_length": len(chunk),
            })

    if not all_chunks:
        logger.warning("ingest_texts | no chunks produced from %d texts", len(texts))
        return {"chunks_stored": 0, "ids": []}

    logger.info(
        "ingest_texts | texts=%d | total_chunks=%d | collection=%s",
        len(texts),
        len(all_chunks),
        collection_name or settings.chroma_collection_name,
    )

    embeddings = embedder.embed(all_chunks)
    ids = store.upsert(
        texts=all_chunks,
        embeddings=embeddings,
        metadatas=all_metadatas,
        collection_name=collection_name,
    )

    logger.info("ingest_texts | stored=%d ids", len(ids))
    return {"chunks_stored": len(ids), "ids": ids}


def ingest_text(
    text: str,
    metadata: Optional[Dict[str, Any]] = None,
    collection_name: Optional[str] = None,
) -> Dict[str, Any]:
    logger.debug("ingest_text | len=%d", len(text))
    return ingest_texts(
        texts=[text],
        metadatas=[metadata] if metadata else None,
        collection_name=collection_name,
    )


def ingest_file(
    file_path: str | Path,
    metadata: Optional[Dict[str, Any]] = None,
    collection_name: Optional[str] = None,
) -> Dict[str, Any]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    content = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not content:
        raise ValueError(f"File is empty or unreadable: {file_path}")

    logger.info("ingest_file | file=%s | len=%d", path.name, len(content))
    source_meta: Dict[str, Any] = {
        "source": path.name,
        "file_path": str(path),
        "type": "file",
        **(metadata or {}),
    }
    return ingest_texts(
        texts=[content],
        metadatas=[source_meta],
        collection_name=collection_name,
    )