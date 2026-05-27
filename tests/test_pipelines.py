import pytest
from unittest.mock import patch, MagicMock
from backend.pipelines.ingestion_pipeline import (
    chunk_text,
    ingest_texts,
    ingest_text,
)
from backend.pipelines.retrieval_pipeline import retrieve


# ── chunk_text ────────────────────────────────────────────────────────────────

def test_chunk_text_short_input():
    """Text shorter than chunk_size returns single chunk."""
    result = chunk_text("Short text.", chunk_size=512, chunk_overlap=64)
    assert len(result) == 1
    assert result[0] == "Short text."


def test_chunk_text_long_input():
    """Text longer than chunk_size is split into multiple chunks."""
    long_text = "word " * 300  # ~1500 chars
    result = chunk_text(long_text, chunk_size=200, chunk_overlap=20)
    assert len(result) > 1
    for chunk in result:
        assert len(chunk) <= 200


def test_chunk_text_overlap():
    """Chunks overlap by the specified amount."""
    text = "a" * 600
    chunks = chunk_text(text, chunk_size=200, chunk_overlap=50)
    assert len(chunks) >= 3


def test_chunk_text_empty_string():
    """Empty string returns empty list."""
    result = chunk_text("", chunk_size=512, chunk_overlap=64)
    assert result == []


def test_chunk_text_whitespace_only():
    """Whitespace-only string returns empty list."""
    result = chunk_text("   \n\t  ", chunk_size=512, chunk_overlap=64)
    assert result == []


# ── ingest_texts ──────────────────────────────────────────────────────────────

def test_ingest_texts_empty_input():
    """Empty texts list returns zero chunks."""
    result = ingest_texts(texts=[])
    assert result["chunks_stored"] == 0
    assert result["ids"] == []


def test_ingest_texts_calls_embedder_and_store(sample_texts):
    """ingest_texts calls embedder.embed and store.upsert."""
    mock_embed = MagicMock(
        return_value=[[0.1] * 384 for _ in range(10)]
    )
    mock_upsert = MagicMock(
        return_value=["id1", "id2", "id3", "id4", "id5"]
    )

    with patch(
        "backend.pipelines.ingestion_pipeline.embedder.embed",
        mock_embed,
    ), patch(
        "backend.pipelines.ingestion_pipeline.store.upsert",
        mock_upsert,
    ):
        result = ingest_texts(texts=sample_texts)

    assert mock_embed.called
    assert mock_upsert.called
    assert result["chunks_stored"] == len(mock_upsert.return_value)


def test_ingest_text_single_string():
    """ingest_text wraps single string correctly."""
    mock_embed = MagicMock(return_value=[[0.1] * 384])
    mock_upsert = MagicMock(return_value=["id1"])

    with patch(
        "backend.pipelines.ingestion_pipeline.embedder.embed", mock_embed
    ), patch(
        "backend.pipelines.ingestion_pipeline.store.upsert", mock_upsert
    ):
        result = ingest_text("Some text.", metadata={"source": "test"})

    assert result["chunks_stored"] == 1


def test_ingest_texts_pre_chunked(sample_texts):
    """pre_chunked=True skips chunking."""
    mock_embed = MagicMock(
        return_value=[[0.1] * 384 for _ in range(len(sample_texts))]
    )
    mock_upsert = MagicMock(
        return_value=[f"id{i}" for i in range(len(sample_texts))]
    )

    with patch(
        "backend.pipelines.ingestion_pipeline.embedder.embed", mock_embed
    ), patch(
        "backend.pipelines.ingestion_pipeline.store.upsert", mock_upsert
    ):
        result = ingest_texts(texts=sample_texts, pre_chunked=True)

    # Exactly len(sample_texts) chunks — no splitting
    call_args = mock_embed.call_args[0][0]
    assert len(call_args) == len(sample_texts)


# ── retrieve ──────────────────────────────────────────────────────────────────

def test_retrieve_returns_search_results(sample_query):
    """retrieve embeds query and calls vector store search."""
    mock_embed_single = MagicMock(return_value=[0.1] * 384)
    mock_search = MagicMock(return_value={
        "ids": [["id1", "id2"]],
        "documents": [["chunk one", "chunk two"]],
        "metadatas": [[{"source": "doc.pdf"}, {"source": "doc.pdf"}]],
        "distances": [[0.1, 0.3]],
    })

    with patch(
        "backend.pipelines.retrieval_pipeline.embedder.embed_single",
        mock_embed_single,
    ), patch(
        "backend.pipelines.retrieval_pipeline.store.search",
        mock_search,
    ):
        results = retrieve(query=sample_query, top_k=2)

    assert len(results) == 2
    assert results[0].score >= results[1].score  # sorted descending


def test_retrieve_empty_store(sample_query):
    """retrieve returns empty list when store has nothing."""
    mock_embed_single = MagicMock(return_value=[0.1] * 384)
    mock_search = MagicMock(return_value={
        "ids": [[]],
        "documents": [[]],
        "metadatas": [[]],
        "distances": [[]],
    })

    with patch(
        "backend.pipelines.retrieval_pipeline.embedder.embed_single",
        mock_embed_single,
    ), patch(
        "backend.pipelines.retrieval_pipeline.store.search",
        mock_search,
    ):
        results = retrieve(query=sample_query)

    assert results == []