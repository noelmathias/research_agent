import shutil
from pathlib import Path
from typing import Optional
from backend.core.pdf_parser import PDFParser
from backend.pipelines.ingestion_pipeline import ingest_texts
from shared.schemas import PDFUploadResponse, PDFPageInfo
from shared.config import get_settings

settings = get_settings()


def save_upload(file_bytes: bytes, filename: str) -> Path:
    """
    Persist uploaded file bytes to the uploads directory.
    Returns the saved file path.
    """
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    dest = upload_dir / filename
    dest.write_bytes(file_bytes)
    return dest


def process_pdf(
    file_path: Path,
    collection_name: Optional[str] = None,
) -> PDFUploadResponse:
    """
    Full PDF processing pipeline:
    1. Parse PDF with PyMuPDF — extract per-page text + metadata
    2. Filter empty pages
    3. Chunk each page's text
    4. Embed all chunks
    5. Store in ChromaDB with source metadata

    Returns PDFUploadResponse with ingestion summary.
    """
    collection = collection_name or settings.chroma_collection_name

    # Step 1 — Parse
    parse_result = PDFParser.parse(file_path)

    # Step 2 — Filter pages with actual content
    valid_pages = [p for p in parse_result.pages if p["text"].strip()]

    if not valid_pages:
        raise ValueError(
            f"PDF '{parse_result.filename}' has no extractable text. "
            "It may be a scanned image PDF — OCR is not supported in this phase."
        )

    # Step 3 — Prepare texts and per-page metadata for ingestion
    texts = [p["text"] for p in valid_pages]
    metadatas = [
        {
            "source": parse_result.filename,
            "page_number": p["page_number"],
            "total_pages": parse_result.total_pages,
            "word_count": p["word_count"],
            "type": "pdf",
        }
        for p in valid_pages
    ]

    # Step 4+5 — Chunk + embed + store (reuses ingestion pipeline)
    result = ingest_texts(
        texts=texts,
        metadatas=metadatas,
        collection_name=collection,
        pre_chunked=False,  # ingestion_pipeline will chunk each page
    )

    # Build page info summary
    page_info = [
        PDFPageInfo(
            page_number=p["page_number"],
            char_count=p["char_count"],
            word_count=p["word_count"],
        )
        for p in parse_result.pages
    ]

    return PDFUploadResponse(
        filename=parse_result.filename,
        file_path=str(file_path),
        total_pages=parse_result.total_pages,
        total_chars=parse_result.total_chars,
        chunks_stored=result["chunks_stored"],
        collection=collection,
        page_info=page_info,
        ids=result["ids"],
    )