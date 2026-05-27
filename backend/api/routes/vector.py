from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from backend.pipelines.ingestion_pipeline import ingest_file, ingest_text
from backend.pipelines.retrieval_pipeline import (
    get_vector_store_status,
    search_documents,
)
from shared.schemas import (
    IngestRequest,
    IngestResponse,
    SearchRequest,
    SearchResponse,
    VectorStoreStatusResponse,
)

router = APIRouter()


@router.get(
    "/vector/status",
    response_model=VectorStoreStatusResponse,
)
async def vector_status():
    try:
        return get_vector_store_status()

    except Exception as e:
        import traceback

        traceback.print_exc()

        raise HTTPException(
        status_code=500,
        detail=f"Vector store unavailable: {str(e)}",
    )


@router.post(
    "/vector/ingest/text",
    response_model=IngestResponse,
)
async def ingest_text_document(request: IngestRequest):
    try:

        return await ingest_text(
            texts=request.texts,
            metadatas=request.metadatas,
            collection_name=request.collection_name,
        )

    except ValueError as e:

        raise HTTPException(
            status_code=400,
            detail=str(e),
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Ingestion failed: {str(e)}",
        )


@router.post(
    "/vector/ingest/file",
    response_model=IngestResponse,
)
async def ingest_file_document(
    file: UploadFile = File(...),
    source: Optional[str] = Form(default=None),
):
    try:

        return await ingest_file(
            file=file,
            source=source,
        )

    except ValueError as e:

        raise HTTPException(
            status_code=400,
            detail=str(e),
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"File ingestion failed: {str(e)}",
        )


@router.post(
    "/vector/search",
    response_model=SearchResponse,
)
async def vector_search(request: SearchRequest):
    try:

        return await search_documents(
            query=request.query,
            top_k=request.top_k,
            collection_name=request.collection_name,
        )

    except ValueError as e:

        raise HTTPException(
            status_code=400,
            detail=str(e),
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Vector search failed: {str(e)}",
        )