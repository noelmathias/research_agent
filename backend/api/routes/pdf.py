from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
from pathlib import Path
from shared.schemas import PDFUploadResponse
from backend.pipelines.pdf_pipeline import save_upload, process_pdf
from shared.config import get_settings

router = APIRouter()
settings = get_settings()

ALLOWED_CONTENT_TYPES = {"application/pdf", "application/octet-stream"}
MAX_BYTES = settings.max_upload_size_mb * 1024 * 1024


@router.post("/pdf/upload", response_model=PDFUploadResponse)
async def upload_pdf(
    file: UploadFile = File(...),
    collection_name: str = Query(default=None),
):
    # Validate filename
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are accepted.",
        )

    # Read and validate size
    file_bytes = await file.read()
    if len(file_bytes) > MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds maximum size of {settings.max_upload_size_mb}MB.",
        )

    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        # Save to disk
        saved_path = save_upload(file_bytes, file.filename)

        # Process — parse + chunk + embed + store
        result = process_pdf(
            file_path=saved_path,
            collection_name=collection_name,
        )
        return result

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"PDF processing failed: {str(e)}"
        )


@router.get("/pdf/list")
async def list_uploaded_pdfs():
    """List all PDFs currently stored in the uploads directory."""
    upload_dir = Path(settings.upload_dir)
    if not upload_dir.exists():
        return {"files": []}

    files = [
        {
            "filename": f.name,
            "size_kb": round(f.stat().st_size / 1024, 2),
            "path": str(f),
        }
        for f in sorted(upload_dir.glob("*.pdf"))
    ]
    return {"files": files, "total": len(files)}