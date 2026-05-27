from pathlib import Path
from typing import Any, Dict, Optional
from backend.core.pdf_parser import PDFParser
from backend.mcp.schemas import ToolCallResult, ToolName
from shared.config import get_settings

settings = get_settings()


async def run_pdf_parse(
    filename: str,
    collection_name: Optional[str] = None,
) -> ToolCallResult:
    """
    MCP PDF tool.
    Locates a previously uploaded PDF by filename, parses it,
    and returns per-page text with metadata.

    Does NOT re-ingest — parsing only.
    For ingestion use pdf_pipeline.process_pdf() directly.
    """
    try:
        file_path = Path(settings.upload_dir) / filename

        if not file_path.exists():
            return ToolCallResult(
                tool_name=ToolName.PDF_PARSE,
                success=False,
                result={},
                error=f"PDF not found in uploads: {filename}",
            )

        parse_result = PDFParser.parse(file_path)

        pages_out = [
            {
                "page_number": p["page_number"],
                "text": p["text"],
                "word_count": p["word_count"],
            }
            for p in parse_result.pages
            if p["text"].strip()
        ]

        return ToolCallResult(
            tool_name=ToolName.PDF_PARSE,
            success=True,
            result=pages_out,
            metadata={
                "filename": parse_result.filename,
                "total_pages": parse_result.total_pages,
                "total_chars": parse_result.total_chars,
                "pages_with_content": len(pages_out),
            },
        )

    except Exception as e:
        return ToolCallResult(
            tool_name=ToolName.PDF_PARSE,
            success=False,
            result={},
            error=str(e),
        )