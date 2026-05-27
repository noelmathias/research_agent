from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from typing import Any, Dict, Optional


# ── Custom exception hierarchy ────────────────────────────────────────────────

class ResearchAgentError(Exception):
    """Base exception for all application errors."""
    status_code: int = 500
    error_code: str = "INTERNAL_ERROR"

    def __init__(self, message: str, detail: Optional[Any] = None):
        super().__init__(message)
        self.message = message
        self.detail = detail


class PipelineError(ResearchAgentError):
    """Raised when an agent pipeline step fails unrecoverably."""
    status_code = 500
    error_code = "PIPELINE_ERROR"


class IngestionError(ResearchAgentError):
    """Raised when document ingestion fails."""
    status_code = 422
    error_code = "INGESTION_ERROR"


class RetrievalError(ResearchAgentError):
    """Raised when vector retrieval fails."""
    status_code = 500
    error_code = "RETRIEVAL_ERROR"


class PDFParseError(ResearchAgentError):
    """Raised when PDF parsing fails."""
    status_code = 422
    error_code = "PDF_PARSE_ERROR"


class MCPToolError(ResearchAgentError):
    """Raised when an MCP tool invocation fails."""
    status_code = 500
    error_code = "MCP_TOOL_ERROR"


class MCPToolNotFoundError(ResearchAgentError):
    """Raised when a requested MCP tool is not registered."""
    status_code = 404
    error_code = "MCP_TOOL_NOT_FOUND"


class ReportError(ResearchAgentError):
    """Raised when report generation or assembly fails."""
    status_code = 500
    error_code = "REPORT_ERROR"


class ConfigurationError(ResearchAgentError):
    """Raised when a required configuration value is missing or invalid."""
    status_code = 500
    error_code = "CONFIGURATION_ERROR"


# ── Error response builder ────────────────────────────────────────────────────

def _error_envelope(
    error_code: str,
    message: str,
    status_code: int,
    detail: Optional[Any] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "error": True,
        "error_code": error_code,
        "message": message,
        "status_code": status_code,
    }
    if detail is not None:
        payload["detail"] = detail
    return payload


# ── FastAPI exception handlers ────────────────────────────────────────────────

def register_exception_handlers(app: FastAPI) -> None:
    """
    Registers all custom exception handlers on the FastAPI app.
    Call once during app initialisation in main.py.
    """

    @app.exception_handler(ResearchAgentError)
    async def handle_research_agent_error(
        request: Request, exc: ResearchAgentError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_envelope(
                error_code=exc.error_code,
                message=exc.message,
                status_code=exc.status_code,
                detail=exc.detail,
            ),
        )

    @app.exception_handler(ValueError)
    async def handle_value_error(
        request: Request, exc: ValueError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=_error_envelope(
                error_code="VALIDATION_ERROR",
                message=str(exc),
                status_code=422,
            ),
        )

    @app.exception_handler(FileNotFoundError)
    async def handle_file_not_found(
        request: Request, exc: FileNotFoundError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=404,
            content=_error_envelope(
                error_code="FILE_NOT_FOUND",
                message=str(exc),
                status_code=404,
            ),
        )

    @app.exception_handler(Exception)
    async def handle_unhandled_exception(
        request: Request, exc: Exception
    ) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content=_error_envelope(
                error_code="UNHANDLED_ERROR",
                message="An unexpected error occurred.",
                status_code=500,
                detail=str(exc),
            ),
        )