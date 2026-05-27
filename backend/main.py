from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.api.routes.health import router as health_router
from backend.api.routes.research import router as research_router
from backend.api.routes.research_stream import router as stream_router
from backend.api.routes.vector import router as vector_router
from backend.api.routes.pdf import router as pdf_router
from backend.api.routes.mcp import router as mcp_router
from backend.api.routes.report import router as report_router
from backend.core.errors import register_exception_handlers
from backend.middleware.logging_middleware import RequestLoggingMiddleware
from backend.core.logger import get_logger
from shared.config import get_settings

settings = get_settings()
logger = get_logger(__name__)


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version="0.10.1",
        description="Autonomous Multi-Agent Research Assistant",
    )

    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_exception_handlers(app)

    app.include_router(health_router, prefix="/api/v1", tags=["Health"])
    app.include_router(research_router, prefix="/api/v1", tags=["Research"])
    app.include_router(stream_router, prefix="/api/v1", tags=["Research Async"])
    app.include_router(vector_router, prefix="/api/v1", tags=["Vector"])
    app.include_router(pdf_router, prefix="/api/v1", tags=["PDF"])
    app.include_router(mcp_router, prefix="/api/v1", tags=["MCP"])
    app.include_router(report_router, prefix="/api/v1", tags=["Report"])

    @app.on_event("startup")
    async def on_startup():
        logger.info("Research Agent v%s starting", app.version)
        logger.info(
            "Endpoints: /research/run_async + /research/stream/{job_id} active"
        )

    @app.on_event("shutdown")
    async def on_shutdown():
        logger.info("Research Agent shutting down.")

    @app.get("/")
    async def root():
        return {
            "message": f"{settings.app_name} is running.",
            "version": app.version,
        }

    return app


app = create_app()