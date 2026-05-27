import time
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from backend.core.logger import get_logger

logger = get_logger("middleware.request")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Logs every HTTP request and response.

    Logged fields:
    - method, path, query params
    - response status code
    - request duration in ms
    - client IP
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        start = time.perf_counter()

        # Log incoming request
        logger.info(
            "REQUEST  %s %s | params=%s | client=%s",
            request.method,
            request.url.path,
            dict(request.query_params),
            request.client.host if request.client else "unknown",
        )

        try:
            response = await call_next(request)
        except Exception as exc:
            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            logger.error(
                "ERROR    %s %s | duration=%sms | error=%s",
                request.method,
                request.url.path,
                duration_ms,
                str(exc),
            )
            raise

        duration_ms = round((time.perf_counter() - start) * 1000, 2)

        level = (
            logger.warning if response.status_code >= 400
            else logger.info
        )
        level(
            "RESPONSE %s %s | status=%s | duration=%sms",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )

        return response