"""Middleware de journalisation : request_id, latence, horodatage.

Log chaque requête (méthode, chemin, statut, latence) avec un identifiant
de requête propagé dans l'en-tête ``X-Request-ID``.
"""

import time
import uuid

from fastapi import Request
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware


class LoggingMiddleware(BaseHTTPMiddleware):
    """Journalise chaque requête et propage un identifiant de corrélation."""

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id

        start = time.perf_counter()
        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception:
            logger.bind(request_id=request_id).exception("Erreur non gérée")
            raise

        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        level = "INFO" if status_code < 400 else "WARNING" if status_code < 500 else "ERROR"
        logger.bind(request_id=request_id).log(
            level,
            "{method} {path} {status} {latency_ms}ms",
            method=request.method,
            path=request.url.path,
            status=status_code,
            latency_ms=latency_ms,
        )

        response.headers["X-Request-ID"] = request_id
        return response
