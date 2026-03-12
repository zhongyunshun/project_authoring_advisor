"""HTTP adapter — connects to any RAG backend that exposes a REST API."""

from __future__ import annotations

import logging
from typing import Optional

import httpx

from voice_agent.adapters.base import BackendAdapter, BackendQuery, BackendResponse

logger = logging.getLogger(__name__)


class HTTPBackendAdapter(BackendAdapter):
    """Calls a REST endpoint (e.g. FastAPI) that wraps the RAG pipeline.

    Expected endpoint contract:
        POST /query
        Body:  {"query": str, "conversation_id": str, "user_id": str, "image_base64": str|null}
        Response: {"answer": str, "sources": [...]}
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        endpoint: str = "/query",
        timeout: float = 60.0,
        api_key: Optional[str] = None,
    ):
        self._base_url = base_url.rstrip("/")
        self._endpoint = endpoint
        self._timeout = timeout
        self._headers = {"Content-Type": "application/json"}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"

    async def process(self, query: BackendQuery) -> BackendResponse:
        payload = {
            "query": query.query,
            "conversation_id": query.conversation_id,
            "user_id": query.user_id,
        }
        if query.image_base64:
            payload["image_base64"] = query.image_base64
        payload.update(query.metadata)

        url = f"{self._base_url}{self._endpoint}"
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(url, json=payload, headers=self._headers)
            resp.raise_for_status()
            data = resp.json()

        return BackendResponse(
            answer=data.get("answer", ""),
            sources=data.get("sources", []),
            metadata=data.get("metadata", {}),
        )

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self._base_url}/health")
                return resp.status_code == 200
        except Exception:
            logger.warning("Backend health check failed for %s", self._base_url)
            return False
