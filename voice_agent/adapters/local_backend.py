"""Local adapter — calls the current project's RAG pipeline directly (no HTTP).

Use this when the voice agent runs in the same process as the RAG system.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from voice_agent.adapters.base import BackendAdapter, BackendQuery, BackendResponse

if TYPE_CHECKING:
    from pipeline.base import BaseRAGPipeline

logger = logging.getLogger(__name__)


class LocalBackendAdapter(BackendAdapter):
    """Wraps any BaseRAGPipeline (RAGEngine, RAGAgent, etc.) as a backend."""

    def __init__(self, pipeline: BaseRAGPipeline):
        self._pipeline = pipeline

    async def process(self, query: BackendQuery) -> BackendResponse:
        # RAGEngine.query is sync — run in executor to avoid blocking the event loop
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, self._pipeline.query, query.query)

        sources = [
            {"text": s.text, "metadata": s.metadata, "score": s.score}
            for s in result.sources
        ]
        return BackendResponse(answer=result.answer, sources=sources)

    async def health_check(self) -> bool:
        return True
