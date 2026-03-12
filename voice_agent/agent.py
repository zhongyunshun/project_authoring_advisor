"""LiveKit Voice Agent — the conversational agent that bridges voice ↔ RAG."""

from __future__ import annotations

import logging

from livekit.agents import Agent, AgentSession, RunContext, function_tool

from voice_agent.adapters.base import BackendAdapter, BackendQuery, BackendResponse
from voice_agent.adapters.http_backend import HTTPBackendAdapter
from voice_agent.config import get_settings

logger = logging.getLogger(__name__)


class VoiceAgent(Agent):
    """Voice-first agent that delegates knowledge queries to a backend adapter.

    The agent itself handles conversation flow; the backend adapter handles
    retrieval / RAG. Swap the adapter to plug into any RAG system.
    """

    def __init__(self, backend: BackendAdapter | None = None):
        settings = get_settings()
        self._backend = backend or HTTPBackendAdapter(
            base_url=settings.text_backend_url,
        )
        super().__init__(
            instructions=(
                "You are a helpful, multilingual voice assistant powered by a "
                "knowledge retrieval system. Use the query_backend tool for any "
                "question requiring document search, technical knowledge, or "
                "project-specific information. Speak clearly and concisely — "
                "your responses will be read aloud."
            ),
        )

    async def on_enter(self):
        """Called when the agent joins a room — greet the user."""
        self.session.generate_reply(
            instructions="Greet the user warmly and ask how you can help today."
        )

    @function_tool
    async def query_backend(
        self,
        context: RunContext,
        user_query: str,
    ) -> str:
        """Send a user question to the knowledge retrieval backend and return
        the answer. Use this for any factual, technical, or project-specific
        question."""
        query = BackendQuery(
            query=user_query,
            conversation_id=str(context.session.id),
            user_id=context.userdata.get("user_id", "default")
            if context.userdata
            else "default",
        )
        try:
            response: BackendResponse = await self._backend.process(query)
            return response.answer
        except Exception as e:
            logger.error("Backend query failed: %s", e)
            return (
                "I'm sorry, I couldn't retrieve that information right now. "
                "Could you try rephrasing your question?"
            )
