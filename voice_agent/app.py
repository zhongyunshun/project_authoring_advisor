"""LiveKit Agent Server entrypoint.

Run with:
    python -m voice_agent.app

Or via the livekit CLI:
    livekit-agents start voice_agent.app
"""

from __future__ import annotations

import logging

from livekit.agents import AgentSession, AgentServer, JobContext
from livekit.plugins import silero, openai

from voice_agent.agent import VoiceAgent
from voice_agent.config import get_settings
from voice_agent.stt_factory import create_stt
from voice_agent.tts_factory import create_tts

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    """Called for each new room / participant. Sets up the full voice pipeline."""
    settings = get_settings()

    # -- Speech-to-Text --
    stt = create_stt(settings.stt)

    # -- Text-to-Speech --
    tts = create_tts(settings.tts)

    # -- LLM (OpenAI-compatible, can point to LiteLLM proxy) --
    llm_kwargs = {"model": settings.llm.model}
    if settings.llm.api_base:
        llm_kwargs["base_url"] = settings.llm.api_base
    if settings.llm.api_key:
        llm_kwargs["api_key"] = settings.llm.api_key
    llm = openai.LLM(**llm_kwargs)

    # -- Voice Activity Detection --
    vad = silero.VAD.load()

    # -- Assemble session --
    session = AgentSession(
        vad=vad,
        stt=stt,
        llm=llm,
        tts=tts,
    )

    logger.info(
        "Starting voice session: stt=%s tts=%s llm=%s",
        settings.stt.provider,
        settings.tts.provider,
        settings.llm.model,
    )
    await session.start(agent=VoiceAgent(), room=ctx.room)


if __name__ == "__main__":
    server.run()
