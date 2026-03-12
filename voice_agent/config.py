"""Voice agent configuration — loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache


@dataclass(frozen=True)
class STTConfig:
    """Speech-to-text settings."""

    provider: str = "deepgram"  # deepgram | local_whisper
    # Deepgram
    deepgram_model: str = "nova-3"
    deepgram_language: str = "multi"  # "multi" = auto-detect
    # Local Whisper (faster-whisper)
    whisper_model: str = "large-v3"
    whisper_device: str = "cuda"  # cuda | cpu
    whisper_compute_type: str = "float16"


@dataclass(frozen=True)
class TTSConfig:
    """Text-to-speech settings."""

    provider: str = "cartesia"  # cartesia | elevenlabs | piper
    # Cartesia
    cartesia_model: str = "sonic-3"
    cartesia_voice: str = ""  # empty = default voice
    # ElevenLabs
    elevenlabs_model: str = "eleven_turbo_v2_5"
    elevenlabs_voice: str = ""
    # Piper (local)
    piper_model_path: str = ""
    piper_config_path: str = ""


@dataclass(frozen=True)
class LLMConfig:
    """LLM settings (routed via OpenAI-compatible endpoint, e.g. LiteLLM)."""

    model: str = "gpt-4o-mini"
    api_base: str = ""  # empty = default OpenAI
    api_key: str = ""   # empty = read from OPENAI_API_KEY
    temperature: float = 0.7


@dataclass(frozen=True)
class VoiceAgentSettings:
    """Top-level settings for the voice agent."""

    # LiveKit connection
    livekit_url: str = "ws://localhost:7880"
    livekit_api_key: str = "devkey"
    livekit_api_secret: str = "secret"

    # Backend RAG service
    text_backend_url: str = "http://localhost:8000"

    # Component configs
    stt: STTConfig = field(default_factory=STTConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)

    @classmethod
    def from_env(cls) -> VoiceAgentSettings:
        """Build settings from environment variables (with sensible defaults)."""
        return cls(
            livekit_url=os.getenv("LIVEKIT_URL", "ws://localhost:7880"),
            livekit_api_key=os.getenv("LIVEKIT_API_KEY", "devkey"),
            livekit_api_secret=os.getenv("LIVEKIT_API_SECRET", "secret"),
            text_backend_url=os.getenv("TEXT_BACKEND_URL", "http://localhost:8000"),
            stt=STTConfig(
                provider=os.getenv("STT_PROVIDER", "deepgram"),
                deepgram_model=os.getenv("DEEPGRAM_MODEL", "nova-3"),
                deepgram_language=os.getenv("DEEPGRAM_LANGUAGE", "multi"),
                whisper_model=os.getenv("WHISPER_MODEL", "large-v3"),
                whisper_device=os.getenv("WHISPER_DEVICE", "cuda"),
                whisper_compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "float16"),
            ),
            tts=TTSConfig(
                provider=os.getenv("TTS_PROVIDER", "cartesia"),
                cartesia_model=os.getenv("CARTESIA_MODEL", "sonic-3"),
                cartesia_voice=os.getenv("CARTESIA_VOICE", ""),
                elevenlabs_model=os.getenv("ELEVENLABS_MODEL", "eleven_turbo_v2_5"),
                elevenlabs_voice=os.getenv("ELEVENLABS_VOICE", ""),
                piper_model_path=os.getenv("PIPER_MODEL_PATH", ""),
                piper_config_path=os.getenv("PIPER_CONFIG_PATH", ""),
            ),
            llm=LLMConfig(
                model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
                api_base=os.getenv("LLM_API_BASE", ""),
                api_key=os.getenv("LLM_API_KEY", ""),
                temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
            ),
        )


@lru_cache(maxsize=1)
def get_settings() -> VoiceAgentSettings:
    """Cached singleton for settings."""
    return VoiceAgentSettings.from_env()
