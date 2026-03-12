"""Voice agent configuration — loaded from .env file and environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

# Load voice_agent/.env (next to this file)
_env_path = Path(__file__).parent / ".env"
load_dotenv(_env_path)


@dataclass(frozen=True)
class STTConfig:
    """Deepgram speech-to-text settings.

    Attributes:
        api_key: Deepgram API key for authentication.
        model: Deepgram STT model name (e.g. "nova-3").
        language: Language code for transcription. Use "multi" for
            automatic language detection.
    """

    api_key: str = ""
    model: str = "nova-3"
    language: str = "multi"


@dataclass(frozen=True)
class TranscriberSettings:
    """Top-level settings for the real-time transcriber.

    Attributes:
        sample_rate: Audio sample rate in Hz.
        channels: Number of audio input channels.
        stt: Deepgram STT configuration.
    """

    sample_rate: int = 16000
    channels: int = 1
    stt: STTConfig = STTConfig()

    @classmethod
    def from_env(cls) -> TranscriberSettings:
        """Build settings from environment variables.

        Returns:
            A ``TranscriberSettings`` instance populated from environment
            variables, falling back to defaults when a variable is unset.
        """
        return cls(
            sample_rate=int(os.getenv("SAMPLE_RATE", "16000")),
            channels=int(os.getenv("CHANNELS", "1")),
            stt=STTConfig(
                api_key=os.getenv("DEEPGRAM_API_KEY", ""),
                model=os.getenv("DEEPGRAM_MODEL", "nova-3"),
                language=os.getenv("DEEPGRAM_LANGUAGE", "multi"),
            ),
        )


@lru_cache(maxsize=1)
def get_settings() -> TranscriberSettings:
    """Return the cached application settings singleton.

    Returns:
        A ``TranscriberSettings`` instance. The same object is returned on
        every call.
    """
    return TranscriberSettings.from_env()
