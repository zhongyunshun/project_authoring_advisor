"""Factory for creating a Deepgram live transcription connection (SDK v6)."""

from __future__ import annotations

from deepgram import DeepgramClient

from voice_agent.config import STTConfig


def create_deepgram_client(config: STTConfig) -> DeepgramClient:
    """Create a Deepgram client from the given STT config.

    Args:
        config: STT configuration containing the Deepgram API key.

    Returns:
        An authenticated ``DeepgramClient`` instance.
    """
    return DeepgramClient(api_key=config.api_key)


def get_connect_options(config: STTConfig, endpointing_ms: int = 300) -> dict[str, str]:
    """Build keyword arguments for ``client.listen.v1.connect()``.

    Args:
        config: STT configuration containing model and language settings.
        endpointing_ms: Silence duration in milliseconds before Deepgram
            finalizes an utterance. Higher values allow longer pauses
            mid-sentence without splitting. Default is 300ms.

    Returns:
        A dictionary of string options to pass as keyword arguments to
        the Deepgram live transcription connect call.
    """
    opts = {
        "model": config.model,
        "language": config.language,
        "encoding": "linear16",
        "sample_rate": "16000",
        "channels": "1",
        "interim_results": "false",
        "punctuate": "true",
        "smart_format": "true",
        "endpointing": str(endpointing_ms),
    }
    return opts
