"""Factory for speech-to-text providers."""

from __future__ import annotations

from voice_agent.config import STTConfig


def create_stt(config: STTConfig):
    """Return a LiveKit STT node based on config."""

    if config.provider == "deepgram":
        from livekit.plugins import deepgram

        return deepgram.STT(
            model=config.deepgram_model,
            language=config.deepgram_language,
        )

    if config.provider == "local_whisper":
        from livekit.plugins import silero
        from faster_whisper import WhisperModel

        # faster-whisper with a custom STT node
        # This returns a WhisperModel; we wrap it in the app.py entrypoint
        # as a custom stt_node for maximum flexibility.
        return _FasterWhisperSTT(
            model_size=config.whisper_model,
            device=config.whisper_device,
            compute_type=config.whisper_compute_type,
        )

    raise ValueError(f"Unsupported STT provider: {config.provider}")


class _FasterWhisperSTT:
    """Thin wrapper around faster-whisper for use as a LiveKit STT node.

    LiveKit agents v1.4+ supports custom STT via the stt parameter accepting
    any object with a `recognize` async method or by subclassing stt.STT.
    """

    def __init__(self, model_size: str, device: str, compute_type: str):
        from faster_whisper import WhisperModel

        self._model = WhisperModel(
            model_size, device=device, compute_type=compute_type
        )

    def transcribe(self, audio_bytes: bytes, sample_rate: int = 16000) -> str:
        """Synchronous transcription — caller should run in executor."""
        import io
        import numpy as np

        audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        segments, _ = self._model.transcribe(audio_array, beam_size=5)
        return " ".join(seg.text for seg in segments).strip()
