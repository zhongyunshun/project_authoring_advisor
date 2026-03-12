"""Factory for text-to-speech providers."""

from __future__ import annotations

from voice_agent.config import TTSConfig


def create_tts(config: TTSConfig):
    """Return a LiveKit TTS node based on config."""

    if config.provider == "cartesia":
        from livekit.plugins import cartesia

        kwargs = {"model": config.cartesia_model}
        if config.cartesia_voice:
            kwargs["voice"] = config.cartesia_voice
        return cartesia.TTS(**kwargs)

    if config.provider == "elevenlabs":
        from livekit.plugins import elevenlabs

        kwargs = {"model": config.elevenlabs_model}
        if config.elevenlabs_voice:
            kwargs["voice"] = config.elevenlabs_voice
        return elevenlabs.TTS(**kwargs)

    if config.provider == "piper":
        # Piper TTS (local ONNX, non-streaming)
        return _PiperTTS(
            model_path=config.piper_model_path,
            config_path=config.piper_config_path,
        )

    raise ValueError(f"Unsupported TTS provider: {config.provider}")


class _PiperTTS:
    """Minimal wrapper for Piper TTS (local ONNX inference).

    Piper is non-streaming — the entire utterance is synthesized at once.
    This wrapper is used when running fully offline.
    """

    def __init__(self, model_path: str, config_path: str):
        if not model_path:
            raise ValueError("PIPER_MODEL_PATH is required for piper TTS")
        self._model_path = model_path
        self._config_path = config_path

    def synthesize(self, text: str) -> bytes:
        """Synchronous synthesis — caller should run in executor."""
        import subprocess
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            out_path = f.name

        cmd = [
            "piper",
            "--model", self._model_path,
            "--output_file", out_path,
        ]
        if self._config_path:
            cmd.extend(["--config", self._config_path])

        proc = subprocess.run(
            cmd, input=text.encode(), capture_output=True, check=True
        )

        with open(out_path, "rb") as f:
            audio = f.read()
        os.unlink(out_path)
        return audio
