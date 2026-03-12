"""Real-time microphone transcription using Deepgram streaming API (SDK v6).

Run with:
    python -m voice_agent.app

Ctrl+C to stop.
"""

from __future__ import annotations

import sys
import threading
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd
from deepgram.core.events import EventType
from deepgram.listen.v1.types.listen_v1results import ListenV1Results

from voice_agent.config import get_settings
from voice_agent.stt_factory import create_deepgram_client, get_connect_options

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

_transcript_file = None


def _on_message(result: object) -> None:
    """Handle transcript message events from Deepgram.

    Only processes ``ListenV1Results`` with ``is_final=True``. Other message
    types (metadata, utterance end, speech started) are silently ignored.

    Args:
        result: A Deepgram SDK message object. Expected to be one of
            ``ListenV1Results``, ``ListenV1Metadata``,
            ``ListenV1UtteranceEnd``, or ``ListenV1SpeechStarted``.
    """
    # Only process transcript results, ignore metadata/utterance_end/speech_started
    if not isinstance(result, ListenV1Results):
        return

    if not result.is_final:
        return

    alternatives = result.channel.alternatives
    if not alternatives:
        return
    transcript = alternatives[0].transcript
    if not transcript:
        return

    # Show detected language if available
    lang = getattr(alternatives[0], "detected_language", None) or ""
    if lang:
        print(f"[{lang}] {transcript}")
    else:
        print(transcript)

    # Save to file
    if _transcript_file:
        prefix = f"[{lang}] " if lang else ""
        _transcript_file.write(f"{prefix}{transcript}\n")
        _transcript_file.flush()


def main() -> None:
    """Entry point for real-time microphone transcription.

    Connects to Deepgram's streaming API, captures audio from the default
    microphone via ``sounddevice``, and prints final transcriptions to the
    terminal. Transcripts are also saved to a timestamped file under
    ``voice_agent/transcripts/``.

    Raises:
        SystemExit: If ``DEEPGRAM_API_KEY`` is not configured.
    """
    global _transcript_file

    settings = get_settings()

    if not settings.stt.api_key:
        logger.error("DEEPGRAM_API_KEY is not set. Export it or add it to voice_agent/.env")
        sys.exit(1)

    # Create output file with timestamp
    output_dir = Path(__file__).parent / "transcripts"
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"transcript_{timestamp}.txt"

    client = create_deepgram_client(settings.stt)
    connect_opts = get_connect_options(settings.stt)

    logger.info("Starting real-time transcription (Ctrl+C to stop)...")
    logger.info("Model: %s | Language: %s | Sample rate: %d",
                settings.stt.model, settings.stt.language, settings.sample_rate)
    logger.info("Saving to: %s", output_path)
    logger.info("")

    stop_event = threading.Event()

    with open(output_path, "w", encoding="utf-8") as f:
        _transcript_file = f

        with client.listen.v1.connect(**connect_opts) as connection:
            connection.on(EventType.MESSAGE, _on_message)
            connection.on(EventType.ERROR, lambda err: logger.error("Deepgram error: %s", err))

            def _audio_callback(
                indata: np.ndarray,
                frames: int,
                time_info: object,
                status: sd.CallbackFlags,
            ) -> None:
                """Sounddevice callback that streams mic audio to Deepgram.

                Args:
                    indata: Input audio buffer as a float32 numpy array.
                    frames: Number of audio frames in this block.
                    time_info: Timing information from PortAudio.
                    status: Stream status flags (e.g. overflow warnings).
                """
                if status:
                    logger.warning("Audio status: %s", status)
                try:
                    audio_bytes = (indata[:, 0] * 32767).astype(np.int16).tobytes()
                    connection.send_media(audio_bytes)
                except Exception:
                    pass  # connection may be closing

            # Start audio capture BEFORE start_listening() to ensure audio
            # flows immediately and Deepgram doesn't time out waiting for data.
            try:
                with sd.InputStream(
                    samplerate=settings.sample_rate,
                    channels=settings.channels,
                    dtype="float32",
                    callback=_audio_callback,
                    blocksize=int(settings.sample_rate * 0.1),  # 100ms blocks
                ):
                    connection.start_listening()
                    stop_event.wait()
            except KeyboardInterrupt:
                pass
            finally:
                logger.info("\nStopping transcription...")
                connection.send_finalize()

        _transcript_file = None

    logger.info("Transcript saved to: %s", output_path)


if __name__ == "__main__":
    main()
