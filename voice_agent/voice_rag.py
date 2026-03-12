"""Voice-powered RAG: ask questions by voice, get answers from the RAG system.

Captures audio from the microphone, transcribes via Deepgram, sends the
transcript as a query to the RAG engine, and prints the answer. Optionally
speaks the answer back using Deepgram TTS.

Run with:
    python -m voice_agent.voice_rag --model openai --embedding openai
    python -m voice_agent.voice_rag --model openai --embedding openai --speak
    python -m voice_agent.voice_rag --model openai --embedding openai --speak --voice male

Ctrl+C to stop.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import logging
import tempfile
from pathlib import Path

import numpy as np
import sounddevice as sd
from deepgram.core.events import EventType
from deepgram.listen.v1.types.listen_v1results import ListenV1Results

from voice_agent.config import get_settings
from voice_agent.stt_factory import create_deepgram_client, get_connect_options

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _play_audio_file(file_path: str, sample_rate: int = 48000) -> None:
    """Play a WAV file through the default audio output.

    Args:
        file_path: Path to the WAV audio file.
        sample_rate: Expected sample rate of the audio file.
    """
    import wave

    with wave.open(file_path, "rb") as wf:
        sr = wf.getframerate()
        channels = wf.getnchannels()
        frames = wf.readframes(wf.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
        if channels > 1:
            audio = audio.reshape(-1, channels)
        sd.play(audio, samplerate=sr)
        sd.wait()


def main() -> None:
    """Entry point for voice-powered RAG.

    Sets up the RAG engine (LLM + embeddings + vector store), then starts
    real-time mic transcription. Each final transcript is sent as a query
    to the RAG engine and the answer is printed. If ``--speak`` is enabled,
    the answer is also spoken back via Deepgram TTS.

    Raises:
        SystemExit: If ``DEEPGRAM_API_KEY`` is not configured or required
            RAG dependencies are missing.
    """
    parser = argparse.ArgumentParser(description="Voice-powered RAG Q&A")
    parser.add_argument("--model", default="openai", choices=["openai", "gemini", "claude"],
                        help="LLM provider (default: openai)")
    parser.add_argument("--model_name", default="", help="Specific model name")
    parser.add_argument("--embedding", default="openai", choices=["openai", "huggingface"],
                        help="Embedding provider (default: openai)")
    parser.add_argument("--collection", default="trca_documents",
                        help="Qdrant collection name (default: trca_documents)")
    parser.add_argument("--top_k", type=int, default=22, help="Number of chunks to retrieve")
    parser.add_argument("--speak", action="store_true",
                        help="Speak the answer back using Deepgram TTS")
    parser.add_argument("--voice", choices=["male", "female"], default="female",
                        help="TTS voice gender when --speak is enabled (default: female)")
    args = parser.parse_args()

    # --- Deepgram STT setup ---
    transcriber_settings = get_settings()
    if not transcriber_settings.stt.api_key:
        logger.error("DEEPGRAM_API_KEY is not set. Export it or add it to voice_agent/.env")
        sys.exit(1)

    # --- RAG engine setup ---
    from config.settings import Settings
    from core.llm_factory import LLMFactory
    from core.embedding_factory import EmbeddingFactory
    from core.vector_store import VectorStoreManager
    from ingest.indexer import Indexer
    from pipeline.rag_engine import RAGEngine

    settings = Settings.from_env()
    settings.apply_env()

    # Load keys fallback
    if not settings.openai_api_key:
        try:
            from config.keys import OPENAI_API_KEY, GOOGLE_API_KEY
            os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
            os.environ.setdefault("GOOGLE_API_KEY", GOOGLE_API_KEY)
        except ImportError:
            pass

    logger.info("Setting up RAG engine...")
    llm = LLMFactory.create(provider=args.model, model=args.model_name)
    embed_model = EmbeddingFactory.create(provider=args.embedding)

    vsm = VectorStoreManager(storage_path="./vector_db/qdrant_storage")
    indexer = Indexer(vsm, embed_model)

    if not vsm.collection_exists(args.collection):
        logger.error("Collection '%s' not found. Run indexing first:\n"
                      "  python main.py --mode chat --reindex --pdf_dir data", args.collection)
        sys.exit(1)

    index = indexer.load_index(args.collection)
    engine = RAGEngine(
        index=index,
        llm=llm,
        system_prompt=settings.system_prompt_chat,
        top_k=args.top_k,
        conversational=True,
    )
    logger.info("RAG engine ready (LLM: %s | Embedding: %s | Collection: %s)",
                args.model, args.embedding, args.collection)

    # --- TTS setup (optional) ---
    tts_model = None
    if args.speak:
        from voice_agent.tts import DEFAULT_MALE, DEFAULT_FEMALE
        tts_model = DEFAULT_MALE if args.voice == "male" else DEFAULT_FEMALE
        logger.info("TTS enabled (voice: %s, model: %s)", args.voice, tts_model)

    # --- Streaming transcription with RAG ---
    client = create_deepgram_client(transcriber_settings.stt)
    # Use longer endpointing (1500ms) so Deepgram waits for the user to
    # finish their full question before finalizing the transcript.
    connect_opts = get_connect_options(transcriber_settings.stt, endpointing_ms=1500)

    logger.info("\nListening... speak your question (say 'exit' or Ctrl+C to stop)\n")

    # Lock to prevent overlapping RAG queries
    query_lock = threading.Lock()
    stop_event = threading.Event()

    # Buffer to accumulate transcript fragments before sending to RAG.
    # A timer fires after BUFFER_WAIT_SEC of silence to flush the buffer.
    BUFFER_WAIT_SEC = 2.0
    _buffer_lock = threading.Lock()
    _buffer_fragments: list[str] = []
    _buffer_timer: threading.Timer | None = None

    def _flush_buffer() -> None:
        """Combine buffered fragments into a single query and send to RAG."""
        with _buffer_lock:
            if not _buffer_fragments:
                return
            full_question = " ".join(_buffer_fragments)
            _buffer_fragments.clear()

        # Check for exit command
        if full_question.strip().lower() in ("exit", "exit."):
            print("\nExiting voice RAG. Goodbye!")
            stop_event.set()
            return

        print(f"\nYou: {full_question}")

        def _run_query():
            with query_lock:
                try:
                    result = engine.query(full_question)
                    print(f"\nAssistant: {result.answer}\n")

                    # Speak the answer if TTS is enabled
                    if tts_model and result.answer.strip():
                        _speak_answer(result.answer, tts_model,
                                      transcriber_settings.stt.api_key)
                except Exception as e:
                    logger.error("RAG query failed: %s", e)

                print("Listening... speak your next question\n")

        threading.Thread(target=_run_query, daemon=True).start()

    def _on_message(result: object) -> None:
        """Buffer final transcripts and flush after a silence gap.

        Accumulates transcript fragments and starts a timer. If no new
        fragment arrives within ``BUFFER_WAIT_SEC``, the buffer is flushed
        as a single query to the RAG engine.

        Args:
            result: A Deepgram SDK message object.
        """
        nonlocal _buffer_timer

        if stop_event.is_set():
            return
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

        with _buffer_lock:
            _buffer_fragments.append(transcript)

            # Reset the flush timer
            if _buffer_timer is not None:
                _buffer_timer.cancel()
            _buffer_timer = threading.Timer(BUFFER_WAIT_SEC, _flush_buffer)
            _buffer_timer.daemon = True
            _buffer_timer.start()

    with client.listen.v1.connect(**connect_opts) as connection:
        connection.on(EventType.MESSAGE, _on_message)
        connection.on(EventType.ERROR, lambda err: (
            logger.error("Deepgram error: %s", err) if not stop_event.is_set() else None
        ))

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
                status: Stream status flags.
            """
            if stop_event.is_set():
                return
            if status:
                logger.warning("Audio status: %s", status)
            try:
                audio_bytes = (indata[:, 0] * 32767).astype(np.int16).tobytes()
                connection.send_media(audio_bytes)
            except Exception:
                pass

        try:
            with sd.InputStream(
                samplerate=transcriber_settings.sample_rate,
                channels=transcriber_settings.channels,
                dtype="float32",
                callback=_audio_callback,
                blocksize=int(transcriber_settings.sample_rate * 0.1),
            ):
                connection.start_listening()
                stop_event.wait()
        except KeyboardInterrupt:
            pass
        finally:
            logger.info("\nStopping voice RAG...")
            try:
                connection.send_finalize()
            except Exception:
                pass  # connection may already be closed
            engine.reset()


def _speak_answer(text: str, model: str, api_key: str) -> None:
    """Synthesize and play the RAG answer using Deepgram TTS.

    Args:
        text: The answer text to speak.
        model: Deepgram Aura-2 voice model identifier.
        api_key: Deepgram API key.
    """
    from voice_agent.tts import synthesize

    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        synthesize(text, tmp_path, model, api_key)
        _play_audio_file(str(tmp_path))
    except Exception as e:
        logger.error("TTS playback failed: %s", e)
    finally:
        try:
            tmp_path.unlink()
        except Exception:
            pass


if __name__ == "__main__":
    main()
