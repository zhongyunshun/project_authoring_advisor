# Voice Agent

Real-time speech-to-text (STT), text-to-speech (TTS), and voice-powered RAG using [Deepgram](https://deepgram.com/) APIs. The voice agent can operate standalone (transcription and TTS) or integrated with the project's RAG system for voice-driven Q&A over indexed documents.

## Architecture

```
voice_agent/
├── config.py          # Configuration (STTConfig, TranscriberSettings)
├── stt_factory.py     # Deepgram live transcription client factory
├── app.py             # Standalone real-time mic transcription
├── tts.py             # Text-to-speech CLI tool
├── voice_rag.py       # Voice-powered RAG (STT → RAG engine → answer)
├── __init__.py        # Module docstring with usage examples
├── requirements.txt   # Python dependencies
├── .env.example       # Environment variable template
├── .env               # Your API keys (not committed)
├── transcripts/       # Auto-generated transcript files (from app.py)
└── tts/
    ├── outputs/       # Auto-generated TTS audio files
    └── *.txt          # Input text files for TTS
```

### Data Flow

**Standalone transcription (`app.py`):**
```
Microphone → sounddevice → Deepgram Streaming API → Terminal + transcript file
```

**Text-to-speech (`tts.py`):**
```
Text file → Deepgram Aura-2 TTS API → Audio file (.wav/.mp3/.ogg)
```

**Voice RAG (`voice_rag.py`):**
```
Microphone → sounddevice → Deepgram STT → [buffer fragments] → RAGEngine.query()
                                                                      ↓
                                            Terminal ← answer ← LLM + Qdrant retrieval
                                                ↓ (optional)
                                          Deepgram TTS → Speaker
```

## Setup

### 1. Install dependencies

```bash
pip install -r voice_agent/requirements.txt
```

### 2. Configure API keys

Copy the example and add your Deepgram API key:

```bash
cp voice_agent/.env.example voice_agent/.env
# Edit voice_agent/.env and set DEEPGRAM_API_KEY
```

### 3. Index documents (required for voice RAG only)

If you haven't already indexed your PDFs:

```bash
python main.py --mode chat --reindex --pdf_dir data --embedding huggingface
```

## Usage

### Standalone Transcription (no RAG)

Captures audio from your microphone and transcribes in real-time. Transcripts are printed to the terminal and saved to `voice_agent/transcripts/`.

```bash
python -m voice_agent.app
```

- Press **Ctrl+C** to stop.
- Output file: `voice_agent/transcripts/transcript_YYYYMMDD_HHMMSS.txt`
- Supports multilingual transcription with `DEEPGRAM_LANGUAGE=multi` (default).

### Text-to-Speech (no RAG)

Converts a text file to an audio file using Deepgram Aura-2 voices. Output is saved to `voice_agent/tts/outputs/`.

```bash
# Female voice (default)
python -m voice_agent.tts voice_agent/tts/tiny_story1.txt -o output.wav

# Male voice
python -m voice_agent.tts voice_agent/tts/tiny_story1.txt -o output.wav --voice male

# Specific voice model
python -m voice_agent.tts voice_agent/tts/tiny_story1.txt -o output.wav --model aura-2-zeus-en

# MP3 output
python -m voice_agent.tts voice_agent/tts/tiny_story1.txt -o output.mp3

# List all available voices
python -m voice_agent.tts --list-voices
```

Supported output formats: `.wav`, `.mp3`, `.ogg`.

### Voice RAG (with RAG system)

Speak questions into the microphone and get answers from the RAG system. Requires an existing Qdrant index.

```bash
# Basic usage
python -m voice_agent.voice_rag --model openai --embedding huggingface

# With spoken answers (TTS reads the answer back)
python -m voice_agent.voice_rag --model openai --embedding huggingface --speak

# With male TTS voice
python -m voice_agent.voice_rag --model openai --embedding huggingface --speak --voice male

# Different LLM provider
python -m voice_agent.voice_rag --model gemini --embedding huggingface

# Custom collection and retrieval settings
python -m voice_agent.voice_rag --model openai --embedding huggingface --collection trca_documents --top_k 10
```

- Say **"exit"** or press **Ctrl+C** to stop.
- The `--embedding` flag must match the embedding model used during indexing (e.g. `huggingface` if indexed with HuggingFace, `openai` if indexed with OpenAI).
- The voice RAG uses a 1500ms endpointing delay and a 2-second fragment buffer, so you can pause briefly mid-sentence without the question being split.

#### Voice RAG CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `openai` | LLM provider (`openai`, `gemini`, `claude`) |
| `--model_name` | _(provider default)_ | Specific model name (e.g. `gpt-4o-mini`) |
| `--embedding` | `openai` | Embedding provider (`openai`, `huggingface`) |
| `--collection` | `trca_documents` | Qdrant collection name |
| `--top_k` | `22` | Number of chunks to retrieve |
| `--speak` | off | Enable TTS to speak answers aloud |
| `--voice` | `female` | TTS voice gender (`male`, `female`) |

## Module Reference

### `config.py` — Configuration

Loads settings from `voice_agent/.env` and environment variables using `python-dotenv`.

| Class / Function | Description |
|------------------|-------------|
| `STTConfig` | Frozen dataclass holding Deepgram STT settings: `api_key`, `model` (default `nova-3`), and `language` (default `multi` for auto-detection). |
| `TranscriberSettings` | Top-level frozen dataclass holding `sample_rate` (default 16000), `channels` (default 1), and an `STTConfig` instance. |
| `TranscriberSettings.from_env()` | Class method that builds a `TranscriberSettings` from environment variables (`DEEPGRAM_API_KEY`, `DEEPGRAM_MODEL`, `DEEPGRAM_LANGUAGE`, `SAMPLE_RATE`, `CHANNELS`). |
| `get_settings()` | Returns a cached singleton `TranscriberSettings` instance. Called by all modules to access configuration. |

### `stt_factory.py` — Deepgram Client Factory

Creates and configures Deepgram SDK v6 clients for live transcription.

| Function | Description |
|----------|-------------|
| `create_deepgram_client(config: STTConfig) -> DeepgramClient` | Instantiates an authenticated `DeepgramClient` using the API key from config. |
| `get_connect_options(config: STTConfig, endpointing_ms: int = 300) -> dict[str, str]` | Builds the keyword arguments dictionary for `client.listen.v1.connect()`. Configures model, language, encoding (linear16), sample rate, punctuation, smart formatting, and endpointing delay. The `endpointing_ms` parameter controls how long Deepgram waits after silence before finalizing an utterance — `app.py` uses 300ms (responsive), `voice_rag.py` uses 1500ms (avoids splitting questions). |

### `app.py` — Standalone Transcription

Real-time microphone transcription that prints to terminal and saves to file.

| Function | Description |
|----------|-------------|
| `_on_message(result: object) -> None` | Callback for Deepgram message events. Filters for `ListenV1Results` with `is_final=True`, extracts the transcript text and detected language, prints to stdout, and appends to the transcript file. Ignores metadata, utterance end, and speech started events. |
| `main() -> None` | Entry point. Validates the API key, creates a timestamped output file in `voice_agent/transcripts/`, opens a Deepgram websocket connection, starts a `sounddevice.InputStream` that captures mic audio in 100ms blocks and sends raw int16 bytes to Deepgram via `connection.send_media()`. The audio stream starts before `connection.start_listening()` to prevent Deepgram timeout. Blocks until `Ctrl+C`. |

### `tts.py` — Text-to-Speech

Converts text files to audio using Deepgram Aura-2 TTS API.

| Constant | Description |
|----------|-------------|
| `MALE_VOICES` | List of 17 Aura-2 male English voice model IDs. |
| `FEMALE_VOICES` | List of 22 Aura-2 female English voice model IDs. |
| `DEFAULT_MALE` | `"aura-2-orion-en"` — default male voice. |
| `DEFAULT_FEMALE` | `"aura-2-asteria-en"` — default female voice. |

| Function | Description |
|----------|-------------|
| `list_voices() -> None` | Prints all available voice models to stdout, grouped by gender, with defaults marked. |
| `synthesize(text: str, output_path: Path, model: str, api_key: str) -> None` | Sends text to `client.speak.v1.audio.generate()` and writes the returned audio chunks to `output_path`. Automatically selects encoding/container based on file extension: `.wav` → linear16/wav, `.mp3` → mp3, `.ogg` → opus/ogg. |
| `main() -> None` | CLI entry point. Parses arguments (`input`, `-o`, `--voice`, `--model`, `--list-voices`), reads the input text file, resolves the voice model, and calls `synthesize()`. Output defaults to `voice_agent/tts/outputs/<input_name>.wav`. |

### `voice_rag.py` — Voice-Powered RAG

Connects the STT pipeline to the project's RAG system for voice-driven Q&A.

| Function | Description |
|----------|-------------|
| `_play_audio_file(file_path: str, sample_rate: int) -> None` | Opens a WAV file, reads its frames into a numpy array, and plays it through the default audio output using `sounddevice.play()`. Used internally for TTS playback. |
| `_speak_answer(text: str, model: str, api_key: str) -> None` | Synthesizes the RAG answer to a temporary WAV file via `tts.synthesize()`, plays it with `_play_audio_file()`, then deletes the temp file. |
| `main() -> None` | Full pipeline entry point. Performs the following setup steps: **(1)** Loads Deepgram config from `voice_agent/.env`. **(2)** Loads RAG settings from the project's `.env`/`config/keys.py`. **(3)** Creates an LLM via `LLMFactory.create()`, embedding model via `EmbeddingFactory.create()`, and loads the Qdrant index. **(4)** Instantiates a conversational `RAGEngine` with chat memory. **(5)** Opens a Deepgram streaming connection with 1500ms endpointing. **(6)** Starts mic capture via `sounddevice`. |
| `_flush_buffer() -> None` | (Defined inside `main`) Combines accumulated transcript fragments into a single question string. If the question is `"exit"`, sets the stop event and shuts down. Otherwise, spawns a background thread that calls `engine.query(full_question)`, prints the answer, and optionally speaks it via TTS. A `threading.Lock` prevents overlapping RAG queries. |
| `_on_message(result: object) -> None` | (Defined inside `main`) Callback for Deepgram events. Filters for final `ListenV1Results`, appends the transcript to a fragment buffer, and resets a 2-second timer. When the timer fires (no new speech for 2 seconds), `_flush_buffer()` is called to send the complete question to the RAG engine. This buffering prevents partial questions from triggering separate queries. |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DEEPGRAM_API_KEY` | _(required)_ | Deepgram API key for STT and TTS |
| `DEEPGRAM_MODEL` | `nova-3` | Deepgram STT model |
| `DEEPGRAM_LANGUAGE` | `multi` | Language code (`en`, `fr`, `multi` for auto-detect) |
| `SAMPLE_RATE` | `16000` | Audio sample rate in Hz |
| `CHANNELS` | `1` | Number of audio input channels |

## TTS Voice Models

The TTS uses Deepgram's **Aura-2** generation (English only). Run `python -m voice_agent.tts --list-voices` to see all 39 available voices.

| Gender | Default | Total Available |
|--------|---------|-----------------|
| Female | `aura-2-asteria-en` | 22 voices |
| Male | `aura-2-orion-en` | 17 voices |

## Dependencies

```
deepgram-sdk>=4.0.0    # Deepgram STT and TTS APIs
sounddevice>=0.5.0     # Microphone audio capture
numpy>=1.24.0          # Audio data conversion
python-dotenv>=1.0.0   # .env file loading
```

Voice RAG (`voice_rag.py`) additionally requires the project's main dependencies (LlamaIndex, Qdrant, etc.) from the root `requirements.txt`.
