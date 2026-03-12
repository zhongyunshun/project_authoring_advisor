"""LiveKit Voice Agent — pluggable voice interface for RAG backends.

Architecture

  ┌─────────────┐    WebRTC     ┌──────────────────────────────┐
  │  User/Phone │◄────────────►│  LiveKit Server (:7880/7881) │
  └─────────────┘               └──────────┬───────────────────┘
                                           │
                                ┌──────────▼───────────────────┐
                                │     voice_agent/app.py       │
                                │  ┌─────┐ ┌───┐ ┌───┐ ┌───┐  │
                                │  │ VAD │ │STT│ │LLM│ │TTS│  │
                                │  │Silero│ │ * │ │ * │ │ * │  │
                                │  └─────┘ └───┘ └─┬─┘ └───┘  │
                                │                   │          │
                                │  ┌────────────────▼───────┐  │
                                │  │  VoiceAgent            │  │
                                │  │  → query_backend tool  │  │
                                │  └────────────┬───────────┘  │
                                └───────────────┼──────────────┘
                                                │
                           ┌────────────────────┼────────────────────┐
                           │ BackendAdapter      │                    │
                      ┌────▼─────┐        ┌─────▼──────┐    ┌───────▼───────┐
                      │HTTPBackend│        │LocalBackend│    │ Your adapter  │
                      │→ REST API │        │→ in-process│    │→ anything     │
                      └────┬─────┘        └────────────┘    └───────────────┘
                           │
                      ┌────▼──────────────┐
                      │ backend_api.py    │
                      │ FastAPI wrapper   │
                      │ around RAGEngine  │
                      └───────────────────┘

  Key Design Decisions

  - Independent of your RAG system — the voice agent talks to backends through the BackendAdapter interface. Swap implementations to connect to any RAG system.
  - Two adapter options included:
    - HTTPBackendAdapter — calls a REST API (default, for separate deployments)
    - LocalBackendAdapter — calls your RAGEngine/RAGAgent directly in-process
  - Multilingual STT/TTS — Deepgram Nova-3 with language="multi" auto-detects language; faster-whisper large-v3 supports 99+ languages for local deployment
  - Pluggable providers — stt_factory.py and tts_factory.py let you switch between cloud (Deepgram/Cartesia/ElevenLabs) and local (faster-whisper/Piper) via env vars

  Running It

  Option 1: Docker Compose (full stack)
  cp voice_agent/.env.example voice_agent/.env
  # Edit .env with your API keys
  docker compose -f docker-compose.voice.yml up

  Option 2: Local development
  pip install -r voice_agent/requirements.txt

  # Terminal 1: LiveKit server
  docker run --rm -p 7880:7880 -p 7881:7881 livekit/livekit-server --dev

  # Terminal 2: RAG backend API
  uvicorn voice_agent.backend_api:app --port 8000

  # Terminal 3: Voice agent
  python -m voice_agent.app

  Option 3: In-process (no HTTP)
  from voice_agent.adapters.local_backend import LocalBackendAdapter
  from voice_agent.agent import VoiceAgent

  agent = VoiceAgent(backend=LocalBackendAdapter(your_rag_engine))

  Plugging Into Another RAG System

  Implement BackendAdapter with a single method:

  class MyAdapter(BackendAdapter):
      async def process(self, query: BackendQuery) -> BackendResponse:
          result = await my_rag.ask(query.query)
          return BackendResponse(answer=result)

  Then pass it to VoiceAgent(backend=MyAdapter()).

"""
