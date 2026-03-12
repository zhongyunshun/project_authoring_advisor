import os
from dataclasses import dataclass, field


@dataclass
class Settings:
    """Central configuration for the RAG application."""

    # API keys (read from environment or config)
    openai_api_key: str = ""
    google_api_key: str = ""
    anthropic_api_key: str = ""
    tavily_api_key: str = ""

    # LLM defaults
    llm_provider: str = "openai"  # openai | gemini | claude | llama_cpp
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 8192

    # Available models per provider (for reference / UI dropdowns)
    AVAILABLE_MODELS: dict = None

    def __post_init__(self):
        if self.AVAILABLE_MODELS is None:
            self.AVAILABLE_MODELS = {
                "openai": [
                    "o3-2025-04-16",
                    "gpt-4.1-mini",
                    "gpt-4.1",
                    "gpt-4.1-nano",
                    "gpt-4.5-preview",
                    "o4-mini",
                    "o3-mini",
                ],
                "gemini": [
                    "models/gemini-2.5-flash",
                    "models/gemini-2.5-pro",
                    "models/gemini-2.0-flash",
                ],
                "claude": [
                    "claude-sonnet-4-6",
                    "claude-opus-4-6",
                    "claude-haiku-4-5-20251001",
                ],
            }

    # Local model settings
    model_path: str = ""
    n_ctx: int = 8192

    # Embedding defaults
    embedding_provider: str = "openai"  # openai | huggingface
    embedding_model: str = ""  # empty = use provider default

    # Qdrant
    qdrant_path: str = "./vector_db/qdrant_storage"
    collection_name: str = "trca_documents"

    # Retrieval
    top_k: int = 22
    chunk_size: int = 700
    chunk_overlap: int = 50

    # System prompts
    system_prompt_chat: str = "You are a helpful assistant with memory. Answer questions accordingly."
    system_prompt_csv: str = (
        "Please generate the correct answer for the given fill-in-the-blank question. "
        "Avoid including unnecessary context, restating the question, or adding explanations"
        "—only return the precise answer."
    )

    def apply_env(self):
        """Push API keys to environment variables for SDK auto-detection."""
        if self.openai_api_key:
            os.environ["OPENAI_API_KEY"] = self.openai_api_key
        if self.google_api_key:
            os.environ["GOOGLE_API_KEY"] = self.google_api_key
        if self.anthropic_api_key:
            os.environ["ANTHROPIC_API_KEY"] = self.anthropic_api_key
        if self.tavily_api_key:
            os.environ["TAVILY_API_KEY"] = self.tavily_api_key

    @classmethod
    def from_env(cls) -> "Settings":
        """Load .env file, then build settings from environment variables."""
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass

        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            google_api_key=os.getenv("GOOGLE_API_KEY", ""),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            tavily_api_key=os.getenv("TAVILY_API_KEY", ""),
        )
