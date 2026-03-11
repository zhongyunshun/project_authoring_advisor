import os
from dataclasses import dataclass, field


@dataclass
class Settings:
    """Central configuration for the RAG application."""

    # API keys (read from environment or config)
    openai_api_key: str = ""
    google_api_key: str = ""
    tavily_api_key: str = ""

    # LLM defaults
    llm_provider: str = "openai"  # openai | gemini | llama_cpp
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 1024

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
        if self.tavily_api_key:
            os.environ["TAVILY_API_KEY"] = self.tavily_api_key

    @classmethod
    def from_env(cls) -> "Settings":
        """Build settings from environment variables."""
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            google_api_key=os.getenv("GOOGLE_API_KEY", ""),
            tavily_api_key=os.getenv("TAVILY_API_KEY", ""),
        )
