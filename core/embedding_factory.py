from llama_index.core.embeddings import BaseEmbedding


class EmbeddingFactory:
    """Creates LlamaIndex embedding model instances from a provider string."""

    @staticmethod
    def create(provider: str, model_name: str = "") -> BaseEmbedding:
        provider = provider.lower()

        if provider == "openai":
            from llama_index.embeddings.openai import OpenAIEmbedding

            return OpenAIEmbedding(model_name=model_name or "text-embedding-3-small")

        elif provider in ("huggingface", "sbert", "sentencebert", "ds"):
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding

            return HuggingFaceEmbedding(
                model_name=model_name or "sentence-transformers/all-MiniLM-L6-v2"
            )

        else:
            raise ValueError(f"Unsupported embedding provider: '{provider}'")
