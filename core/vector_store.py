from typing import List, Optional

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.schema import TextNode
from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore


class VectorStoreManager:
    """Manages Qdrant collections in local file-storage mode (no server required)."""

    def __init__(self, storage_path: str = "./vector_db/qdrant_storage"):
        self._storage_path = storage_path
        self._client = QdrantClient(path=storage_path)

    @property
    def client(self) -> QdrantClient:
        return self._client

    def get_vector_store(self, collection_name: str) -> QdrantVectorStore:
        return QdrantVectorStore(
            client=self._client,
            collection_name=collection_name,
        )

    def get_index(
        self,
        collection_name: str,
        embed_model: BaseEmbedding,
    ) -> VectorStoreIndex:
        """Load an existing Qdrant collection as a VectorStoreIndex."""
        vector_store = self.get_vector_store(collection_name)
        return VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=embed_model,
        )

    def create_index(
        self,
        collection_name: str,
        nodes: List[TextNode],
        embed_model: BaseEmbedding,
    ) -> VectorStoreIndex:
        """Create a new Qdrant collection from nodes and return the index."""
        vector_store = self.get_vector_store(collection_name)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        return VectorStoreIndex(
            nodes=nodes,
            embed_model=embed_model,
            storage_context=storage_context,
        )

    def add_nodes(
        self,
        collection_name: str,
        nodes: List[TextNode],
        embed_model: BaseEmbedding,
    ) -> VectorStoreIndex:
        """Add nodes to an existing collection (incremental upsert)."""
        vector_store = self.get_vector_store(collection_name)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=embed_model,
        )
        index.insert_nodes(nodes)
        return index

    def collection_exists(self, collection_name: str) -> bool:
        try:
            collections = self._client.get_collections().collections
            return any(c.name == collection_name for c in collections)
        except Exception:
            return False
