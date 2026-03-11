from llama_index.core import VectorStoreIndex
from llama_index.core.embeddings import BaseEmbedding

from core.vector_store import VectorStoreManager
from ingest.pdf_loader import PDFLoader
from ingest.chunker import DocumentChunker


class Indexer:
    """Orchestrates the full ingestion pipeline: load -> chunk -> embed -> store."""

    def __init__(
        self,
        vector_store_manager: VectorStoreManager,
        embed_model: BaseEmbedding,
        chunk_size: int = 700,
        chunk_overlap: int = 50,
    ):
        self._vsm = vector_store_manager
        self._embed_model = embed_model
        self._chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def index_directory(self, pdf_dir: str, collection_name: str) -> VectorStoreIndex:
        """Load all PDFs from a directory, chunk, embed, and store in Qdrant."""
        docs = PDFLoader.load_directory(pdf_dir)
        if not docs:
            raise ValueError(f"No PDF documents found in {pdf_dir}")

        nodes = self._chunker.chunk(docs)
        print(f"Loaded {len(docs)} documents, created {len(nodes)} chunks")

        return self._vsm.create_index(collection_name, nodes, self._embed_model)

    def index_uploaded_file(self, uploaded_file, collection_name: str) -> VectorStoreIndex:
        """Handle a Streamlit uploaded PDF: chunk, embed, add to existing collection."""
        docs = PDFLoader.load_uploaded_file(uploaded_file)
        nodes = self._chunker.chunk(docs)

        if self._vsm.collection_exists(collection_name):
            return self._vsm.add_nodes(collection_name, nodes, self._embed_model)
        else:
            return self._vsm.create_index(collection_name, nodes, self._embed_model)

    def load_index(self, collection_name: str) -> VectorStoreIndex:
        """Load an existing index from Qdrant."""
        return self._vsm.get_index(collection_name, self._embed_model)
