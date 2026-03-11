from typing import List

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document, TextNode


class DocumentChunker:
    """Splits documents into text nodes using sentence-aware chunking."""

    def __init__(self, chunk_size: int = 700, chunk_overlap: int = 50):
        self._parser = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def chunk(self, documents: List[Document]) -> List[TextNode]:
        return self._parser.get_nodes_from_documents(documents)
