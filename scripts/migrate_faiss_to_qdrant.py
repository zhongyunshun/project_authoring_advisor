"""One-time migration utility: FAISS vector stores -> Qdrant.

Reads existing FAISS indices (created by old LangChain code),
extracts the documents, and re-indexes them into Qdrant.

Usage:
    python scripts/migrate_faiss_to_qdrant.py --faiss_dir vector_db/ds_chunk_700_embedding \
        --collection trca_documents --embedding huggingface
"""

import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.embedding_factory import EmbeddingFactory
from core.vector_store import VectorStoreManager
from ingest.chunker import DocumentChunker


def migrate(faiss_dir: str, collection_name: str, embedding_provider: str):
    """Load a FAISS index via LangChain and re-index into Qdrant."""
    # Use old LangChain code for loading
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_openai import OpenAIEmbeddings

    if embedding_provider in ("huggingface", "sbert", "ds"):
        lc_embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    else:
        lc_embedding = OpenAIEmbeddings()

    print(f"Loading FAISS index from {faiss_dir}...")
    faiss_db = FAISS.load_local(faiss_dir, lc_embedding, allow_dangerous_deserialization=True)

    # Extract all documents from FAISS
    docstore = faiss_db.docstore
    all_docs = list(docstore._dict.values())
    print(f"Found {len(all_docs)} documents in FAISS index")

    # Convert LangChain Documents to LlamaIndex TextNodes
    from llama_index.core.schema import TextNode

    nodes = []
    for doc in all_docs:
        node = TextNode(
            text=doc.page_content,
            metadata=doc.metadata or {},
        )
        nodes.append(node)

    # Create Qdrant index
    embed_model = EmbeddingFactory.create(provider=embedding_provider)
    vsm = VectorStoreManager(storage_path="./vector_db/qdrant_storage")

    print(f"Creating Qdrant collection '{collection_name}' with {len(nodes)} nodes...")
    vsm.create_index(collection_name, nodes, embed_model)
    print("Migration complete!")


def main():
    parser = argparse.ArgumentParser(description="Migrate FAISS to Qdrant")
    parser.add_argument("--faiss_dir", required=True, help="Path to FAISS index directory")
    parser.add_argument("--collection", default="trca_documents", help="Qdrant collection name")
    parser.add_argument("--embedding", default="huggingface",
                        choices=["openai", "huggingface", "sbert", "ds"])
    args = parser.parse_args()

    migrate(args.faiss_dir, args.collection, args.embedding)


if __name__ == "__main__":
    main()
