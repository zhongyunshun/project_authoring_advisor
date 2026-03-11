import os
import tempfile
from typing import List

from llama_index.core.schema import Document


class PDFLoader:
    """Loads PDF documents using LlamaIndex readers."""

    @staticmethod
    def load_directory(path: str) -> List[Document]:
        """Load all PDFs from a directory, preserving source metadata."""
        from llama_index.readers.file import PyMuPDFReader

        reader = PyMuPDFReader()
        all_docs = []

        for filename in os.listdir(path):
            if not filename.lower().endswith(".pdf"):
                continue
            filepath = os.path.join(path, filename)
            try:
                docs = reader.load_data(file_path=filepath)
                for doc in docs:
                    doc.metadata["source"] = filename
                all_docs.extend(docs)
            except Exception as e:
                print(f"Failed to load {filename}: {e}")

        return all_docs

    @staticmethod
    def load_uploaded_file(uploaded_file) -> List[Document]:
        """Load a Streamlit UploadedFile object."""
        from llama_index.readers.file import PyMuPDFReader

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        try:
            reader = PyMuPDFReader()
            docs = reader.load_data(file_path=tmp_path)
            for doc in docs:
                doc.metadata["source"] = uploaded_file.name
            return docs
        finally:
            os.unlink(tmp_path)
