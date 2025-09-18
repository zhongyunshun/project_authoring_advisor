from embeddings.embeddings import load_vector_db
from config.keys import OPENAI_API_KEY
import os

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

loaded_vector_db = load_vector_db("vector_db/pdf_faiss_index")

retriever = loaded_vector_db.as_retriever(search_kwargs={"k": 5})
retrieved_docs = retriever.invoke("Who conducted a geotechnical investigation at 21 Peacham Crescent to assess the existing soil conditions?")

# Print content and metadata
for i, doc in enumerate(retrieved_docs):
    print(f"\n--- Document {i + 1} ---")
    print(f"Content:\n{doc.page_content[:300]}...")  # Print first 300 chars
    print("Metadata:")
    for key, value in doc.metadata.items():
        print(f"  {key}: {value}")

