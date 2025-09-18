from embeddings.embeddings import create_vector_db_from_pdfs, save_vector_db, load_vector_db
from config.keys import OPENAI_API_KEY
import os

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Step 1: Generate and save
pdf_folder = "data/ALL_PDFS"
vector_db = create_vector_db_from_pdfs(pdf_folder)
save_vector_db(vector_db, "pdf_faiss_index")

# # Step 2: Load later
# loaded_vector_db = load_vector_db("pdf_faiss_index")
