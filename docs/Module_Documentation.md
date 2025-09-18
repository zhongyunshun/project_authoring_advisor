# **Function Documentation**

## `answer_question(question, rag_model)`
   - **Module**: `main.py`
   - **Purpose**: Answers a single question using the provided RAG model.
   - **Parameters**:
     - `question` (str): The question to be answered.
     - `rag_model` (RAG model object): The RAG model used to retrieve relevant information and generate the response.
   - **Usage**:
     ```python
     answer_question("What is TRCA?", rag_model)
     # prints 💡 **Answer**: ...
     ```

## `process_questions(csv_file, rag_model, output_csv_file)`
   - **Module**: `main.py`
   - **Purpose**: Processes a CSV file containing questions, answers them using the RAG model, and saves the results.
   - **Parameters**:
     - `csv_file` (str): Path to the CSV file containing questions.
     - `rag_model` (RAG model object): The model used for answering questions.
     - `output_csv_file` (str): Output path for saving answers.
   - **Usage**:
     ```python
     process_questions("questions.csv", rag_model, "output.csv")
     # Save processed dataframe to csv
     ```

## `ConversationalRAGChat(rag_model)`
   - **Module**: `main.py`
   - **Purpose**: Provides an interactive chatbot for real-time Q&A.
   - **Parameters**:
     - `rag_model` (ConversationalRAG object): The model used for interactive Q&A.
   - **Usage**:
     ```python
     ConversationalRAGChat(rag_model)
     ```

## `save_embeddings_to_database_pickel(chunks, save_path)`
   - **Module**: `embeddings/database.py`
   - **Purpose**: Not used for now. Saves embeddings to a database using pickle.
   - **Parameters**:
     - `chunks` (list): Data chunks to be embedded.
     - `save_path` (str, optional): Path to save the database file (default: `"./embeddings/vector_db.pkl"`).
   - **Usage**:
     ```python
     save_embeddings_to_database_pickel(chunks)
     # return vector_db
     ```

## `load_embeddings_from_database_pickel(save_path)`
   - **Module**: `embeddings/database.py`
   - **Purpose**: Not used for now. Loads embeddings from a pickle database.
   - **Parameters**:
     - `save_path` (str, optional): Path to the database file (default: `"./embeddings/vector_db.pkl"`).
   - **Usage**:
     ```python
     embeddings = load_embeddings_from_database_pickel()
     # return vector_db
     ```

## `save_embeddings_to_database(chunks)`
   - **Module**: `embeddings/embeddings.py`
   - **Purpose**: Saves embeddings to a database.
   - **Parameters**:
     - `chunks` (list): Data chunks to be embedded.
   - **Usage**:
     ```python
     save_embeddings_to_database(chunks)
     # return vector_db
     ```

## `save_embeddings_to_file(vector_db, filename)`
   - **Module**: `embeddings/embeddings.py`
   - **Purpose**: Saves embeddings to a file.
   - **Parameters**:
     - `vector_db` (object): The vector database containing embeddings.
     - `filename` (str): File path to save embeddings.
   - **Usage**:
     ```python
     save_embeddings_to_file(vector_db, "folder_name")
     # Vector db saved to folder
     ```

## `load_embeddings_from_file(filename)`
   - **Module**: `embeddings/embeddings.py`
   - **Purpose**: Loads embeddings from a file.
   - **Parameters**:
     - `filename` (str): File path to load embeddings.
   - **Usage**:
     ```python
     embeddings = load_embeddings_from_file("embeddings_folder_name")
     # return faiss_index
     ```

## `generate_and_save_embeddings(chunks, embedding_file)`
   - **Module**: `embeddings/embeddings.py`
   - **Purpose**: Generates and saves embeddings.
   - **Parameters**:
     - `chunks` (list): Data chunks to embed.
     - `embedding_file` (str): File path to save embeddings.
   - **Usage**:
     ```python
     generate_and_save_embeddings(chunks, "embeddings_folder")
     # Embedding saved to folder
     # return vector_db
     ```

## `calculate_bleu(csv_file, n=3)`
   - **Module**: `evaluation/bleu.py`
   - **Purpose**: Calculates the BLEU score for text evaluation.
   - **Parameters**:
     - `csv_file` (str): Path to CSV file containing text data.
     - `n` (int, optional): N-gram size for BLEU calculation (default: 3).
   - **Usage**:
     ```python
     score = calculate_bleu("test_data.csv", n=4)
     # return avg_bleu (float), scores (list)
     ```

## `evaluate_with_retry(dataset, llm)`
   - **Module**: `evaluation/folder_eval_ragas.py`
   - **Purpose**: Retries evaluation to handle API rate limits.
   - **Parameters**:
     - `dataset` (object): The dataset to evaluate.
     - `llm` (object): The language model used for evaluation.
   - **Usage**:
     ```python
     results = evaluate_with_retry(dataset, llm)
     ```

## `evaluate_ragas(csv_file)`
   - **Module**: `evaluation/folder_eval_ragas.py`
   - **Purpose**: Evaluates text using the RAGAS framework.
   - **Parameters**:
     - `csv_file` (str): Path to CSV file containing evaluation data.
   - **Usage**:
     ```python
     results = evaluate_ragas("evaluation.csv")
     # RAGAS evaluation results saved to csv file
     ```

## `pdf_to_text(pdf_path)`
   - **Module**: `preprocessing/pdf_processing.py`
   - **Purpose**: Converts a PDF file to text.
   - **Usage**:
     ```python
     text = pdf_to_text("document.pdf")
     # return string of text
     ```

## `pdfs_to_text(pdf_directory)`
   - **Module**: `preprocessing/pdf_processing.py`
   - **Purpose**: Converts all PDFs in a directory to text.
   - **Parameters**:
     - `pdf_directory` (str): Path to the directory containing PDFs.
   - **Usage**:
     ```python
     texts = pdfs_to_text("pdf_folder")
     # return string of all pdf texts combined with \n
     ```

## `split_text_into_chunks_nltk(text, chunk_size=200)`
   - **Module**: `preprocessing/text_splitter.py`
   - **Purpose**: Splits text into chunks using NLTK.
   - **Parameters**:
     - `text` (str): The input text.
     - `chunk_size` (int, optional): Maximum chunk size (default: 200).
   - **Usage**:
     ```python
     chunks = split_text_into_chunks_nltk(text, chunk_size=300)
     # return list of chunks
     ```

## `save_text_to_file(text, output_file_path)`
   - **Module**: `preprocessing/save_file.py`
   - **Purpose**: Saves text to a file.
   - **Parameters**:
     - `text` (str): The text to be saved.
     - `output_file_path` (str): Path to save the file.
   - **Usage**:
     ```python
     save_text_to_file("Hello, World!", "output.txt")
     ```

# Utility Scripts

## `folder_eval`
   - **Purpose**: Evaluates the performance of RAG Pipeline outputs using BLEU and ROUGE scores.
   - **Usage**:
     ```bash
     python evaluation/folder_eval.py
     # saved evaluation scores to csv
     ```

## `folder_eval_ragas`
   - **Purpose**: Evaluates the performance of RAG Pipeline outputs using RAGAS.
   - **Usage**:
     ```bash
     python -m evaluation/folder_eval_ragas
     # saved evaluation scores to csv
     ```

## `pdf_extractor`
   - **Purpose**: Extracts text content from PDF files for further processing or analysis.
   - **Usage**:
     ```bash
     python preprocessing/pdf_extractor.py
     # save pdf extractions to .txt file
     ```

## `re_clean_txt`
   - **Purpose**: Cleans and processes raw text by removing unnecessary characters, formatting issues, or other distractions.
   - **Usage**:
     ```bash
     python preprocessing/re_clean_txt.py
     # save cleaned txt file into another .txt file
     ```

## `tuning_scripts`
   - **Purpose**: Tuning scripts for RAG pipeline hyperparameters.
   - **Usage**:
     ```bash
     bash top_k.ps1
     ...
     # save responses to given folder
     ```

