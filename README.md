# TRCA-LLM: Intelligent Chatbot for TRCA Document Knowledge

## Overview

The **TRCA-LLM** project leverages Large Language Models (LLMs) to create an intelligent chatbot capable of answering questions related to TRCA documents. By processing these documents into embeddings and using a Retrieval-Augmented Generation (RAG) pipeline, the chatbot can provide insightful responses based on the content of these documents.

---

## Project Structure

The project consists of the following directories:

- **`config/`**: Store config and keys including OpenAI API key.
- **`data/`**: Contains the TRCA document PDFs for processing.
- **`embeddings/`**: Modules for generating and loading document embeddings.
- **`pipelines/`**: Modules for pipeline implementations.
- **`preprocessing/`**: Utilities for preprocessing documents.
- **`pipeline_preprocess_file_exec/`**: PDF processing execution scripts.
- **`QA_pair/`**: All returning responses (question-answer from RAG pipeline) saved as csv.
- **`tuning_scripts/`**: Scripts for tuning model parameters (top_k, chunk_size, search_type).
- **`vector_db/`**: Contains the vector database for storing document embeddings.
- **`evaluation/`**: Includes evaluation scripts for assessing the RAG response performance.
- **`utils/`**: Helper functions used across the project.
- **`output/`**: Stores the processed txt from pdf documents.

For more information regarding the modules and helper functions, please view the file `docs/Module_Documentation.md`.

---

## Setup

### Download Repo

```bash
git clone https://github.com/AlezHibali/TRCA-LLM.git
```

### Requirements

This project requires Python 3.12 and the dependencies listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```

### Configuration

The OpenAI API key is required for interacting with the language model. The API key should be stored in the **`config/keys.py`** file as follows:

```python
OPENAI_API_KEY = "your-openai-api-key"
```

## Running the Project

### Mode 1: **Batch Processing (CSV)**

This mode processes a folder of CSV files containing a list of questions and generates answers for each question, saving the results in a new CSV.

To run this mode, use the following command:

```bash
python main.py --mode csv --input_csv_file path/to/input.csv --output_csv_path path/to/output/
```

### Mode 2: **Interactive Chat Mode**

In this mode, the chatbot runs in an interactive loop where users can type questions and get answers in real-time. Type "exit" to stop the chat.

To start this mode, use the following command:

```bash
python main.py --mode chat
```

### Evaluation

To evaluate generated answers from RAG pipeline, we have two evaluation methods:

```bash
# This is to evaluate the BLEU and ROUGE scores on generated answers.
python evaluation/folder_eval.py
```

```bash
# This is to evaluate RAGAS score based on faithfulness, answer_relevancy, context_precision, context_recall.
python -m evaluation/folder_eval_ragas
```

---

## Customization

You can adjust the following parameters when running the script, either for tuning or for chatting:

- **`chunk_length`**: The length of text chunks for splitting the documents (default: 700).
- **`top_k`**: The number of top relevant chunks to retrieve during the RAG process (default: 22).
- **`search_type`**: The method used to measure the similarity of text chunks (default: "similarity"). Options: `similarity`, `mmr`, `similarity_score_threshold`.

---

## Detailed Workflow

1. **Text Splitting**: The input documents (TRCA PDFs) are split into smaller chunks to make it easier for the LLM to process and retrieve relevant sections.
2. **Embedding Generation**: Text chunks are converted into embeddings, which are stored in a vector database. 
3. **RAG Pipeline**: A Retrieval-Augmented Generation pipeline is employed to retrieve relevant information from the vector database and generate an answer based on it.
4. **Question Processing**: The system processes questions either in tuning/testing (CSV mode) or interactively (chat mode) using the RAG model, providing answers based on the TRCA document knowledge.

---

## Troubleshooting

- Ensure your OpenAI API key is correctly set in the `config/keys.py` file.
- If you're running into issues with embeddings or vector DB, ensure the correct environment and dependencies are set up (check `requirements.txt`).
- If encountering issues with loading embeddings, ensure the file paths and directory structure are correct.

---

