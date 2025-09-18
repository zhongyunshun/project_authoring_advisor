# TRCA-LLM: Intelligent Chatbot for TRCA Document Knowledge

## Overview

The **TRCA-LLM** project leverages Large Language Models (LLMs) to create an intelligent chatbot capable of answering questions related to TRCA documents. By processing these documents into embeddings and using a Retrieval-Augmented Generation (RAG) pipeline, the chatbot can provide insightful responses based on the content of these documents.

---

## Project Structure

The project consists of the following directories:

- **`config/`**: Store config and keys including OpenAI API key.
- **`data/`**: Contains the TRCA document PDFs for processing.
- **`docs/`**: Documentation about all modules and RAGAS evaluation details.
- **`embeddings/`**: Modules for generating and loading document embeddings. #TODO: add how embedding is created, how is worked, etc.
- **`evaluation/`**: Includes evaluation scripts for assessing the RAG response performance.
- **`output/`**: Stores the processed txt from pdf documents.
- **`Pages/`**: Streamlit subpages.
- **`pipeline_preprocess_file_exec/`**: PDF processing helper functions.
- **`pipelines/`**: Modules for pipeline implementations.
- **`preprocessing/`**: Helper functions for preprocessing documents.
- **`prompt_engineer_ragas/`**: Prompt engineer related codes, including evaluation and prompt generation and inference.
- **`QA_pair/`**: All returning responses (question-answer from RAG pipeline) saved as csv.
- **`streamlit_class/`**: Streamlit helper class and functions.
- **`trials/`**: Trial functions and scripts for testing.
- **`tuning_scripts/`**: Scripts for tuning model parameters (top_k, chunk_size, search_type).
- **`utils/`**: Useful executable scripts used to handle processings.
- **`vector_db/`**: Contains the vector database for storing document embeddings.

For more information regarding the modules and helper functions, please view the file `docs/Module_Documentation.md`.

For more information regarding streamlit integration, check out [Streamlit Integration Section](#streamlit-integration).

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

## Generating Embeddings from PDF Documents

To generate embeddings for your documents, follow these steps:

### 1. Extract text from PDFs
- Place all PDF documents in a directory (e.g., `data/ALL_PDFS`)
- Run the PDF extractor script:
  ```bash
  python utils/pdf_extractor.py
  ```
This generates raw text files in `output/` directory

### 2. Clean and format extracted text
- Run the text cleaning script:

    ```bash
    python utils/re_clean_txt.py
    ```
- This removes unnecessary blanks and formats the text

### 3. Update input file path
- In main_model.py, update the input file path (around line 88):

    ```python
    input_file = "./output/cleaned_17.txt"  # Replace with your file
    ```

### 4. Generate and reuse embeddings
- Embeddings generate automatically when running main application

- To reuse existing embeddings (avoid regeneration), you should update Line 81 in main_model.py

    ```python
    embedding_file = f"vector_db/openai_chunk_{args.chunk_length}_embedding"
    ```
    
### Notes:
First run will take longer while generating embeddings
Subsequent runs with `embedding_file` variable updated will be faster.

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

---

## Streamlit Integration

The project includes a **Streamlit-based web interface** for interacting with the chatbot.

### Directory Structure

- **`StreamLitUI.py`**: The main Streamlit application.
- **`Pages/`**: Contains subpages for additional Streamlit functionalities.
- **`streamlit_class/conversations.py`**: Defines the `Conversation` class for managing chat sessions.

### Running the Streamlit App

```bash
streamlit run StreamLitUI.py
```

This will launch the chatbot in your web browser.

### Features

- **Session-based chat storage** with editable chat titles.
- **New Conversation button** to start a new conversation session.
- **Sidebar for entering OpenAI API keys** dynamically.
- **Document-aware responses** using the RAG pipeline.
- **Multi-page support** with subpages in the `pages/` folder.

---

## Evaluation

To evaluate generated answers from RAG pipeline, we have two evaluation methods:

```bash
# This is to evaluate the BLEU and ROUGE scores on generated answers.
python evaluation/folder_eval.py
```

```bash
# This is to evaluate RAGAS score based on faithfulness, answer_relevancy, context_precision, context_recall.
python -m evaluation.folder_eval_ragas
```

In addition, we have Custom ChatGPT-based Evaluation for Prompt Engineering
Uses a ChatGPT persona to act as an evaluator, scoring the quality of responses and prompts according to customized criteria. This helps analyze how different prompt designs affect answer quality. 
Note: Before running, update `line 15` in `custom_eval.py` to the path containing your evaluation data (including model responses).
You can see an example of the expected data format in: `prompt_engineer_ragas/prompting_results`. And then run:

```bash
python prompt_engineer_ragas/custom_eval.py
```

Detailed description of Prompt Engineering Evaluation is shown in below section.

## Prompt Engineering Evaluations

This module evaluates the effectiveness of different prompt engineering strategies using automated metrics. The pipeline takes in question-prompt pairs and their corresponding LLM-generated responses, then runs custom and RAGAS-based evaluations to assess answer quality. (Custom evaluation is having another LLM playing the role of a judge to determine the score of the responses from our LLM pipeline.)

The prompting types include:

    COT+Format Template
    GPT with RAG only
    Original GPT without RAG
    Persona+COT
    Persona+COT+Format Template
    Persona+Format Template

---

### Key Files

directories:
- `prompting_results/` – All generated inference response for 30 questions of six prompt types.
- `prompts/` – All generated prompts for 30 questions.
- `summary/` – Summary of custom and ragas evaluation results for all 30 questions of six prompt types.
- `templates/` – python files storing prompt templates
- `data/` – all txt csv files for inputs and outputs.

python files:
- `generate_responses.py` – Generates model inference responses from input prompts.
- `ragas_eval.py` – Applies RAGAS metrics evaluation.
- `custom_eval.py` – Runs custom evaluations.
- `summarize_csv_txt.py` – Summarizes txt results into csv format.
- `questions.py` – Includes all 30 questions for evaluation.

data files
- `custom_eval_results.csv` – Results from custom evaluation.
- `evaluation_results.csv` – Stores evaluation results. 
- `prompting_ragas_input.csv` – Formatted input for RAGAS evaluation.
- (.._with_gt.csv is the file with Persona+COT+Format Template as ground truth for evaluation.)

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

