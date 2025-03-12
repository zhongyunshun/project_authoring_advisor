import os
import sys
import pandas as pd
import argparse
import ast

# ragas ver = 0.1.21
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.evaluation import evaluate
from datasets import Dataset  # Required for RAGAS
from langchain_openai import ChatOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

# Import the API key from config
from config.keys import OPENAI_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Initialize RAGAS evaluator
llm = ChatOpenAI(model_name="gpt-3.5-turbo")

# Retry decorator with exponential backoff
@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=10))
def evaluate_with_retry(dataset, llm):
    """Retries evaluation in case of API rate limits"""
    return evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm
    )

def evaluate_ragas(csv_file):
    """
    Evaluate a CSV file using RAGAS metrics.
    Requires 'user_input', 'reference' (GT), 'response' (model output), and 'retrieved_contexts' columns.
    """
    df = pd.read_csv(csv_file)

    df = df.head(20) # for small scale testing

    # Define the fixed system message
    system_message_csv = "Please generate the correct answer for the given fill-in-the-blank question. Avoid including unnecessary context, restating the question, or adding explanations—only return the precise answer."
    
    # Rename columns accordingly
    df.rename(columns={
        "question": "user_input",
        "answer": "reference",
        "generated_answer": "response"
    }, inplace=True)
    
    # Attach system message as prefix to user_input
    if "user_input" in df.columns:
        df["user_input"] = system_message_csv + " " + df["user_input"]
    
    # Ensure retrieved_contexts is a list of strings (not list of lists)
    if "retrieved_contexts" in df.columns:
        def process_contexts(context):
            if isinstance(context, str):
                context = ast.literal_eval(context)  # Convert stringified list to Python list
            return context

        df["retrieved_contexts"] = df["retrieved_contexts"].apply(process_contexts)

    # Validate required columns
    required_columns = {"user_input", "retrieved_contexts", "response", "reference"}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise ValueError(f"CSV file is missing required columns: {missing}")
    
    # Convert Pandas DataFrame to Hugging Face Dataset
    dataset = Dataset.from_pandas(df)
    
    # Perform RAGAS evaluation
    try:
        scores = evaluate_with_retry(dataset, llm)
    except Exception as e:
        print(f"Failed after retries: {e}")
        return None
    
    return scores

def evaluate_folder(input_folder, output_csv):
    """
    Evaluates all CSV files in a folder using RAGAS metrics.
    Outputs a summary CSV file.
    """
    results = []

    for filename in os.listdir(input_folder):
        if filename.endswith(".csv"):
            # Extract parameters from filename
            params = filename.replace("output_", "").replace(".csv", "").split("_")
            chunk_length = params[0].replace("chunk", "")
            top_k = params[1].replace("top", "")
            search_type = params[2]

            csv_file = os.path.join(input_folder, filename)

            # Compute RAGAS metrics
            ragas_scores = evaluate_ragas(csv_file)

            # Append results
            results.append({
                "Filename": filename,
                "Chunk Length": chunk_length,
                "Top K": top_k,
                "Search Type": search_type,
                "Faithfulness": ragas_scores["faithfulness"],
                "Answer Relevancy": ragas_scores["answer_relevancy"],
                "Context Precision": ragas_scores["context_precision"],
                "Context Recall": ragas_scores["context_recall"]
            })

    # Save results to CSV
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_csv, index=False)
    print(f"RAGAS evaluation results saved to {output_csv}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, default="QA_pair/ragas_pairs/temp", help="Folder containing CSV files for evaluation")
    parser.add_argument("--output_csv", type=str, default="evaluation/folder_eval_out_ragas.csv", help="Output CSV file for storing evaluation results")
    args = parser.parse_args()
    
    evaluate_folder(args.input_folder, args.output_csv)

if __name__ == "__main__":
    main()
