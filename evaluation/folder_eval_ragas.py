import os
import sys
import pandas as pd
import argparse
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.evaluation import evaluate
from datasets import Dataset  # Required for RAGAS
from langchain_openai import ChatOpenAI

# Import the API key from config
from config.keys import OPENAI_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Initialize RAGAS evaluator
llm = ChatOpenAI(model_name="gpt-3.5-turbo")

def evaluate_ragas(csv_file):
    """
    Evaluate a CSV file using RAGAS metrics.
    Requires 'question', 'answer' (GT), 'generated_answer' (model output), and 'context' columns.
    """
    df = pd.read_csv(csv_file)

    # Define a fixed system message for context if missing
    system_message = "Please generate the correct answer for the given fill-in-the-blank question. Avoid including unnecessary context, restating the question, or adding explanations—only return the precise answer."

    # Ensure 'context' exists, wrap it in a list
    if "context" not in df.columns:
        df["context"] = system_message  # Assign the system message to all rows

    df["context"] = df["context"].apply(lambda x: [x])

    # Ensure 'answer' (ground truth) exists
    if "answer" not in df.columns:
        raise ValueError("CSV file must contain an 'answer' column (ground truth).")

    # Ensure 'generated_answer' (model output) exists
    if "generated_answer" not in df.columns:
        raise ValueError("CSV file must contain a 'generated_answer' column (model output).")

    # Convert Pandas DataFrame to Hugging Face Dataset
    dataset = Dataset.from_pandas(df)

    # Perform RAGAS evaluation
    scores = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm
    )
    
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
    parser.add_argument("--input_folder", type=str, default="QA_pair/qa_pair_200_0210/output_200", help="Folder containing CSV files for evaluation")
    parser.add_argument("--output_csv", type=str, default="evaluation/folder_eval_out_ragas.csv", help="Output CSV file for storing evaluation results")
    args = parser.parse_args()
    
    evaluate_folder(args.input_folder, args.output_csv)

if __name__ == "__main__":
    main()
