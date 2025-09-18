import os
import sys
import pandas as pd
import argparse
import re
import ast

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ragas ver = 0.1.21
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.evaluation import evaluate, RunConfig
from datasets import Dataset  # Required for RAGAS
from langchain_openai import ChatOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from config.keys import OPENAI_API_KEY

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Initialize RAGAS evaluator
llm = ChatOpenAI(model_name="gpt-3.5-turbo")
run_config = RunConfig(timeout = 180, max_workers = 6)

# Function to retrieve query and context list from prompt txt files
def parse_question_file(input_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Extract query between #### markers
    query_match = re.search(r"The query will be delimited with #### characters: ####\s*(.*?)\s*####", content, re.DOTALL)
    query = query_match.group(1).strip() if query_match else None

    # Extract the block starting from the injected domain info
    inject_start = re.search(r"# Inject Domain Information\s+Here is the retrieved passage:\s+\{", content)
    if not inject_start:
        context_list = []
        # raise ValueError("Injected domain information block not found.")
    else:
        # Extract each document starting with --- Document x ---
        context_block = content[inject_start.end():].strip()
        context_list = re.findall(r"--- Document \d+ ---.*?(?=(?:--- Document \d+ ---|$))", context_block, re.DOTALL)

    return query, context_list


# Function to retrieve response from prompting result txt files
def parse_response_file(response_path):
    with open(response_path, 'r', encoding='utf-8') as f:
        response = f.read().strip()
    return response


# Retry decorator with exponential backoff
@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=10))
def evaluate_with_retry(dataset, llm):
    """Retries evaluation in case of API rate limits"""
    return evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall], 
        llm=llm,
        run_config=run_config
    )


def build_ragas_dataset(question_dir, response_dir, output_csv_dir=None):
    rows = []

    for root, _, files in os.walk(question_dir):
        for file in files:
            if not file.endswith(".txt"):
                continue

            rel_path = os.path.relpath(os.path.join(root, file), question_dir)
            qpath = os.path.join(question_dir, rel_path)
            rpath = os.path.join(response_dir, rel_path)

            if not os.path.isfile(rpath):
                print(f"Missing response for {rel_path}, skipping.")
                continue

            try:
                # We take Persona+COT+Format Template as ground truth and skip processing it.
                if "Persona+COT+Format Template" in rel_path.replace("\\", "/"):
                    reference = parse_response_file(rpath)
                    continue  
                else:
                    query, contexts = parse_question_file(qpath)
                    response = parse_response_file(rpath)

                    prompting_type = rel_path.split(os.sep)[0]  # First folder under response_dir (e.g., "COT+Format Template")
                    question_name = os.path.splitext(os.path.basename(rel_path))[0]  # Filename without extension

                    # Get reference (gt) through path
                    reference_path = os.path.join(response_dir, "Persona+COT+Format Template", rel_path.split(os.sep)[-1])
                    reference = parse_response_file(reference_path) if os.path.isfile(reference_path) else ""

                    rows.append({
                        "prompting_type": prompting_type,
                        "question_name": question_name,
                        "user_input": query,
                        "retrieved_contexts": contexts,
                        "response": response,
                        "reference": reference
                    })

            except Exception as e:
                print(f"Error processing {rel_path}: {e}")

    df = pd.DataFrame(rows)

    # Save dataframe to a csv
    if output_csv_dir:
        df.to_csv(output_csv_dir, index=False)

    return Dataset.from_pandas(df)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question_dir", default="prompt_engineer_ragas/prompts", type=str, help="Folder containing question .txt files")
    parser.add_argument("--response_dir", default="prompt_engineer_ragas/prompting_results", type=str,  help="Folder containing response .txt files")
    parser.add_argument("--output_csv", type=str, default="prompt_engineer_ragas/evaluation_results_with_gt.csv", help="Output CSV for evaluation metrics")
    parser.add_argument("--prompting_csv_path", type=str, default="prompt_engineer_ragas/prompting_ragas_input_with_gt.csv", help="Output CSV path for the summarized input from txt")
    args = parser.parse_args()

    # Find or Build ragas dataset from input txt file
    if os.path.exists(args.prompting_csv_path):
        df = pd.read_csv(args.prompting_csv_path)

        # Make sure retrieved_contexts is loaded back as list instead of strings
        if "retrieved_contexts" in df.columns:
            df["retrieved_contexts"] = df["retrieved_contexts"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )

        dataset = Dataset.from_pandas(df)
        print("Found ragas dataset")
    else:
        dataset = build_ragas_dataset(args.question_dir, args.response_dir, args.prompting_csv_path)
        print("Finish building ragas dataset")

    scores = evaluate_with_retry(dataset, llm)
    df_scores = scores.to_pandas()

    # Add prompting_type and question_name from input
    df_input = pd.read_csv(args.prompting_csv_path)
    df_scores["prompting_type"] = df_input["prompting_type"]
    df_scores["question_name"] = df_input["question_name"]

    # print(df_scores)
    df_scores.to_csv(args.output_csv, index=False)
    print(f"\n✅ Evaluation saved to {args.output_csv}")

if __name__ == "__main__":
    main()
