import pandas as pd
import re
import json
import os

def summarize_csv_scores(input_csv_path, output_dir):
    df = pd.read_csv(input_csv_path)
    os.makedirs(output_dir, exist_ok=True)

    for metric in ["faithfulness", "answer_relevancy"]:
        pivot = df.pivot_table(index="prompting_type", 
                               columns="question_name", 
                               values=metric, 
                               aggfunc="mean")
        pivot["Average"] = pivot.mean(axis=1)
        pivot.to_csv(os.path.join(output_dir, f"{metric}_summary.csv"))
    print("CSV summaries written to:", output_dir)

def summarize_txt_scores(input_txt_path, output_dir):
    with open(input_txt_path, "r", encoding="utf-8") as f:
        text = f.read()

    question_blocks = re.split(r"===== Evaluation for ((?:gm|hbpe|pc)_question\d+) =====", text)[1:]
    question_data = list(zip(question_blocks[::2], question_blocks[1::2]))

    score_tables = {}

    for question_name, block in question_data:
        # Match both ```json and ```python fenced blocks
        code_match = re.search(r"```(?:json|python)(.*?)```", block, re.DOTALL)
        if not code_match:
            print(f"Warning: No summary code block found for {question_name}")
            continue

        try:
            summary_json = json.loads(code_match.group(1).strip())
        except json.JSONDecodeError:
            print(f"Error decoding JSON for {question_name}")
            continue

        patterns = summary_json["Patterns"]
        for metric, scores in summary_json.items():
            if metric == "Patterns":
                continue
            metric_name = metric.split(" (")[0].strip()
            if metric_name not in score_tables:
                score_tables[metric_name] = pd.DataFrame(index=patterns)
            score_tables[metric_name][question_name] = scores

    os.makedirs(output_dir, exist_ok=True)

    for metric_name, df in score_tables.items():
        df["Average"] = df.mean(axis=1)
        filename = f"{metric_name.lower().replace(' ', '_')}_summary.csv"
        df.to_csv(os.path.join(output_dir, filename))
        print(f"✅ TXT metric summary written to: {os.path.join(output_dir, filename)}")


if __name__ == "__main__":
    input_csv = "prompt_engineer_ragas/evaluation_results.csv"
    input_txt = "prompt_engineer_ragas/custom_eval_results.txt"
    output_folder_csv = "prompt_engineer_ragas/summary/ragas"
    output_folder_txt = "prompt_engineer_ragas/summary/custom"
    summarize_csv_scores(input_csv, output_folder_csv)
    summarize_txt_scores(input_txt, output_folder_txt)
