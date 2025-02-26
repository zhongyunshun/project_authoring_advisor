import pandas as pd
from rouge_score import rouge_scorer

def calculate_rouge(csv_file):
    """
    Calculates the ROUGE-1, ROUGE-2, and ROUGE-3 scores between 
    reference answers and generated answers.
    
    - Converts text to lowercase for case insensitivity.
    - Uses 'alternative_answer' as an additional reference if available.
    - Ensures all inputs are treated as strings.

    Args:
        csv_file (str): Path to the CSV file containing 'answer', 'generated_answer', and optionally 'alternative_answer'.

    Returns:
        dict: Average ROUGE scores (ROUGE-1, ROUGE-2, and ROUGE-3).
    """
    df = pd.read_csv(csv_file, dtype=str)  # ✅ Force all columns to be read as strings

    if "answer" not in df.columns or "generated_answer" not in df.columns:
        raise ValueError("CSV file must contain 'answer' and 'generated_answer' columns.")
    
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rouge3"], use_stemmer=True)

    rouge1_scores = []
    rouge2_scores = []
    rouge3_scores = []

    for _, row in df.iterrows():
        # Convert all values to strings and handle NaN cases safely
        references = [str(row["answer"]).strip().lower() if pd.notna(row["answer"]) else ""]
        
        if "alternative_answer" in df.columns and pd.notna(row["alternative_answer"]):
            references.append(str(row["alternative_answer"]).strip().lower())

        # Ensure generated answer is a valid string
        generated_answer = str(row["generated_answer"]).strip().lower() if pd.notna(row["generated_answer"]) else ""

        # Only compute ROUGE if references exist and hypothesis is not empty
        if references and generated_answer:
            scores = [scorer.score(ref, generated_answer) for ref in references]
            
            best_scores = {
                "rouge1": max(s["rouge1"].fmeasure for s in scores),
                "rouge2": max(s["rouge2"].fmeasure for s in scores),
                "rouge3": max(s["rouge3"].fmeasure for s in scores)
            }
        else:
            best_scores = {"rouge1": 0, "rouge2": 0, "rouge3": 0}  # Assign 0 if missing

        rouge1_scores.append(best_scores["rouge1"])
        rouge2_scores.append(best_scores["rouge2"])
        rouge3_scores.append(best_scores["rouge3"])
    
    avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else 0
    avg_rouge2 = sum(rouge2_scores) / len(rouge2_scores) if rouge2_scores else 0
    avg_rouge3 = sum(rouge3_scores) / len(rouge3_scores) if rouge3_scores else 0

    avg_scores = {
        "ROUGE-1": avg_rouge1,
        "ROUGE-2": avg_rouge2,
        "ROUGE-3": avg_rouge3
    }

    return avg_scores

if __name__ == "__main__":
    csv_file = "QA_pair/qa_pair_200_0210/output/TRCA_All_Files_Combined_with_alternative_answers_output.csv"
    
    if pd.read_csv(csv_file).empty:
        print("🚨 The CSV file is empty or missing required data!")
    else:
        rouge_scores = calculate_rouge(csv_file)
        print(rouge_scores)
