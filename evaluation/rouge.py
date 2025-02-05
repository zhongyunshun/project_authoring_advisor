import pandas as pd
from rouge_score import rouge_scorer

def calculate_rouge(csv_file):
    """
    Calculates the ROUGE-1, ROUGE-2, and ROUGE-L scores between 
    the 'answer' and 'generated_answer' columns in a CSV file.
    
    - Converts text to lowercase for case insensitivity.
    
    Args:
        csv_file (str): Path to the CSV file containing 'answer' and 'generated_answer' columns.

    Returns:
        dict: Average ROUGE scores (ROUGE-1, ROUGE-2, and ROUGE-L).
    """
    df = pd.read_csv(csv_file)

    if "answer" not in df.columns or "generated_answer" not in df.columns:
        raise ValueError("CSV file must contain 'answer' and 'generated_answer' columns.")

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []

    for ref, hyp in zip(df["answer"], df["generated_answer"]):
        ref, hyp = ref.lower(), hyp.lower()  # Convert to lowercase
        scores = scorer.score(ref, hyp)

        rouge1_scores.append(scores["rouge1"].fmeasure)
        rouge2_scores.append(scores["rouge2"].fmeasure)
        rougeL_scores.append(scores["rougeL"].fmeasure)

    avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else 0
    avg_rouge2 = sum(rouge2_scores) / len(rouge2_scores) if rouge2_scores else 0
    avg_rougeL = sum(rougeL_scores) / len(rougeL_scores) if rougeL_scores else 0

    avg_scores = {
        "ROUGE-1": avg_rouge1,
        "ROUGE-2": avg_rouge2,
        "ROUGE-L": avg_rougeL
    }

    return avg_scores

if __name__ == "__main__":
    csv_file = "QA_pair/qa_pair_170_0204/TRCA_All_Files_Combined_output.csv"
    rouge_scores = calculate_rouge(csv_file)
    print(rouge_scores)
