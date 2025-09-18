import pandas as pd
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import os

def calculate_bleu(csv_file, n=3):
    """
    Calculates the BLEU score (n-gram based) between reference answers and generated answers.
    - Converts text to lowercase for case insensitivity.
    - Uses 'alternative_answer' as an additional reference if available.
    - Ensures all inputs are treated as strings.

    Args:
        csv_file (str): Path to the CSV file containing 'answer', 'generated_answer', and optionally 'alternative_answer'.
        n (int): Maximum n-gram order for BLEU score.

    Returns:
        float: Average BLEU score.
    """
    df = pd.read_csv(csv_file, dtype=str)  # ✅ Force all columns to be read as strings

    if "answer" not in df.columns or "generated_answer" not in df.columns:
        raise ValueError("CSV file must contain 'answer' and 'generated_answer' columns.")
    
    smoothie = SmoothingFunction().method1  # Smooth for better BLEU scoring
    scores = []

    # Create n-gram weights dynamically
    weights = tuple((1/n for _ in range(n))) + (0,) * (4 - n)  # Ensures proper BLEU weight format

    for _, row in df.iterrows():
        # Convert all values to strings (handling missing values safely)
        ref_answers = [str(row["answer"]).strip().lower() if pd.notna(row["answer"]) else ""]

        if "alternative_answer" in df.columns and pd.notna(row["alternative_answer"]):
            ref_answers.append(str(row["alternative_answer"]).strip().lower())

        references = [r.split() for r in ref_answers if r]  # Tokenize only non-empty references

        # Convert generated answer to string safely
        generated_answer = str(row["generated_answer"]).strip().lower() if pd.notna(row["generated_answer"]) else ""
        hyp_tokens = generated_answer.split() if generated_answer else []

        # Only compute BLEU if both references and hypothesis exist
        if references and hyp_tokens:
            score = sentence_bleu(references, hyp_tokens, weights=weights, smoothing_function=smoothie)
        else:
            score = 0  # Assign BLEU score 0 if either is missing

        scores.append(score)

    avg_bleu = sum(scores) / len(scores) if scores else 0
    return avg_bleu, scores

if __name__ == "__main__":
    # Define parameters
    chunk_length = 200
    search_type = "similarity"

    top_k_values = [2, 3, 4, 5, 6, 7, 8, 9]

    for top_k in top_k_values:
        csv_file = f"QA_pair/qa_pair_200_0210/output/output_chunk{chunk_length}_top{top_k}_{search_type}.csv"
        
        if os.path.exists(csv_file):
            bleu_score, _ = calculate_bleu(csv_file)
            print(f"BLEU Score for top_k={top_k}: {bleu_score:.4f}")
        else:
            print(f"File not found: {csv_file}")
