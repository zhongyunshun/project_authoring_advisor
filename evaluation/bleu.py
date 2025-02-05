import pandas as pd
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def calculate_bleu(csv_file, n=3):
    """
    Calculates the BLEU score (n-gram based) between the 'answer' and 'generated_answer' columns in a CSV file.
    - Converts text to lowercase for case insensitivity.
    - Allows custom n-gram weighting.

    Args:
        csv_file (str): Path to the CSV file containing 'answer' and 'generated_answer' columns.
        n (int): Maximum n-gram order for BLEU score.

    Returns:
        float: Average BLEU score.
    """
    df = pd.read_csv(csv_file)

    if "answer" not in df.columns or "generated_answer" not in df.columns:
        raise ValueError("CSV file must contain 'answer' and 'generated_answer' columns.")

    smoothie = SmoothingFunction().method1  # Smooth for better BLEU scoring
    scores = []

    # Create n-gram weights dynamically
    weights = tuple((1/n for _ in range(n))) + (0,) * (4 - n)  # Ensures proper BLEU weight format

    for ref, hyp in zip(df["answer"], df["generated_answer"]):
        ref_tokens = ref.lower().split()  # Convert to lowercase and tokenize
        hyp_tokens = hyp.lower().split()  # Convert to lowercase and tokenize

        score = sentence_bleu([ref_tokens], hyp_tokens, weights=weights, smoothing_function=smoothie)
        scores.append(score)

    avg_bleu = sum(scores) / len(scores) if scores else 0
    return avg_bleu

if __name__ == "__main__":
    csv_file = "QA_pair/qa_pair_170_0204/TRCA_All_Files_Combined_output.csv"
    bleu_score = calculate_bleu(csv_file)
    print(bleu_score)
