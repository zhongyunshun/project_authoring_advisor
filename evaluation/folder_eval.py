import os
import pandas as pd
import argparse
from bleu import calculate_bleu
from rouge import calculate_rouge

def evaluate_folder(input_folder, output_csv):
    """
    Evaluates all CSV files in a folder using BLEU and ROUGE metrics
    and outputs a summary CSV file.
    """
    results = []
    
    # Iterate over all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".csv"):
            # Extract parameters from filename
            params = filename.replace("output_", "").replace(".csv", "").split("_")
            chunk_length = params[0].replace("chunk", "")
            top_k = params[1].replace("top", "")
            search_type = params[2]

            csv_file = os.path.join(input_folder, filename)
            
            # Calculate BLEU and ROUGE scores
            bleu_score_1, _ = calculate_bleu(csv_file, n=1)
            bleu_score_2, _ = calculate_bleu(csv_file, n=2)
            bleu_score_3, _ = calculate_bleu(csv_file, n=3)
            bleu_score_avg = (bleu_score_1 + bleu_score_2 + bleu_score_3) / 3

            rouge_scores = calculate_rouge(csv_file)
            rouge_score_avg = (rouge_scores["ROUGE-1"] + rouge_scores["ROUGE-2"] + rouge_scores["ROUGE-3"]) / 3

            total_score = (bleu_score_avg + rouge_score_avg) / 2
            
            # Append results
            results.append({
                "Filename": filename,
                "Chunk Length": chunk_length,
                "Top K": top_k,
                "Search Type": search_type,
                "BLEU Score": bleu_score_avg,
                "ROUGE-1": rouge_scores["ROUGE-1"],
                "ROUGE-2": rouge_scores["ROUGE-2"],
                "ROUGE-3": rouge_scores["ROUGE-3"],
                "Total avg score": total_score
            })
    
    # Convert to DataFrame and save
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_csv, index=False)
    print(f"Evaluation results saved to {output_csv}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, default="QA_pair/qa_pair_200_0210/output_200", help="Folder containing CSV files for evaluation")
    parser.add_argument("--output_csv", type=str, default="evaluation/folder_eval_out.csv", help="Output CSV file for storing evaluation results")
    args = parser.parse_args()
    
    evaluate_folder(args.input_folder, args.output_csv)

if __name__ == "__main__":
    main()
