import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from openai import OpenAI

from prompt_engineer_ragas import questions
from configs.keys import OPENAI_API_KEY

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
client = OpenAI()

model = "gemini"

def main():
    base_answer_path = f"prompt_engineer_ragas/prompting_results_separate/{model}"
    # prompt_types = [
    #     "Persona+COT+Format Template",
    #     "COT+Format Template",
    #     "Persona+Format Template",
    #     "Persona+COT",
    #     "GPT with RAG only",
    #     "Original GPT without RAG"
    # ]

    prompt_types = [
        "GPT with RAG",
        "GPT with RAG + COT",
        "GPT with RAG + Format Template",
        "GPT with RAG + Persona",
        "Original GPT without RAG"
    ]

    # Load answers grouped by question
    answers_by_question = {}

    for p_type in prompt_types:
        folder_path = os.path.join(base_answer_path, p_type)
        for fname in os.listdir(folder_path):
            if fname.endswith(".txt"):
                q_name = fname.replace(".txt", "")
                file_path = os.path.join(folder_path, fname)

                with open(file_path, "r", encoding="utf-8") as f:
                    answer = f.read().strip()

                if q_name not in answers_by_question:
                    answers_by_question[q_name] = []

                answers_by_question[q_name].append((p_type, answer))

    # Sort answers in prompt_types order
    for q_name, answers in answers_by_question.items():
        answers_by_question[q_name] = sorted(answers, key=lambda x: prompt_types.index(x[0]))

    # Evaluation prompt template
    metrics_with_explanations = '''
Act as an answer evaluator.
Using the information provided in TRCA's technical documents, please score the following six answers to the provided question based on the five specified metrics:
Comprehensiveness, Accuracy, Relevance, Clarity and Understandability, and Conciseness.
Each metric should be scored on a scale of 0-20 points, with a total possible score of 100 points.
Provide a brief justification for each score, highlighting the strengths and weaknesses of the answer in relation to the question it addresses.

Consider the following aspects when scoring:

Comprehensiveness (0-20 points): Assess whether the answer fully addresses all parts of the question, including any sub-questions or implied inquiries. Consider the depth and detail of the response.

Accuracy (0-20 points): Evaluate the factual correctness of the answer. Verify that the information provided aligns with data or statements within the TRCA's technical documents.

Relevance (0-20 points): Determine if the answer stays focused on the question's topic without deviating into unrelated content. The response should directly address the posed question.

Clarity and Understandability (0-20 points): Judge how easily the answer can be understood. The language should be clear, technical terms (if used) are explained, and the answer is structured logically.

Conciseness (0-20 points): Consider if the answer is succinct yet comprehensive. It should convey necessary information without unnecessary length or redundancy.

After scoring each metric, calculate the total score and provide a final assessment of the answer's quality.
Summarize the scores in a dictionary with this format:
{
"Patterns": ["GPT with RAG",
        "GPT with RAG + COT",
        "GPT with RAG + Format Template",
        "GPT with RAG + Persona",
        "Original GPT without RAG"],
    "Comprehensiveness (0-20 points)": [],
    "Accuracy (0-20 points)": [],
    "Relevance (0-20 points)": [],
    "Clarity and Understandability (0-20 points)": [],
    "Conciseness (0-20 points)": [],
    "Total (0-100 points)": []
}

Now I will provide the question and five answers for evaluation.

'''

    # Choose one or loop through all questions
    for q_name, answers in answers_by_question.items():
        try:
            question = getattr(questions, q_name)
        except AttributeError:
            print(f"⚠️ Skipping {q_name}: not found in questions module.")
            continue

        # Construct the appended part with answers
        # index in sequence of prompt_types list
        question_and_answers = f"\nQuestion: {question}\n\n"
        for idx, (ptype, ans) in enumerate(answers):
            question_and_answers += f"Answer {idx+1} ({ptype}):\n{ans}\n\n"

        # Append final prompt
        final_prompt = metrics_with_explanations + question_and_answers

        # Call gpt4o mini
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": final_prompt}
                ],
            )

            print(f"===== Evaluation for {q_name} =====")
            print(response.choices[0].message.content)
            print("\n\n")

        except Exception as e:
            print(f"Error during OpenAI API call for {q_name}: {e}")

if __name__ == "__main__":
    log_file = open(f"prompt_engineer_ragas/data/custom_eval_results_separate_{model}.txt", "w", encoding="utf-8")
    original_stdout = sys.stdout
    sys.stdout = log_file

    try:
        main()
    finally:
        sys.stdout = original_stdout
        log_file.close()
