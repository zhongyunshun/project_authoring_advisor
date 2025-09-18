metrics_with_explanations = f'''
Act as an answer evaluator.
Using the information provided in TRCA's technical documents, please score the following five answers to the provided question based on the five specified metrics:
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
"Patterns": ["Persona+COT+Format Template", "COT+Format Template", "Persona+Format Template",
                                "Persona+COT", "Original GPT without RAG"],
        "Comprehensiveness (0-20 points)": [],
        "Accuracy (0-20 points)": [],
        "Relevance (0-20 points)": [],
        "Clarity and Understandability (0-20 points)": [],
        "Conciseness (0-20 points)": [],
        "Total (0-100 points)": []
}

Now I will provide the question and six answers for evaluation.

### Add question and answers here ###
'''