python -m evaluation.folder_eval

Required args from csv:
faithfulness: ['user_input', 'retrieved_contexts', 'response']
answer_relevancy: ['response', 'user_input']
context_precision: ['user_input', 'retrieved_contexts', 'reference']
context_recall: ['retrieved_contexts', 'user_input', 'reference']
