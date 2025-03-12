```bash
python -m evaluation.folder_eval_ragas
```

Required args from csv (see the folder_eval_ragas.py for information on how the columns are renamed):

- faithfulness: ['user_input', 'retrieved_contexts', 'response']
- answer_relevancy: ['response', 'user_input']
- context_precision: ['user_input', 'retrieved_contexts', 'reference']
- context_recall: ['retrieved_contexts', 'user_input', 'reference']
