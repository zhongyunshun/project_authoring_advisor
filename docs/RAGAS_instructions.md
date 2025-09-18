```bash
python -m evaluation.folder_eval_ragas
```

Required args from csv (see the folder_eval_ragas.py for information on how the columns are renamed):

- faithfulness: ['user_input', 'retrieved_contexts', 'response']
- https://docs.ragas.io/en/v0.1.21/concepts/metrics/faithfulness.html

- answer_relevancy: ['response', 'user_input']
- https://docs.ragas.io/en/v0.1.21/concepts/metrics/answer_relevance.html

- context_precision: ['user_input', 'retrieved_contexts', 'reference']
- https://docs.ragas.io/en/v0.1.21/concepts/metrics/context_precision.html

- context_recall: ['retrieved_contexts', 'user_input', 'reference']
- https://docs.ragas.io/en/v0.1.21/concepts/metrics/context_recall.html
