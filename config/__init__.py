'''

  ┌──────────────────┬─────────────────────────────────────────────────────┬───────────────────────────────────────────────────────────┐
  │      Aspect      │                Old (LangChain+FAISS)                │                  New (LlamaIndex+Qdrant)                  │
  ├──────────────────┼─────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────┤
  │ Pipeline classes │ 12 (4 classes × 3 files)                            │ 2 (RAGEngine + PromptingRAGEngine)                        │
  ├──────────────────┼─────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────┤
  │ Embedding files  │ 4 duplicate files                                   │ 1 Indexer class                                           │
  ├──────────────────┼─────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────┤
  │ Vector store     │ FAISS (save/load/merge)                             │ Qdrant (incremental upsert, no merge)                     │
  ├──────────────────┼─────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────┤
  │ Memory           │ Manual ConversationBufferMemory + string formatting │ Built-in ChatMemoryBuffer + CondensePlusContextChatEngine │
  ├──────────────────┼─────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────┤
  │ Web search       │ Manual flag use_web=True                            │ Agent decides autonomously                                │
  ├──────────────────┼─────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────┤
  │ CLI entry points │ 2 separate scripts                                  │ 1 unified main.py                                         │
  ├──────────────────┼─────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────┤
  │ Adding a new LLM │ Copy-paste a whole pipeline file                    │ Add 5 lines to LLMFactory                                 │
  └──────────────────┴─────────────────────────────────────────────────────┴───────────────────────────────────────────────────────────┘

Interactive chat: python main.py --mode chat --embedding huggingface
(Using ChatGPT) python main.py --mode chat --model openai --model_name gpt-4.1-mini --embedding huggingface
(Using gemini) python main.py --mode chat --model gemini --model_name models/gemini-2.0-flash --embedding huggingface
(Using Claude) python main.py --mode chat --model claude --model_name claude-sonnet-4-6 --embedding huggingface
Streamlit web UI:  streamlit run app.py
For agentic mode (auto web search):  python main.py --mode agent --embedding huggingface
python main.py --mode agent --embedding huggingface
'''

