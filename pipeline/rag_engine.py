import re
from typing import Optional

from llama_index.core import VectorStoreIndex
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.llms import LLM
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer

from pipeline.base import BaseRAGPipeline, RAGResponse, SourceNode
from pipeline.prompt_templates import build_prompt


def _extract_sources(source_nodes) -> list[SourceNode]:
    """Convert LlamaIndex NodeWithScore objects to our SourceNode format."""
    sources = []
    for node_with_score in source_nodes:
        node = node_with_score.node
        text = re.sub(r"\x03+", " ", node.get_content())
        sources.append(SourceNode(
            text=text,
            metadata=node.metadata or {},
            score=node_with_score.score or 0.0,
        ))
    return sources


class RAGEngine(BaseRAGPipeline):
    """Unified RAG engine that replaces ConversationalRAG, StatelessRAG, and ConversationalPDFRAG.

    Set ``conversational=True`` for multi-turn chat with memory,
    or ``conversational=False`` for stateless single-query mode.
    """

    def __init__(
        self,
        index: VectorStoreIndex,
        llm: LLM,
        system_prompt: str = "You are a helpful assistant.",
        top_k: int = 22,
        conversational: bool = True,
        memory_token_limit: int = 4096,
    ):
        self._index = index
        self._llm = llm
        self._top_k = top_k
        self._conversational = conversational

        self._retriever = index.as_retriever(similarity_top_k=top_k)

        if conversational:
            self._memory = ChatMemoryBuffer.from_defaults(token_limit=memory_token_limit)
            self._engine = CondensePlusContextChatEngine.from_defaults(
                retriever=self._retriever,
                llm=llm,
                memory=self._memory,
                system_prompt=system_prompt,
            )
        else:
            self._memory = None
            self._engine = RetrieverQueryEngine.from_args(
                retriever=self._retriever,
                llm=llm,
                response_mode="compact",
            )

    def query(self, question: str) -> RAGResponse:
        if self._conversational:
            response = self._engine.chat(question)
        else:
            response = self._engine.query(question)

        source_nodes = getattr(response, "source_nodes", [])
        return RAGResponse(
            answer=str(response),
            sources=_extract_sources(source_nodes),
        )

    def reset(self):
        if self._memory:
            self._memory.reset()


class PromptingRAGEngine(BaseRAGPipeline):
    """Stateless RAG for prompt engineering research.

    Uses named prompt patterns (persona+cot+format, rag-only, etc.)
    to wrap retrieved context into structured prompts.
    """

    def __init__(
        self,
        index: VectorStoreIndex,
        llm: LLM,
        top_k: int = 22,
        pattern: str = "rag-only",
    ):
        self._index = index
        self._llm = llm
        self._top_k = top_k
        self._pattern = pattern
        self._retriever = index.as_retriever(similarity_top_k=top_k)

    def query(self, question: str) -> RAGResponse:
        nodes = self._retriever.retrieve(question)

        # Build retrieval log
        retrieval_log = ""
        for i, node_ws in enumerate(nodes):
            text = re.sub(r"\x03+", " ", node_ws.node.get_content())
            retrieval_log += f"\n--- Document {i + 1} ---\n"
            retrieval_log += f"Content:\n{text[:700]}...\n"
            retrieval_log += "Metadata:\n"
            for key, value in (node_ws.node.metadata or {}).items():
                display_val = value + 1 if key == "page" else value
                retrieval_log += f"- **{key}**: {display_val}\n"

        full_prompt = build_prompt(self._pattern, question, retrieval_log)
        response = self._llm.complete(full_prompt)

        return RAGResponse(
            answer=str(response),
            sources=_extract_sources(nodes),
        )

    def reset(self):
        pass  # stateless
