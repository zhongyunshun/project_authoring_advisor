from typing import List, Optional

from llama_index.core.agent import ReActAgent
from llama_index.core.llms import LLM
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.tools import BaseTool

from pipeline.base import BaseRAGPipeline, RAGResponse, SourceNode


class RAGAgent(BaseRAGPipeline):
    """Agentic RAG that uses ReAct reasoning to select tools (document search, web search, etc.).

    The agent autonomously decides whether to query local documents, search the web,
    or answer from its own knowledge—replacing the manual ``use_web=True/False`` flag.
    """

    SYSTEM_PROMPT = (
        "You are a helpful TRCA (Toronto and Region Conservation Authority) assistant. "
        "You have access to tools for searching technical documents and the web. "
        "Always try the document search tool first for TRCA-specific questions. "
        "Use web search for current events or information not in the documents. "
        "Cite your sources when possible."
    )

    def __init__(
        self,
        tools: List[BaseTool],
        llm: LLM,
        system_prompt: Optional[str] = None,
        conversational: bool = True,
        memory_token_limit: int = 4096,
        verbose: bool = False,
    ):
        self._conversational = conversational
        self._memory = (
            ChatMemoryBuffer.from_defaults(token_limit=memory_token_limit)
            if conversational
            else None
        )

        self._agent = ReActAgent.from_tools(
            tools=tools,
            llm=llm,
            memory=self._memory,
            system_prompt=system_prompt or self.SYSTEM_PROMPT,
            verbose=verbose,
        )

    def query(self, question: str) -> RAGResponse:
        response = self._agent.chat(question)

        sources = []
        for node_ws in getattr(response, "source_nodes", []):
            node = node_ws.node
            sources.append(SourceNode(
                text=node.get_content(),
                metadata=node.metadata or {},
                score=getattr(node_ws, "score", 0.0),
            ))

        return RAGResponse(
            answer=str(response),
            sources=sources,
        )

    def reset(self):
        if self._memory:
            self._memory.reset()
        self._agent.reset()
