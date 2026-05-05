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

        try:
            self._agent = ReActAgent(
                tools=tools,
                llm=llm,
                memory=self._memory,
                system_prompt=system_prompt or self.SYSTEM_PROMPT,
                verbose=verbose,
            )
        except TypeError:
            # Older llama-index API used a classmethod constructor.
            self._agent = ReActAgent.from_tools(
                tools=tools,
                llm=llm,
                memory=self._memory,
                system_prompt=system_prompt or self.SYSTEM_PROMPT,
                verbose=verbose,
            )

    def query(self, question: str) -> RAGResponse:
        sources: List[SourceNode] = []

        try:
            # Legacy sync API (older llama-index versions).
            response = self._agent.chat(question)
            answer = str(response)
            for node_ws in getattr(response, "source_nodes", []) or []:
                node = node_ws.node
                sources.append(SourceNode(
                    text=node.get_content(),
                    metadata=node.metadata or {},
                    score=getattr(node_ws, "score", 0.0),
                ))
            return RAGResponse(answer=answer, sources=sources)
        except AttributeError:
            pass  # Fall through to modern workflow API.

        import asyncio

        try:
            from llama_index.core.agent.workflow import ToolCallResult
        except ImportError:
            from llama_index.core.agent.workflow.workflow_events import ToolCallResult

        async def _run():
            handler = self._agent.run(user_msg=question)
            captured: List[SourceNode] = []
            async for ev in handler.stream_events():
                if isinstance(ev, ToolCallResult):
                    tool_output = getattr(ev, "tool_output", None)
                    raw = getattr(tool_output, "raw_output", None) if tool_output else None
                    for nws in getattr(raw, "source_nodes", []) or []:
                        n = nws.node
                        captured.append(SourceNode(
                            text=n.get_content(),
                            metadata=n.metadata or {},
                            score=getattr(nws, "score", 0.0),
                        ))
            result = await handler
            return result, captured

        response, sources = asyncio.run(_run())

        msg = getattr(response, "response", None)
        if msg is not None and hasattr(msg, "content") and msg.content:
            answer = msg.content
        else:
            answer = str(response)

        return RAGResponse(answer=answer, sources=sources)

    def reset(self):
        if self._memory:
            self._memory.reset()
        try:
            self._agent.reset()
        except AttributeError:
            pass  # Modern workflow agent has no .reset(); memory.reset() above is enough.
