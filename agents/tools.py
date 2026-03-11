from typing import List, Optional

from llama_index.core import VectorStoreIndex
from llama_index.core.llms import LLM
from llama_index.core.tools import QueryEngineTool, FunctionTool


def create_document_query_tool(
    index: VectorStoreIndex,
    llm: LLM,
    top_k: int = 22,
) -> QueryEngineTool:
    """Wraps the vector index as a query tool for the agent."""
    query_engine = index.as_query_engine(
        llm=llm,
        similarity_top_k=top_k,
        response_mode="compact",
    )
    return QueryEngineTool.from_defaults(
        query_engine=query_engine,
        name="trca_documents",
        description=(
            "Search TRCA technical documents for information about conservation, "
            "environmental projects, shoreline maintenance, permits, and regulations. "
            "Use this tool for any question about TRCA project scopes, design notes, "
            "or institutional knowledge."
        ),
    )


def create_web_search_tool(
    provider: str = "tavily",
    max_results: int = 3,
) -> FunctionTool:
    """Web search tool that fetches results from the open web."""

    def web_search(query: str) -> str:
        """Search the web for current information about a topic. Returns text content from top results."""
        try:
            if provider == "tavily":
                from tavily import TavilyClient
                import os

                client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY", ""))
                results = client.search(query, max_results=max_results)
                texts = []
                for r in results.get("results", []):
                    texts.append(f"Source: {r.get('url', 'N/A')}\n{r.get('content', '')}")
                return "\n\n---\n\n".join(texts) if texts else "No web results found."

            elif provider == "serper":
                import requests
                import os

                resp = requests.get(
                    "https://google.serper.dev/search",
                    headers={"X-API-KEY": os.getenv("SERPER_API_KEY", "")},
                    params={"q": query, "num": max_results},
                )
                results = resp.json()
                texts = []
                for o in results.get("organic", [])[:max_results]:
                    texts.append(f"Source: {o.get('link', 'N/A')}\n{o.get('snippet', '')}")
                return "\n\n---\n\n".join(texts) if texts else "No web results found."

            else:
                return f"Unsupported web search provider: {provider}"

        except Exception as e:
            return f"Web search failed: {e}"

    return FunctionTool.from_defaults(
        fn=web_search,
        name="web_search",
        description="Search the open web for current information not found in TRCA documents.",
    )
