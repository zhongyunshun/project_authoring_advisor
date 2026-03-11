from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any


@dataclass
class SourceNode:
    """Simplified representation of a retrieved source chunk."""
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0


@dataclass
class RAGResponse:
    """Unified response from any RAG pipeline."""
    answer: str
    sources: List[SourceNode] = field(default_factory=list)

    @property
    def context_text(self) -> str:
        return "\n\n".join(s.text for s in self.sources)


class BaseRAGPipeline(ABC):
    @abstractmethod
    def query(self, question: str) -> RAGResponse:
        ...

    @abstractmethod
    def reset(self):
        """Reset conversation state (if any)."""
        ...
