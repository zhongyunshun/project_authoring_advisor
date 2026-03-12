"""Abstract backend adapter — implement this to plug in any RAG system."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class BackendQuery:
    """Payload sent to the backend."""

    query: str
    conversation_id: str = ""
    user_id: str = "default"
    image_base64: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BackendResponse:
    """Payload returned by the backend."""

    answer: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BackendAdapter(ABC):
    """Interface every RAG backend adapter must implement."""

    @abstractmethod
    async def process(self, query: BackendQuery) -> BackendResponse:
        """Send a query and return the answer."""
        ...

    async def health_check(self) -> bool:
        """Return True if the backend is reachable."""
        return True
