"""
Enterprise AI Platform: Abstract Base LLM Provider
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import AsyncIterator, List, Optional, Type, TypeVar
from pydantic import BaseModel
from core.schemas import ChatMessage, LLMResponse, LLMStreamChunk, ProviderType

T = TypeVar("T", bound=BaseModel)


class BaseLLMProvider(ABC):
    """Abstract interface that every enterprise LLM provider must implement."""

    def __init__(self, provider_type: ProviderType, default_model: str):
        self.provider_type = provider_type
        self.default_model = default_model

    @abstractmethod
    async def generate(
        self,
        messages: List[ChatMessage],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """Standard text generation across providers."""
        pass

    @abstractmethod
    async def generate_stream(
        self,
        messages: List[ChatMessage],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> AsyncIterator[LLMStreamChunk]:
        """Real-time streaming generation for Copilot UIs."""
        pass

    @abstractmethod
    async def generate_structured(
        self,
        messages: List[ChatMessage],
        response_model: Type[T],
        model: Optional[str] = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        """Strict schema enforcement and parsing using Pydantic."""
        pass

    @abstractmethod
    async def embed(
        self,
        texts: List[str],
        model: Optional[str] = None,
    ) -> List[List[float]]:
        """Vector embeddings for RAG and semantic retrieval."""
        pass

    @abstractmethod
    async def is_healthy(self) -> bool:
        """Circuit breaker health check probe."""
        pass
