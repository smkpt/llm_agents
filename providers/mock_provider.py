"""
Enterprise AI Platform: Deterministic Mock Provider for Offline Development & CI/CD
"""

from __future__ import annotations
import asyncio
import time
from typing import AsyncIterator, List, Optional, Type, TypeVar
from pydantic import BaseModel

from core.base import BaseLLMProvider
from core.schemas import (
    ChatMessage,
    LLMResponse,
    LLMStreamChunk,
    ProviderType,
    TokenUsage,
)

T = TypeVar("T", bound=BaseModel)


class MockLLMProvider(BaseLLMProvider):
    """
    Simulates high-fidelity LLM behavior for offline development,
    unit testing, and circuit breaker failover drills.
    """

    def __init__(
        self,
        default_model: str = "mock-enterprise-model",
        simulated_latency_ms: float = 120.0,
        should_fail: bool = False,
    ):
        super().__init__(ProviderType.MOCK, default_model)
        self.simulated_latency_ms = simulated_latency_ms
        self.should_fail = should_fail

    async def generate(
        self,
        messages: List[ChatMessage],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        if self.should_fail:
            raise RuntimeError("Mock Provider Error: Upstream service unavailable (Simulated 503 Outage)")

        await asyncio.sleep(self.simulated_latency_ms / 1000.0)
        last_msg = messages[-1].content if messages else "Hello"
        content = f"[Mock AI Response] Echoing query: '{last_msg[:60]}...' generated with model={model or self.default_model}."

        prompt_tokens = sum(len(m.content.split()) for m in messages) * 2
        comp_tokens = len(content.split()) * 2

        return LLMResponse(
            content=content,
            model=model or self.default_model,
            provider=self.provider_type,
            latency_ms=self.simulated_latency_ms,
            finish_reason="stop",
            usage=TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=comp_tokens,
                total_tokens=prompt_tokens + comp_tokens,
                estimated_cost_usd=0.0,
            ),
        )

    async def generate_stream(
        self,
        messages: List[ChatMessage],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> AsyncIterator[LLMStreamChunk]:
        tokens = ["Enterprise", " AI", " Platform", " [Streaming", " Mock", " Stream]"]
        for t in tokens:
            await asyncio.sleep(0.04)
            yield LLMStreamChunk(delta=t)
        yield LLMStreamChunk(delta="", finish_reason="stop")

    async def generate_structured(
        self,
        messages: List[ChatMessage],
        response_model: Type[T],
        model: Optional[str] = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        await asyncio.sleep(self.simulated_latency_ms / 1000.0)
        
        # Instantiate model with default/mock values
        fields = response_model.model_fields
        mock_data = {}
        for fname, finfo in fields.items():
            if finfo.annotation == str:
                mock_data[fname] = f"Mock {fname}"
            elif finfo.annotation == int:
                mock_data[fname] = 42
            elif finfo.annotation == float:
                mock_data[fname] = 99.5
            elif finfo.annotation == bool:
                mock_data[fname] = True
            elif getattr(finfo.annotation, "__origin__", None) == list:
                mock_data[fname] = ["Mock Item 1", "Mock Item 2"]
            else:
                mock_data[fname] = None

        instance = response_model.model_validate(mock_data)

        return LLMResponse(
            content=instance.model_dump_json(),
            parsed=instance,
            model=model or self.default_model,
            provider=self.provider_type,
            latency_ms=self.simulated_latency_ms,
            finish_reason="stop",
            usage=TokenUsage(
                prompt_tokens=50,
                completion_tokens=25,
                total_tokens=75,
                estimated_cost_usd=0.0,
            ),
        )

    async def embed(self, texts: List[str], model: Optional[str] = None) -> List[List[float]]:
        # Return deterministic 1536-dimensional mock vectors
        return [[0.01 * (i + 1) for _ in range(1536)] for i, _ in enumerate(texts)]

    async def is_healthy(self) -> bool:
        return not self.should_fail
