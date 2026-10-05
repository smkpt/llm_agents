"""
Enterprise AI Platform: Standard OpenAI Provider Implementation
"""

from __future__ import annotations
import os
import time
from typing import AsyncIterator, List, Optional, Type, TypeVar
from openai import AsyncOpenAI
from pydantic import BaseModel

from core.base import BaseLLMProvider
from core.schemas import (
    ChatMessage,
    LLMResponse,
    LLMStreamChunk,
    ProviderType,
    TokenUsage,
)
from core.telemetry import calculate_cost

T = TypeVar("T", bound=BaseModel)


class OpenAIProvider(BaseLLMProvider):
    """Direct OpenAI API Provider."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: str = "gpt-4o",
        default_embedding_model: str = "text-embedding-3-small",
    ):
        super().__init__(ProviderType.OPENAI, default_model)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.default_model = default_model
        self.embedding_model = default_embedding_model
        self._client: Optional[AsyncOpenAI] = None

    def _ensure_client(self) -> AsyncOpenAI:
        if not self._client:
            if not self.api_key:
                raise ValueError("OpenAI API key is missing. Set OPENAI_API_KEY.")
            self._client = AsyncOpenAI(api_key=self.api_key)
        return self._client

    async def generate(
        self,
        messages: List[ChatMessage],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        client = self._ensure_client()
        target_model = model or self.default_model
        formatted_messages = [{"role": m.role.value, "content": m.content} for m in messages]

        start_time = time.perf_counter()
        resp = await client.chat.completions.create(
            model=target_model,
            messages=formatted_messages,  # type: ignore
            temperature=temperature,
            max_tokens=max_tokens,
        )
        latency_ms = (time.perf_counter() - start_time) * 1000

        usage_info = resp.usage
        prompt_tokens = usage_info.prompt_tokens if usage_info else 0
        comp_tokens = usage_info.completion_tokens if usage_info else 0
        total_tokens = usage_info.total_tokens if usage_info else 0
        cost = calculate_cost(target_model, prompt_tokens, comp_tokens)

        return LLMResponse(
            content=resp.choices[0].message.content or "",
            model=target_model,
            provider=self.provider_type,
            latency_ms=round(latency_ms, 2),
            finish_reason=resp.choices[0].finish_reason,
            usage=TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=comp_tokens,
                total_tokens=total_tokens,
                estimated_cost_usd=cost,
            ),
        )

    async def generate_stream(
        self,
        messages: List[ChatMessage],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> AsyncIterator[LLMStreamChunk]:
        client = self._ensure_client()
        target_model = model or self.default_model
        formatted_messages = [{"role": m.role.value, "content": m.content} for m in messages]

        stream = await client.chat.completions.create(
            model=target_model,
            messages=formatted_messages,  # type: ignore
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        async for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices and chunk.choices[0].delta else ""
            finish_reason = chunk.choices[0].finish_reason if chunk.choices else None
            if delta:
                yield LLMStreamChunk(delta=delta, finish_reason=finish_reason)

    async def generate_structured(
        self,
        messages: List[ChatMessage],
        response_model: Type[T],
        model: Optional[str] = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        client = self._ensure_client()
        target_model = model or self.default_model
        formatted_messages = [{"role": m.role.value, "content": m.content} for m in messages]

        start_time = time.perf_counter()
        resp = await client.beta.chat.completions.parse(
            model=target_model,
            messages=formatted_messages,  # type: ignore
            response_format=response_model,
            temperature=temperature,
        )
        latency_ms = (time.perf_counter() - start_time) * 1000

        usage_info = resp.usage
        prompt_tokens = usage_info.prompt_tokens if usage_info else 0
        comp_tokens = usage_info.completion_tokens if usage_info else 0
        total_tokens = usage_info.total_tokens if usage_info else 0
        cost = calculate_cost(target_model, prompt_tokens, comp_tokens)

        return LLMResponse(
            content=resp.choices[0].message.content or "",
            parsed=resp.choices[0].message.parsed,
            model=target_model,
            provider=self.provider_type,
            latency_ms=round(latency_ms, 2),
            finish_reason=resp.choices[0].finish_reason,
            usage=TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=comp_tokens,
                total_tokens=total_tokens,
                estimated_cost_usd=cost,
            ),
        )

    async def embed(
        self,
        texts: List[str],
        model: Optional[str] = None,
    ) -> List[List[float]]:
        client = self._ensure_client()
        target_model = model or self.embedding_model
        resp = await client.embeddings.create(input=texts, model=target_model)
        return [item.embedding for item in resp.data]

    async def is_healthy(self) -> bool:
        if not self.api_key:
            return False
        try:
            await self.generate([ChatMessage(role=Role.USER, content="ping")], max_tokens=1)
            return True
        except Exception:
            return False
