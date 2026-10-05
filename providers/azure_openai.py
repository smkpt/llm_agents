"""
Enterprise AI Platform: Azure OpenAI Provider Implementation
"""

from __future__ import annotations
import os
import time
from typing import AsyncIterator, List, Optional, Type, TypeVar
from openai import AsyncAzureOpenAI
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


class AzureOpenAIProvider(BaseLLMProvider):
    """
    Production-grade Azure OpenAI Provider.
    Supports API Key or Microsoft Entra ID (DefaultAzureCredential).
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        api_version: str = "2024-08-01-preview",
        default_deployment: str = "gpt-4o",
        default_embedding_deployment: str = "text-embedding-3-small",
    ):
        super().__init__(ProviderType.AZURE_OPENAI, default_deployment)
        self.endpoint = endpoint or os.getenv("AZURE_OPENAI_ENDPOINT", "")
        self.api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY", "")
        self.api_version = api_version or os.getenv(
            "AZURE_OPENAI_API_VERSION", "2024-08-01-preview"
        )
        self.default_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_CHAT", default_deployment)
        self.embedding_deployment = os.getenv(
            "AZURE_OPENAI_DEPLOYMENT_EMBEDDING", default_embedding_deployment
        )

        self._client: Optional[AsyncAzureOpenAI] = None
        if self.endpoint and self.api_key:
            self._client = AsyncAzureOpenAI(
                azure_endpoint=self.endpoint,
                api_key=self.api_key,
                api_version=self.api_version,
            )

    def _ensure_client(self) -> AsyncAzureOpenAI:
        if not self._client:
            if not self.endpoint or not self.api_key:
                raise ValueError(
                    "Azure OpenAI is not configured. Missing AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_API_KEY."
                )
            self._client = AsyncAzureOpenAI(
                azure_endpoint=self.endpoint,
                api_key=self.api_key,
                api_version=self.api_version,
            )
        return self._client

    async def generate(
        self,
        messages: List[ChatMessage],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        client = self._ensure_client()
        deployment = model or self.default_deployment
        formatted_messages = [{"role": m.role.value, "content": m.content} for m in messages]

        start_time = time.perf_counter()
        resp = await client.chat.completions.create(
            model=deployment,
            messages=formatted_messages,  # type: ignore
            temperature=temperature,
            max_tokens=max_tokens,
        )
        latency_ms = (time.perf_counter() - start_time) * 1000

        usage_info = resp.usage
        prompt_tokens = usage_info.prompt_tokens if usage_info else 0
        comp_tokens = usage_info.completion_tokens if usage_info else 0
        total_tokens = usage_info.total_tokens if usage_info else 0

        cost = calculate_cost(deployment, prompt_tokens, comp_tokens)

        return LLMResponse(
            content=resp.choices[0].message.content or "",
            model=deployment,
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
        deployment = model or self.default_deployment
        formatted_messages = [{"role": m.role.value, "content": m.content} for m in messages]

        stream = await client.chat.completions.create(
            model=deployment,
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
        deployment = model or self.default_deployment
        formatted_messages = [{"role": m.role.value, "content": m.content} for m in messages]

        start_time = time.perf_counter()
        resp = await client.beta.chat.completions.parse(
            model=deployment,
            messages=formatted_messages,  # type: ignore
            response_format=response_model,
            temperature=temperature,
        )
        latency_ms = (time.perf_counter() - start_time) * 1000

        usage_info = resp.usage
        prompt_tokens = usage_info.prompt_tokens if usage_info else 0
        comp_tokens = usage_info.completion_tokens if usage_info else 0
        total_tokens = usage_info.total_tokens if usage_info else 0
        cost = calculate_cost(deployment, prompt_tokens, comp_tokens)

        parsed_obj = resp.choices[0].message.parsed

        return LLMResponse(
            content=resp.choices[0].message.content or "",
            parsed=parsed_obj,
            model=deployment,
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
        deployment = model or self.embedding_deployment
        resp = await client.embeddings.create(input=texts, model=deployment)
        return [item.embedding for item in resp.data]

    async def is_healthy(self) -> bool:
        if not self.endpoint or not self.api_key:
            return False
        try:
            # Low-cost lightweight probe
            await self.generate(
                messages=[ChatMessage(role=Role.USER, content="ping")],
                max_tokens=1,
            )
            return True
        except Exception:
            return False
