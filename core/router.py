"""
Enterprise AI Platform: Gateway Router with Resilient Fallbacks, Caching & Telemetry
"""

from __future__ import annotations
import logging
from typing import AsyncIterator, Dict, List, Optional, Type, TypeVar
from pydantic import BaseModel

from core.base import BaseLLMProvider
from core.cache import BaseCache, InMemoryCache
from core.schemas import (
    ChatMessage,
    LLMResponse,
    LLMStreamChunk,
    ProviderType,
    RoutingPolicy,
)
from core.telemetry import TelemetryTracker

logger = logging.getLogger("EnterpriseAIGateway")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

T = TypeVar("T", bound=BaseModel)


class EnterpriseAIGateway:
    """
    Production Enterprise AI Gateway.
    Orchestrates Multi-LLM routing, automatic circuit-breaking fallbacks,
    caching, structured enforcement, and FinOps telemetry.
    """

    def __init__(
        self,
        providers: Dict[ProviderType, BaseLLMProvider],
        policy: Optional[RoutingPolicy] = None,
        cache: Optional[BaseCache] = None,
        telemetry: Optional[TelemetryTracker] = None,
    ):
        self.providers = providers
        self.policy = policy or RoutingPolicy()
        self.cache = cache or InMemoryCache()
        self.telemetry = telemetry or TelemetryTracker()

    def register_provider(self, provider: BaseLLMProvider):
        self.providers[provider.provider_type] = provider

    def _determine_route(self, task_type: Optional[str] = None) -> List[ProviderType]:
        """Determine primary target provider and fallback priority chain."""
        primary = self.policy.primary_provider
        if task_type and task_type in self.policy.task_routing_map:
            primary = self.policy.task_routing_map[task_type]

        # Build execution order: [primary, *fallbacks]
        route_chain = [primary]
        for fb in self.policy.fallback_providers:
            if fb not in route_chain:
                route_chain.append(fb)

        # Filter only registered providers
        return [p for p in route_chain if p in self.providers]

    async def generate(
        self,
        messages: List[ChatMessage],
        task_type: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        skip_cache: bool = False,
    ) -> LLMResponse:
        """Route request through primary provider with automatic failover cascade."""
        cache_key = ""
        if not skip_cache and isinstance(self.cache, InMemoryCache):
            cache_key = self.cache.generate_key(messages, model or "default", temperature)
            cached_resp = self.cache.get(cache_key)
            if cached_resp:
                logger.info("Semantic cache HIT for query. Serving from cache (<1ms).")
                self.telemetry.record_request(
                    cached_resp.provider, cached_resp.usage, 0.5, cached=True
                )
                return cached_resp

        route_chain = self._determine_route(task_type)
        if not route_chain:
            raise RuntimeError("No available LLM providers registered in Gateway.")

        failover_history: List[str] = []
        last_exception: Optional[Exception] = None

        for idx, provider_type in enumerate(route_chain):
            provider = self.providers[provider_type]
            is_failover = idx > 0

            try:
                if is_failover:
                    logger.warning(
                        f"Initiating Gateway Failover to secondary provider: {provider_type.value}"
                    )

                resp = await provider.generate(
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                if is_failover:
                    resp.failover_occurred = True
                    resp.failover_history = failover_history

                # Record telemetry
                self.telemetry.record_request(
                    provider=provider_type,
                    usage=resp.usage,
                    latency_ms=resp.latency_ms,
                    cached=False,
                    failover=is_failover,
                )

                # Store in cache
                if not skip_cache and cache_key:
                    self.cache.set(cache_key, resp)

                return resp

            except Exception as e:
                logger.error(
                    f"Provider [{provider_type.value}] execution failed: {str(e)}"
                )
                failover_history.append(f"{provider_type.value}: {str(e)}")
                last_exception = e

        raise RuntimeError(
            f"All providers in fallback chain exhausted! Failover log: {failover_history}"
        ) from last_exception

    async def generate_structured(
        self,
        messages: List[ChatMessage],
        response_model: Type[T],
        task_type: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        """Execute structured output generation with failover support."""
        route_chain = self._determine_route(task_type)
        failover_history: List[str] = []

        for idx, provider_type in enumerate(route_chain):
            provider = self.providers[provider_type]
            is_failover = idx > 0
            try:
                resp = await provider.generate_structured(
                    messages=messages,
                    response_model=response_model,
                    model=model,
                    temperature=temperature,
                )
                if is_failover:
                    resp.failover_occurred = True
                    resp.failover_history = failover_history

                self.telemetry.record_request(
                    provider=provider_type,
                    usage=resp.usage,
                    latency_ms=resp.latency_ms,
                    cached=False,
                    failover=is_failover,
                )
                return resp
            except Exception as e:
                logger.error(f"Structured output failed on [{provider_type.value}]: {str(e)}")
                failover_history.append(f"{provider_type.value}: {str(e)}")

        raise RuntimeError(f"All providers failed structured generation: {failover_history}")

    async def generate_stream(
        self,
        messages: List[ChatMessage],
        task_type: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> AsyncIterator[LLMStreamChunk]:
        """Stream chunks from active provider."""
        route_chain = self._determine_route(task_type)
        for provider_type in route_chain:
            provider = self.providers[provider_type]
            try:
                async for chunk in provider.generate_stream(
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                ):
                    yield chunk
                return
            except Exception as e:
                logger.error(f"Streaming failed on [{provider_type.value}]: {e}. Attempting next provider.")
                continue

        raise RuntimeError("All providers failed during streaming.")

    def get_telemetry_report(self) -> Dict[str, object]:
        """Return FinOps dashboard metrics."""
        return self.telemetry.summary()
