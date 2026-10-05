"""
Enterprise AI Platform: Response Caching Layer
"""

from __future__ import annotations
import hashlib
import time
from typing import Dict, List, Optional
from core.schemas import ChatMessage, LLMResponse


class BaseCache:
    """Abstract interface for enterprise LLM response caching."""

    def get(self, key: str) -> Optional[LLMResponse]:
        raise NotImplementedError

    def set(self, key: str, value: LLMResponse, ttl_seconds: int = 3600) -> None:
        raise NotImplementedError


class InMemoryCache(BaseCache):
    """In-memory cache with TTL expiration and hash keying."""

    def __init__(self):
        self._store: Dict[str, tuple[LLMResponse, float]] = {}

    @staticmethod
    def generate_key(messages: List[ChatMessage], model: str, temperature: float = 0.0) -> str:
        """Create a deterministic SHA-256 hash from conversation history and parameters."""
        raw_repr = f"{model}:{temperature}:" + "|".join(
            f"{m.role.value}:{m.content.strip()}" for m in messages
        )
        return hashlib.sha256(raw_repr.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[LLMResponse]:
        if key not in self._store:
            return None
        response, expires_at = self._store[key]
        if time.time() > expires_at:
            del self._store[key]
            return None
        # Return a copy marked as cached
        cached_resp = response.model_copy()
        cached_resp.cached = True
        cached_resp.latency_ms = 0.5  # Instant cache hit
        return cached_resp

    def set(self, key: str, value: LLMResponse, ttl_seconds: int = 3600) -> None:
        expires_at = time.time() + ttl_seconds
        self._store[key] = (value, expires_at)

    def clear(self) -> None:
        self._store.clear()
