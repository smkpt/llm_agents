"""
Enterprise AI Platform: Standardized Core Data Contracts and Schemas
"""

from __future__ import annotations
from enum import Enum
from typing import Any, Dict, List, Optional, TypeVar
from pydantic import BaseModel, Field

T = TypeVar("T", bound=BaseModel)


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ProviderType(str, Enum):
    AZURE_OPENAI = "azure_openai"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    LOCAL_LLAMA = "local_llama"
    MOCK = "mock"


class ChatMessage(BaseModel):
    role: Role
    content: str
    name: Optional[str] = None


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0


class LLMResponse(BaseModel):
    content: str
    parsed: Optional[Any] = None
    model: str
    provider: ProviderType
    usage: TokenUsage = Field(default_factory=TokenUsage)
    latency_ms: float = 0.0
    finish_reason: Optional[str] = None
    cached: bool = False
    failover_occurred: bool = False
    failover_history: List[str] = Field(default_factory=list)


class LLMStreamChunk(BaseModel):
    delta: str
    finish_reason: Optional[str] = None


class RoutingPolicy(BaseModel):
    primary_provider: ProviderType = ProviderType.AZURE_OPENAI
    fallback_providers: List[ProviderType] = Field(
        default_factory=lambda: [
            ProviderType.ANTHROPIC,
            ProviderType.OPENAI,
            ProviderType.LOCAL_LLAMA,
            ProviderType.MOCK,
        ]
    )
    timeout_seconds: float = 25.0
    retry_attempts: int = 2
    task_routing_map: Dict[str, ProviderType] = Field(
        default_factory=lambda: {
            "code_generation": ProviderType.ANTHROPIC,
            "deep_reasoning": ProviderType.ANTHROPIC,
            "large_document": ProviderType.GEMINI,
            "pii_restricted": ProviderType.LOCAL_LLAMA,
            "default": ProviderType.AZURE_OPENAI,
        }
    )
