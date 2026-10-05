"""
Enterprise AI Platform: FinOps Token Cost Calculation and Telemetry
"""

from __future__ import annotations
from typing import Dict, Tuple
from core.schemas import ProviderType, TokenUsage

# Pricing per 1,000,000 tokens (USD)
# (Prompt Cost, Completion Cost)
MODEL_PRICING_TABLE: Dict[str, Tuple[float, float]] = {
    # OpenAI & Azure OpenAI
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "o1": (15.00, 60.00),
    "o3-mini": (1.10, 4.40),
    # Anthropic
    "claude-3-5-sonnet-latest": (3.00, 15.00),
    "claude-3-5-haiku-latest": (0.80, 4.00),
    # Google Gemini
    "gemini-1.5-pro": (1.25, 5.00),
    "gemini-1.5-flash": (0.075, 0.30),
    "gemini-2.0-flash": (0.10, 0.40),
    # Local & Mock
    "llama3.1:8b": (0.00, 0.00),
    "mock-model": (0.00, 0.00),
}


def calculate_cost(model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Calculate the estimated USD cost for a given model and token count."""
    clean_model = model_name.lower()
    
    # Match pricing
    pricing = None
    for key, price in MODEL_PRICING_TABLE.items():
        if key in clean_model:
            pricing = price
            break
            
    if not pricing:
        pricing = (2.00, 8.00)  # Default fallback benchmark rate

    prompt_cost = (prompt_tokens / 1_000_000.0) * pricing[0]
    completion_cost = (completion_tokens / 1_000_000.0) * pricing[1]
    return round(prompt_cost + completion_cost, 6)


class TelemetryTracker:
    """Tracks cumulative enterprise gateway metrics: spend, tokens, requests, failovers."""

    def __init__(self):
        self.total_requests: int = 0
        self.total_tokens: int = 0
        self.total_cost_usd: float = 0.0
        self.cache_hits: int = 0
        self.failover_count: int = 0
        self.provider_request_counts: Dict[str, int] = {}

    def record_request(
        self,
        provider: ProviderType,
        usage: TokenUsage,
        latency_ms: float,
        cached: bool = False,
        failover: bool = False,
    ):
        self.total_requests += 1
        p_name = provider.value
        self.provider_request_counts[p_name] = self.provider_request_counts.get(p_name, 0) + 1

        if cached:
            self.cache_hits += 1
            return

        if failover:
            self.failover_count += 1

        self.total_tokens += usage.total_tokens
        self.total_cost_usd += usage.estimated_cost_usd

    def summary(self) -> Dict[str, object]:
        return {
            "total_requests": self.total_requests,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": f"{(self.cache_hits / max(1, self.total_requests)) * 100:.1f}%",
            "failover_count": self.failover_count,
            "total_tokens": self.total_tokens,
            "total_cost_usd": f"${self.total_cost_usd:.5f}",
            "providers_routed": self.provider_request_counts,
        }
