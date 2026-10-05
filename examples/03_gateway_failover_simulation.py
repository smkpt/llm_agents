"""
Phase 1 - Lab 3: Zero-Downtime Gateway Failover Simulation
Demonstrates enterprise resilience when a primary LLM provider suffers an outage.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.schemas import ChatMessage, Role, ProviderType, RoutingPolicy
from core.router import EnterpriseAIGateway
from providers.mock_provider import MockLLMProvider


async def main():
    print("=" * 70)
    print("Enterprise AI Architecture: Circuit Breaker & Failover Drill")
    print("=" * 70)

    # 1. Simulate an unstable Primary Provider (e.g., Azure OpenAI experiencing 503 outage)
    failing_primary = MockLLMProvider(
        default_model="azure-gpt-4o",
        simulated_latency_ms=200.0,
        should_fail=True,  # Simulates rate limit (429) or outage (503)
    )

    # 2. Configure resilient secondary fallback provider (e.g., Claude 3.5 Sonnet or local fallback)
    healthy_fallback = MockLLMProvider(
        default_model="claude-3-5-sonnet-fallback",
        simulated_latency_ms=150.0,
        should_fail=False,
    )

    # 3. Setup Gateway with failover policy
    providers = {
        ProviderType.AZURE_OPENAI: failing_primary,
        ProviderType.ANTHROPIC: healthy_fallback,
    }

    policy = RoutingPolicy(
        primary_provider=ProviderType.AZURE_OPENAI,
        fallback_providers=[ProviderType.ANTHROPIC],
    )

    gateway = EnterpriseAIGateway(providers=providers, policy=policy)

    # 4. Execute request
    messages = [
        ChatMessage(role=Role.USER, content="Critical enterprise transaction analysis.")
    ]

    print("\n--> Sending mission-critical query with Azure OpenAI as primary...")
    response = await gateway.generate(messages=messages)

    print("\n[Execution Completed Without Downtime]")
    print(f"Final Responding Provider: {response.provider.value}")
    print(f"Responding Model:           {response.model}")
    print(f"Failover Occurred:          {response.failover_occurred}")
    print(f"Failover History Log:       {response.failover_history}")
    print(f"Response Content:           {response.content}")

    # 5. Review Gateway Metrics
    print("\n" + "=" * 70)
    print("Enterprise Gateway Telemetry Audit:")
    print(gateway.get_telemetry_report())
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
