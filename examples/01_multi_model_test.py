"""
Phase 1 - Lab 1: Multi-Model Gateway Execution & Unified Interface Test
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from core.schemas import ChatMessage, Role, ProviderType
from core.router import EnterpriseAIGateway
from providers import (
    AzureOpenAIProvider,
    OpenAIProvider,
    ClaudeProvider,
    GeminiProvider,
    LocalLlamaProvider,
    MockLLMProvider,
)


async def main():
    print("=" * 70)
    print("Enterprise AI Architecture: Multi-Model Abstraction Layer Test")
    print("=" * 70)

    # 1. Initialize Providers
    # If live Azure credentials are not set in .env, we gracefully inject Mock providers
    azure_configured = bool(os.getenv("AZURE_OPENAI_ENDPOINT") and os.getenv("AZURE_OPENAI_API_KEY"))
    
    providers = {}
    if azure_configured:
        print("[+] Azure OpenAI credentials detected. Registering live Azure provider.")
        providers[ProviderType.AZURE_OPENAI] = AzureOpenAIProvider()
    else:
        print("[!] Azure OpenAI credentials not in .env. Registering High-Fidelity Mock for Azure.")
        providers[ProviderType.AZURE_OPENAI] = MockLLMProvider(default_model="azure-gpt-4o")

    # Secondary / Local providers
    providers[ProviderType.MOCK] = MockLLMProvider(default_model="mock-fast-model")
    providers[ProviderType.LOCAL_LLAMA] = LocalLlamaProvider()

    # 2. Instantiate Enterprise Gateway
    gateway = EnterpriseAIGateway(providers=providers)

    # 3. Test Unified Query Interface
    messages = [
        ChatMessage(
            role=Role.SYSTEM,
            content="You are an Enterprise AI Architecture assistant.",
        ),
        ChatMessage(
            role=Role.USER,
            content="Explain the trade-off between RAG and Long Context Windows in 2 sentences.",
        ),
    ]

    print("\n--> Sending query through Enterprise Gateway...")
    response = await gateway.generate(messages=messages, temperature=0.2)

    print(f"\n[Response Received]")
    print(f"Provider:    {response.provider.value}")
    print(f"Model:       {response.model}")
    print(f"Latency:     {response.latency_ms:.2f} ms")
    print(f"Tokens:      {response.usage.total_tokens} (Prompt: {response.usage.prompt_tokens}, Comp: {response.usage.completion_tokens})")
    print(f"Est Cost:    ${response.usage.estimated_cost_usd:.6f}")
    print(f"Content:\n{response.content}\n")

    # 4. Test Semantic Caching Speedup
    print("--> Repeating identical query to verify Gateway Cache...")
    cached_response = await gateway.generate(messages=messages, temperature=0.2)
    print(f"Cache Hit:   {cached_response.cached}")
    print(f"Latency:     {cached_response.latency_ms} ms (Expected < 1ms)")

    # 5. Review Gateway FinOps Dashboard
    print("\n" + "=" * 70)
    print("Gateway Telemetry & FinOps Metrics:")
    print(gateway.get_telemetry_report())
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
