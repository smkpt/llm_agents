"""
Phase 1 - Lab 2: Structured Output & Schema Enforcement
Enforces deterministic typed data contracts for downstream microservices.
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import List
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from core.schemas import ChatMessage, Role, ProviderType
from core.router import EnterpriseAIGateway
from providers import AzureOpenAIProvider, MockLLMProvider


# --- 1. Define Strict Enterprise Data Contract ---
class ArchitectureAssessment(BaseModel):
    system_name: str = Field(description="Name of the enterprise system being reviewed")
    primary_architecture_pattern: str = Field(
        description="Core architecture pattern (e.g., Hybrid RAG, Multi-Agent Supervisor)"
    )
    compliance_score: float = Field(
        description="Security and compliance readiness score from 0.0 to 100.0"
    )
    key_risks: List[str] = Field(description="Identified architectural risks or bottlenecks")
    mitigations: List[str] = Field(description="Recommended architectural mitigations")


async def main():
    print("=" * 70)
    print("Enterprise AI Architecture: Strict Structured Output Enforcement")
    print("=" * 70)

    # Provider setup
    azure_configured = bool(os.getenv("AZURE_OPENAI_ENDPOINT") and os.getenv("AZURE_OPENAI_API_KEY"))
    if azure_configured:
        provider = AzureOpenAIProvider()
    else:
        print("[!] Using High-Fidelity Mock for deterministic testing.")
        provider = MockLLMProvider(default_model="azure-gpt-4o")

    gateway = EnterpriseAIGateway(providers={ProviderType.AZURE_OPENAI: provider})

    messages = [
        ChatMessage(
            role=Role.SYSTEM,
            content="You are a Principal AI Solution Architect auditing system designs.",
        ),
        ChatMessage(
            role=Role.USER,
            content=(
                "Evaluate an AI customer service platform handling 100,000 queries/day "
                "with sensitive PII data and high availability requirements."
            ),
        ),
    ]

    print("\n--> Requesting structured architecture assessment...")
    response = await gateway.generate_structured(
        messages=messages,
        response_model=ArchitectureAssessment,
    )

    assessment: ArchitectureAssessment = response.parsed

    print("\n[Structured Object Successfully Parsed and Validated]")
    print(f"System:              {assessment.system_name}")
    print(f"Primary Pattern:     {assessment.primary_architecture_pattern}")
    print(f"Compliance Score:    {assessment.compliance_score} / 100")
    print(f"Key Risks:           {assessment.key_risks}")
    print(f"Mitigations:         {assessment.mitigations}")
    print("\nRaw JSON representation:")
    print(assessment.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
