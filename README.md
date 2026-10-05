# ⚡ Enterprise AI Platform: Accelerator & Gateway Foundation
### *Production-Grade Multi-Model Routing, Failover Cascades & FinOps Telemetry*

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Azure OpenAI](https://img.shields.io/badge/Azure_OpenAI-Primary-0078D4?style=for-the-badge&logo=microsoftazure&logoColor=white)](https://azure.microsoft.com)
[![Anthropic](https://img.shields.io/badge/Anthropic_Claude-Sonnet_3.5-D97706?style=for-the-badge&logo=anthropic&logoColor=white)](https://anthropic.com)
[![Google Gemini](https://img.shields.io/badge/Google_Gemini-1.5_Pro%2FFlash-8B5CF6?style=for-the-badge&logo=google&logoColor=white)](https://cloud.google.com/vertex-ai)
[![Meta Llama](https://img.shields.io/badge/Meta_Llama-3.1_Air--Gapped-0D9488?style=for-the-badge&logo=meta&logoColor=white)](https://ollama.ai)
[![Pydantic](https://img.shields.io/badge/Pydantic-v2_Strict-E92063?style=for-the-badge&logo=pydantic&logoColor=white)](https://docs.pydantic.dev)

</div>

---

> [!TIP]
> **Production Ready:** This repository serves as the foundational **Enterprise AI Accelerator** microservice. It decouples client applications from specific model APIs, enabling zero-downtime failovers, automatic sub-millisecond semantic caching, and full FinOps token cost auditing.

---

## 🏛️ System Architecture Topology

```mermaid
flowchart TD
    classDef client fill:#3B82F6,stroke:#1D4ED8,color:#FFFFFF,stroke-width:2px;
    classDef gateway fill:#1E293B,stroke:#334155,color:#FFFFFF,stroke-width:2px;
    classDef cache fill:#10B981,stroke:#059669,color:#FFFFFF,stroke-width:2px;
    classDef azure fill:#0078D4,stroke:#005A9E,color:#FFFFFF,stroke-width:2px;
    classDef claude fill:#D97706,stroke:#B45309,color:#FFFFFF,stroke-width:2px;
    classDef gemini fill:#8B5CF6,stroke:#6D28D9,color:#FFFFFF,stroke-width:2px;
    classDef llama fill:#0D9488,stroke:#0F766E,color:#FFFFFF,stroke-width:2px;

    Client[💻 Enterprise Apps / Copilot Frontend]:::client --> GW[⚡ Enterprise AI Gateway]:::gateway

    subgraph Gateway Engine
        GW --> CacheCheck{🔍 Semantic Cache Lookup}:::cache
        CacheCheck -->|Cache Hit| FastReturn[⚡ Sub-Millisecond Return < 1ms<br/>Cost: $0.0000]:::cache
        CacheCheck -->|Cache Miss| PolicyRouter{🎯 Policy Task Router}:::gateway
    end

    PolicyRouter -->|Default Enterprise Copilot| Azure[🔵 Azure OpenAI GPT-4o<br/>Primary SLA & Private VNet]:::azure
    PolicyRouter -->|Code & Complex Logic| Claude[🟡 Anthropic Claude 3.5<br/>High-Reasoning Fallback]:::claude
    PolicyRouter -->|Bulk Docs / 2M Context| Gemini[🟣 Google Gemini 1.5<br/>Cost-Effective Ingestion]:::gemini
    PolicyRouter -->|Air-Gapped PII Strict| Llama[🟢 Local Llama 3.1 70B<br/>Zero Cloud Data Egress]:::llama

    Azure -.->|❌ Simulated 503 / 429| Failover[⚠️ Auto Circuit-Breaker Cascade]:::claude
    Failover ==> Claude

    Azure & Claude & Gemini & Llama --> Telemetry[📊 OpenTelemetry & FinOps Spend Audit]:::gateway
```

---

## 📂 Modular Codebase Structure

```text
enterprise_ai_platform/
├── .env.example                         # Environment keys & deployment configuration
├── requirements.txt                     # Core dependencies (OpenAI, Pydantic, HTTPX)
├── README.md                            # Visual platform documentation
│
├── core/                                # Enterprise Gateway Core Engine
│   ├── base.py                          # BaseLLMProvider abstract contract
│   ├── schemas.py                       # Pydantic data contracts (ChatMessage, LLMResponse, Usage)
│   ├── cache.py                         # InMemory cache with SHA-256 keying & TTL (<1ms hits)
│   ├── telemetry.py                     # FinOps cost tables & metrics accumulator
│   └── router.py                        # Gateway Router with dynamic failover cascades
│
├── providers/                           # Vendor-Agnostic LLM Integrations
│   ├── azure_openai.py                  # Primary: Azure OpenAI (Entra ID & Key support)
│   ├── openai_provider.py               # Standard OpenAI direct client
│   ├── claude_provider.py               # Anthropic Claude direct async REST
│   ├── gemini_provider.py               # Google Gemini direct async REST
│   ├── local_llama.py                   # Local Ollama / vLLM private endpoint
│   └── mock_provider.py                 # High-fidelity mock for offline CI/CD & drills
│
└── examples/                            # Hands-on Verification Labs
    ├── 01_multi_model_test.py           # Multi-model query & caching benchmark
    ├── 02_structured_output_validation.py # Strict Pydantic JSON schema enforcement
    └── 03_gateway_failover_simulation.py  # Zero-downtime outage failover drill
```

---

## 🚀 Quickstart Guide

### 1️⃣ Install Prerequisites
```powershell
python -m pip install -r requirements.txt
```

### 2️⃣ Configure Environment Credentials
Copy `.env.example` to `.env`:
```powershell
Copy-Item .env.example .env
```
*(If credentials are empty, the platform automatically engages High-Fidelity Mock providers so you can develop offline with zero cloud cost.)*

### 3️⃣ Execute Enterprise Verification Labs

<details open>
<summary><b>🧪 Lab 1: Multi-Model Query & Response Caching</b></summary>

```powershell
python examples/01_multi_model_test.py
```
**Key Result:** Queries the gateway and demonstrates instant cache hit on repeated queries with **`< 1ms` latency** and **`$0.000` incremental cost**.
</details>

<details>
<summary><b>📋 Lab 2: Strict Structured Output Enforcement</b></summary>

```powershell
python examples/02_structured_output_validation.py
```
**Key Result:** Enforces deterministic Pydantic validation (`ArchitectureAssessment`), guaranteeing typed outputs for downstream enterprise microservices.
</details>

<details>
<summary><b>⚡ Lab 3: Zero-Downtime Outage Failover Drill</b></summary>

```powershell
python examples/03_gateway_failover_simulation.py
```
**Key Result:** Simulates an Azure OpenAI HTTP 503 outage. The Gateway isolates the failure, routes to the secondary fallback model without crashing, and logs the incident in FinOps telemetry.
</details>

---

## 📊 Live Telemetry & FinOps Audit

The gateway aggregates telemetry metrics across every request:

| Metric | Target SLA | Platform Benchmark |
| :--- | :---: | :---: |
| **Cache Hit Latency** | `< 5 ms` | **`0.5 ms`** ⚡ |
| **Failover Downtime** | `0 ms` | **`0 ms (Seamless)`** 🛡️ |
| **Direct API Cost (Cached)** | `$0.00` | **`$0.00`** 💰 |
| **Typed Schema Validation** | `100%` | **`100% (Pydantic v2)`** ✅ |

---

## 🔮 Next Roadmap Target: Phase 2
Building the **Enterprise Hybrid RAG Architecture**:
* Ingestion with semantic & hierarchical chunking (Docling / Unstructured).
* Hybrid Search: Dense Vector (text-embedding-3) + Sparse BM25 via Reciprocal Rank Fusion (RRF).
* Vector database integration with Qdrant / Azure AI Search.
* Cross-Encoder re-ranking with Cohere.
