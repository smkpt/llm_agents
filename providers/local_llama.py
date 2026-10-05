"""
Enterprise AI Platform: Local / Air-Gapped Llama Provider (Ollama / vLLM)
"""

from __future__ import annotations
import json
import os
import time
from typing import AsyncIterator, List, Optional, Type, TypeVar
import httpx
from pydantic import BaseModel

from core.base import BaseLLMProvider
from core.schemas import (
    ChatMessage,
    LLMResponse,
    LLMStreamChunk,
    ProviderType,
    TokenUsage,
)

T = TypeVar("T", bound=BaseModel)


class LocalLlamaProvider(BaseLLMProvider):
    """
    Local and Air-Gapped Llama inference using vLLM or Ollama OpenAI-compatible API.
    Zero cloud data egress for sensitive enterprise PII.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        default_model: str = "llama3.1:8b",
    ):
        super().__init__(ProviderType.LOCAL_LLAMA, default_model)
        self.base_url = (base_url or os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:11434/v1")).rstrip("/")
        self.default_model = os.getenv("LOCAL_LLM_MODEL", default_model)

    async def generate(
        self,
        messages: List[ChatMessage],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        target_model = model or self.default_model
        payload = {
            "model": target_model,
            "messages": [{"role": m.role.value, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        start_time = time.perf_counter()
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(f"{self.base_url}/chat/completions", json=payload)
            resp.raise_for_status()
            data = resp.json()
        latency_ms = (time.perf_counter() - start_time) * 1000

        content = data["choices"][0]["message"]["content"]
        finish_reason = data["choices"][0].get("finish_reason")
        usage_info = data.get("usage", {})
        prompt_tokens = usage_info.get("prompt_tokens", 0)
        comp_tokens = usage_info.get("completion_tokens", 0)
        total_tokens = usage_info.get("total_tokens", prompt_tokens + comp_tokens)

        return LLMResponse(
            content=content,
            model=target_model,
            provider=self.provider_type,
            latency_ms=round(latency_ms, 2),
            finish_reason=finish_reason,
            usage=TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=comp_tokens,
                total_tokens=total_tokens,
                estimated_cost_usd=0.0,  # Zero direct API cost
            ),
        )

    async def generate_stream(
        self,
        messages: List[ChatMessage],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> AsyncIterator[LLMStreamChunk]:
        target_model = model or self.default_model
        payload = {
            "model": target_model,
            "messages": [{"role": m.role.value, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("POST", f"{self.base_url}/chat/completions", json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        if not data_str or data_str == "[DONE]":
                            continue
                        try:
                            chunk = json.loads(data_str)
                            delta = chunk["choices"][0].get("delta", {}).get("content", "")
                            if delta:
                                yield LLMStreamChunk(delta=delta)
                        except Exception:
                            pass

    async def generate_structured(
        self,
        messages: List[ChatMessage],
        response_model: Type[T],
        model: Optional[str] = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        # Prompt injection of schema for open-source models supporting JSON mode
        target_model = model or self.default_model
        schema = response_model.model_json_schema()
        
        system_instruction = (
            f"You must respond ONLY with a valid JSON object strictly matching this JSON schema:\n"
            f"{json.dumps(schema)}\nDo not include code markdown ticks or explanatory text."
        )

        prompt_messages = [{"role": "system", "content": system_instruction}] + [
            {"role": m.role.value, "content": m.content} for m in messages
        ]

        payload = {
            "model": target_model,
            "messages": prompt_messages,
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        }

        start_time = time.perf_counter()
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(f"{self.base_url}/chat/completions", json=payload)
            resp.raise_for_status()
            data = resp.json()
        latency_ms = (time.perf_counter() - start_time) * 1000

        content = data["choices"][0]["message"]["content"]
        parsed_obj = response_model.model_validate_json(content)

        usage_info = data.get("usage", {})
        prompt_tokens = usage_info.get("prompt_tokens", 0)
        comp_tokens = usage_info.get("completion_tokens", 0)

        return LLMResponse(
            content=content,
            parsed=parsed_obj,
            model=target_model,
            provider=self.provider_type,
            latency_ms=round(latency_ms, 2),
            usage=TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=comp_tokens,
                total_tokens=prompt_tokens + comp_tokens,
                estimated_cost_usd=0.0,
            ),
        )

    async def embed(self, texts: List[str], model: Optional[str] = None) -> List[List[float]]:
        target_model = model or "nomic-embed-text"
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{self.base_url}/embeddings",
                json={"model": target_model, "input": texts},
            )
            resp.raise_for_status()
            data = resp.json()
        return [item["embedding"] for item in data.get("data", [])]

    async def is_healthy(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                resp = await client.get(f"{self.base_url}/models")
                return resp.status_code == 200
        except Exception:
            return False
