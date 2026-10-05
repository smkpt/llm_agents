"""
Enterprise AI Platform: Google Gemini Provider Implementation (HTTP REST)
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
    Role,
    TokenUsage,
)
from core.telemetry import calculate_cost

T = TypeVar("T", bound=BaseModel)


class GeminiProvider(BaseLLMProvider):
    """Google Gemini Provider using asynchronous REST API."""

    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: str = "gemini-1.5-flash",
    ):
        super().__init__(ProviderType.GEMINI, default_model)
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")

    def _format_contents(self, messages: List[ChatMessage]) -> tuple[Optional[dict], list]:
        system_text = "\n\n".join(m.content for m in messages if m.role == Role.SYSTEM)
        system_instruction = (
            {"parts": [{"text": system_text}]} if system_text else None
        )

        contents = []
        for m in messages:
            if m.role == Role.SYSTEM:
                continue
            role = "user" if m.role == Role.USER else "model"
            contents.append({"role": role, "parts": [{"text": m.content}]})

        return system_instruction, contents

    async def generate(
        self,
        messages: List[ChatMessage],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        if not self.api_key:
            raise ValueError("Gemini API key is missing. Set GEMINI_API_KEY.")

        target_model = model or self.default_model
        system_instruction, contents = self._format_contents(messages)

        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        if system_instruction:
            payload["systemInstruction"] = system_instruction

        url = f"{self.BASE_URL}/{target_model}:generateContent?key={self.api_key}"

        start_time = time.perf_counter()
        async with httpx.AsyncClient(timeout=45.0) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
        latency_ms = (time.perf_counter() - start_time) * 1000

        text_content = ""
        candidates = data.get("candidates", [])
        finish_reason = None
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            text_content = "".join(p.get("text", "") for p in parts)
            finish_reason = candidates[0].get("finishReason")

        usage_meta = data.get("usageMetadata", {})
        prompt_tokens = usage_meta.get("promptTokenCount", 0)
        comp_tokens = usage_meta.get("candidatesTokenCount", 0)
        total_tokens = usage_meta.get("totalTokenCount", prompt_tokens + comp_tokens)
        cost = calculate_cost(target_model, prompt_tokens, comp_tokens)

        return LLMResponse(
            content=text_content,
            model=target_model,
            provider=self.provider_type,
            latency_ms=round(latency_ms, 2),
            finish_reason=finish_reason,
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
        # Gemini SSE stream endpoint
        target_model = model or self.default_model
        system_instruction, contents = self._format_contents(messages)

        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        if system_instruction:
            payload["systemInstruction"] = system_instruction

        url = f"{self.BASE_URL}/{target_model}:streamGenerateContent?alt=sse&key={self.api_key}"

        async with httpx.AsyncClient(timeout=45.0) as client:
            async with client.stream("POST", url, json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        if not data_str:
                            continue
                        try:
                            chunk_data = json.loads(data_str)
                            candidates = chunk_data.get("candidates", [])
                            if candidates:
                                parts = candidates[0].get("content", {}).get("parts", [])
                                delta = "".join(p.get("text", "") for p in parts)
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
        target_model = model or self.default_model
        system_instruction, contents = self._format_contents(messages)

        # Gemini supports responseSchema in generationConfig
        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "responseMimeType": "application/json",
                "responseSchema": response_model.model_json_schema(),
            },
        }
        if system_instruction:
            payload["systemInstruction"] = system_instruction

        url = f"{self.BASE_URL}/{target_model}:generateContent?key={self.api_key}"

        start_time = time.perf_counter()
        async with httpx.AsyncClient(timeout=45.0) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
        latency_ms = (time.perf_counter() - start_time) * 1000

        text_content = ""
        candidates = data.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            text_content = "".join(p.get("text", "") for p in parts)

        parsed_obj = None
        if text_content:
            parsed_obj = response_model.model_validate_json(text_content)

        usage_meta = data.get("usageMetadata", {})
        prompt_tokens = usage_meta.get("promptTokenCount", 0)
        comp_tokens = usage_meta.get("candidatesTokenCount", 0)
        total_tokens = usage_meta.get("totalTokenCount", prompt_tokens + comp_tokens)
        cost = calculate_cost(target_model, prompt_tokens, comp_tokens)

        return LLMResponse(
            content=text_content,
            parsed=parsed_obj,
            model=target_model,
            provider=self.provider_type,
            latency_ms=round(latency_ms, 2),
            finish_reason=candidates[0].get("finishReason") if candidates else None,
            usage=TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=comp_tokens,
                total_tokens=total_tokens,
                estimated_cost_usd=cost,
            ),
        )

    async def embed(self, texts: List[str], model: Optional[str] = None) -> List[List[float]]:
        target_model = model or "text-embedding-004"
        url = f"{self.BASE_URL}/{target_model}:batchEmbedContents?key={self.api_key}"
        requests_payload = [
            {"model": f"models/{target_model}", "content": {"parts": [{"text": t}]}}
            for t in texts
        ]
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, json={"requests": requests_payload})
            resp.raise_for_status()
            data = resp.json()

        embeddings = []
        for item in data.get("embeddings", []):
            embeddings.append(item.get("values", []))
        return embeddings

    async def is_healthy(self) -> bool:
        return bool(self.api_key)
