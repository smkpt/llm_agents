"""
Enterprise AI Platform: Anthropic Claude Provider Implementation (HTTP REST)
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


class ClaudeProvider(BaseLLMProvider):
    """Anthropic Claude Provider via direct async HTTP."""

    BASE_URL = "https://api.anthropic.com/v1/messages"

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: str = "claude-3-5-sonnet-latest",
    ):
        super().__init__(ProviderType.ANTHROPIC, default_model)
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")

    def _get_headers(self) -> dict:
        if not self.api_key:
            raise ValueError("Anthropic API key is missing. Set ANTHROPIC_API_KEY.")
        return {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

    async def generate(
        self,
        messages: List[ChatMessage],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        headers = self._get_headers()
        target_model = model or self.default_model

        # Separate system messages from user/assistant turns
        system_content = "\n\n".join(m.content for m in messages if m.role == Role.SYSTEM)
        convo_messages = [
            {"role": "user" if m.role == Role.USER else "assistant", "content": m.content}
            for m in messages
            if m.role != Role.SYSTEM
        ]

        payload = {
            "model": target_model,
            "messages": convo_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system_content:
            payload["system"] = system_content

        start_time = time.perf_counter()
        async with httpx.AsyncClient(timeout=45.0) as client:
            resp = await client.post(self.BASE_URL, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
        latency_ms = (time.perf_counter() - start_time) * 1000

        text_content = ""
        for block in data.get("content", []):
            if block.get("type") == "text":
                text_content += block.get("text", "")

        usage_data = data.get("usage", {})
        prompt_tokens = usage_data.get("input_tokens", 0)
        comp_tokens = usage_data.get("output_tokens", 0)
        total_tokens = prompt_tokens + comp_tokens
        cost = calculate_cost(target_model, prompt_tokens, comp_tokens)

        return LLMResponse(
            content=text_content,
            model=target_model,
            provider=self.provider_type,
            latency_ms=round(latency_ms, 2),
            finish_reason=data.get("stop_reason"),
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
        headers = self._get_headers()
        target_model = model or self.default_model

        system_content = "\n\n".join(m.content for m in messages if m.role == Role.SYSTEM)
        convo_messages = [
            {"role": "user" if m.role == Role.USER else "assistant", "content": m.content}
            for m in messages
            if m.role != Role.SYSTEM
        ]

        payload = {
            "model": target_model,
            "messages": convo_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        if system_content:
            payload["system"] = system_content

        async with httpx.AsyncClient(timeout=45.0) as client:
            async with client.stream("POST", self.BASE_URL, headers=headers, json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        if not data_str or data_str == "[DONE]":
                            continue
                        try:
                            event = json.loads(data_str)
                            if event.get("type") == "content_block_delta":
                                delta = event.get("delta", {}).get("text", "")
                                if delta:
                                    yield LLMStreamChunk(delta=delta)
                            elif event.get("type") == "message_stop":
                                yield LLMStreamChunk(delta="", finish_reason="stop")
                        except Exception:
                            pass

    async def generate_structured(
        self,
        messages: List[ChatMessage],
        response_model: Type[T],
        model: Optional[str] = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        headers = self._get_headers()
        target_model = model or self.default_model
        schema = response_model.model_json_schema()
        tool_name = schema.get("title", "structured_output")

        system_content = "\n\n".join(m.content for m in messages if m.role == Role.SYSTEM)
        convo_messages = [
            {"role": "user" if m.role == Role.USER else "assistant", "content": m.content}
            for m in messages
            if m.role != Role.SYSTEM
        ]

        payload = {
            "model": target_model,
            "messages": convo_messages,
            "max_tokens": 2048,
            "temperature": temperature,
            "tools": [
                {
                    "name": tool_name,
                    "description": "Extract structured output matching the schema",
                    "input_schema": schema,
                }
            ],
            "tool_choice": {"type": "tool", "name": tool_name},
        }
        if system_content:
            payload["system"] = system_content

        start_time = time.perf_counter()
        async with httpx.AsyncClient(timeout=45.0) as client:
            resp = await client.post(self.BASE_URL, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
        latency_ms = (time.perf_counter() - start_time) * 1000

        parsed_obj = None
        for block in data.get("content", []):
            if block.get("type") == "tool_use" and block.get("name") == tool_name:
                input_data = block.get("input", {})
                parsed_obj = response_model.model_validate(input_data)
                break

        usage_data = data.get("usage", {})
        prompt_tokens = usage_data.get("input_tokens", 0)
        comp_tokens = usage_data.get("output_tokens", 0)
        total_tokens = prompt_tokens + comp_tokens
        cost = calculate_cost(target_model, prompt_tokens, comp_tokens)

        return LLMResponse(
            content=json.dumps(parsed_obj.model_dump() if parsed_obj else {}),
            parsed=parsed_obj,
            model=target_model,
            provider=self.provider_type,
            latency_ms=round(latency_ms, 2),
            finish_reason=data.get("stop_reason"),
            usage=TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=comp_tokens,
                total_tokens=total_tokens,
                estimated_cost_usd=cost,
            ),
        )

    async def embed(self, texts: List[str], model: Optional[str] = None) -> List[List[float]]:
        raise NotImplementedError("Anthropic Claude does not provide direct vector embedding endpoints.")

    async def is_healthy(self) -> bool:
        return bool(self.api_key)
