"""上游大模型客户端：OpenAI 兼容 /chat/completions，支持流式 SSE 解析。

仅依赖 httpx（与 requirements.txt 保持一致），不引入 openai SDK。
每次 chat_stream 调用结束后，self.last_usage 记录该次请求的 usage（上游支持时）。
"""
from __future__ import annotations

import json
from typing import AsyncIterator, Optional

import httpx

from .config import Config


class LLMUpstreamError(RuntimeError):
    """上游大模型调用失败（网络错误 / 非 200 / 未配置 Key）。"""


class LLMClient:
    def __init__(
        self,
        config: Config,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ):
        """transport 用于测试注入 MockTransport，生产环境保持 None。"""
        self._cfg = config
        self._transport = transport
        self.last_usage: Optional[dict] = None

    def _endpoint(self) -> str:
        base = self._cfg.llm_base_url.rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        return base + "/chat/completions"

    def _payload(self, messages, temperature, max_tokens) -> dict:
        payload = {
            "model": self._cfg.llm_model,
            "messages": messages,
            "stream": True,
        }
        if self._cfg.llm_include_usage:
            # 在最后一个 chunk 中携带 usage（部分上游忽略）
            payload["stream_options"] = {"include_usage": True}
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        return payload

    async def chat_stream(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[str]:
        """流式调用上游，逐段产出 choices[0].delta.content 文本增量。"""
        self.last_usage = None
        # 注入 transport（测试）时视为离线调用，不强制要求真实 Key
        if not self._cfg.llm_api_key and self._transport is None:
            raise LLMUpstreamError(
                "未配置 LLM_API_KEY：请复制 .env.example 为 .env 并填入上游模型凭证"
            )
        timeout = httpx.Timeout(self._cfg.llm_timeout)
        headers = {
            "Authorization": f"Bearer {self._cfg.llm_api_key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(transport=self._transport, timeout=timeout) as client:
            async with client.stream(
                "POST",
                self._endpoint(),
                json=self._payload(messages, temperature, max_tokens),
                headers=headers,
            ) as resp:
                if resp.status_code != 200:
                    body = (await resp.aread()).decode("utf-8", "replace")[:300]
                    raise LLMUpstreamError(
                        f"上游返回 HTTP {resp.status_code}: {body}"
                    )
                async for line in resp.aiter_lines():
                    line = line.strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if obj.get("usage"):
                        self.last_usage = obj["usage"]
                    choices = obj.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    content = delta.get("content")
                    if content:
                        yield content

    async def chat(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """非流式便捷方法：收集完整文本后返回。"""
        chunks: list[str] = []
        async for c in self.chat_stream(messages, temperature=temperature, max_tokens=max_tokens):
            chunks.append(c)
        return "".join(chunks)
