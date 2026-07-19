import time
from typing import Any

import httpx


def chat_completion(
    api_key: str,
    base_url: str,
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.3,
    max_tokens: int = 4096,
    timeout: float = 90,
    retries: int = 1,
) -> str:
    """Call an OpenAI-compatible chat endpoint with one transient-failure retry."""
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload: dict[str, Any] = {
        "model": "deepseek-chat",
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    for attempt in range(retries + 1):
        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.post(f"{base_url.rstrip('/')}/chat/completions", headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            if not isinstance(content, str) or not content.strip():
                raise RuntimeError("上游模型返回了空回答")
            return content
        except (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError, KeyError, IndexError, TypeError, ValueError):
            if attempt == retries:
                raise
            time.sleep(0.5 * (attempt + 1))

    raise RuntimeError("模型调用失败")
