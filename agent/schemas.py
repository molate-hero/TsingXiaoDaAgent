"""Pydantic 数据模型：OpenAI 兼容的对话请求 / 消息结构。"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """协议要求支持 role ∈ {system, user, assistant}，content 为字符串。"""

    role: str = Field(description="system / user / assistant")
    content: Any = Field(description="字符串内容（协议要求）；兼容多模态 list 输入")

    def to_dict(self) -> dict:
        content = self.content
        if isinstance(content, list):  # 多模态输入兼容：只取文本片段
            parts = [
                p.get("text", "")
                for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            ]
            content = "".join(parts) or str(content)
        else:
            content = str(content)
        return {"role": self.role, "content": content}


class ChatCompletionRequest(BaseModel):
    """OpenAI 兼容对话请求体。额外字段一律容忍（extra=allow）。"""

    model: Optional[str] = None
    messages: list[ChatMessage]
    stream: bool = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    sessionId: Optional[str] = None

    model_config = {"extra": "allow"}
