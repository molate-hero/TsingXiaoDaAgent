"""ReAct 智能体：Thought → Action → Observation 循环，全程流式产出事件。

事件协议（供服务层映射为 SSE）：
- {"type": "trace", "delta": str}   思维链增量（Thought / Action / Action Input / Observation 摘要）
- {"type": "answer", "delta": str}  最终回答增量（Final Answer 之后的文本）
- {"type": "done",  "usage": dict}  循环结束（含累计 token 用量）
"""
from __future__ import annotations

import json
import re
from typing import AsyncIterator, Optional

from .config import Config
from .knowledge import KnowledgeBase
from .llm import LLMClient
from .prompts import build_system_prompt
from .tools import Tool, build_tools

# 容错解析：标记允许与上一段文本同行（上游 flash 模型偶发不换行），前缀用 [^\n]*? 容忍；
# 工具名只取英文标识符，避免误吞 "Action: get_minor_detail Action Input: {...}" 同行场景；
# \b 词边界 + Action 后必须直接跟冒号（而非 Input），避免误匹配 Action Input 行
_FINAL_RE = re.compile(r"(?m)^[^\n]*?\bFinal\s+Answer\s*:\s*(.+)", re.S)
_ACTION_RE = re.compile(r"(?m)^[^\n]*?\bAction\s*:\s*([A-Za-z_][A-Za-z0-9_]*)")
_ACTION_INPUT_RE = re.compile(r"(?m)^[^\n]*?\bAction\s+Input\s*:\s*(.+)", re.S)
_THOUGHT_RE = re.compile(r"(?m)^\s*Thought\s*:\s*(.*?)(?=\n\s*(?:Action|Final\s+Answer)\s*:|\Z)", re.S)

OBSERVATION_PREVIEW_CHARS = 240


def parse_react_step(text: str) -> dict:
    """解析模型一轮输出，返回 {thought, action, action_input, final_answer}。

    优先级：Final Answer > Action(+Action Input)；无任何标记时全部字段为空。
    """
    text = (text or "").strip()
    result: dict[str, str | None] = {"thought": None, "action": None, "action_input": None, "final_answer": None}
    if not text:
        return result

    final_m = _FINAL_RE.search(text)
    pre = text
    if final_m:
        result["final_answer"] = final_m.group(1).strip()
        pre = text[: final_m.start()]

    action_m = _ACTION_RE.search(pre)
    if action_m:
        result["action"] = action_m.group(1).strip()
        input_m = _ACTION_INPUT_RE.search(pre)
        result["action_input"] = input_m.group(1).strip() if input_m else None

    thought_m = _THOUGHT_RE.search(pre)
    result["thought"] = (thought_m.group(1).strip() if thought_m else pre.strip()) or None
    return result


def chunk_text(text: str, size: int = 12) -> list[str]:
    """把最终回答切成小片段，用于本地流式回放（保留换行）。"""
    pieces: list[str] = []
    buf = ""
    for ch in text:
        buf += ch
        if len(buf) >= size or ch in "\n":
            pieces.append(buf)
            buf = ""
    if buf:
        pieces.append(buf)
    return pieces


class ReActAgent:
    def __init__(self, llm: LLMClient, knowledge: KnowledgeBase, config: Config):
        self.llm = llm
        self.knowledge = knowledge
        self.config = config
        self.tools: dict[str, Tool] = build_tools(knowledge)

    def system_prompt(self) -> str:
        return build_system_prompt(self.tools)

    async def run_stream(self, history: list[dict]) -> AsyncIterator[dict]:
        """流式运行 ReAct 循环（见模块 docstring 的事件协议）。"""
        messages: list[dict] = [{"role": "system", "content": self.system_prompt()}]
        messages.extend(history)
        scratch: list[dict] = []  # 已发生的 Thought/Action → Observation 对
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        final_text: Optional[str] = None

        for _ in range(self.config.max_react_steps):
            buffer: list[str] = []
            try:
                async for delta in self.llm.chat_stream(
                    messages + scratch,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                ):
                    buffer.append(delta)
                    yield {"type": "trace", "delta": delta}
            finally:
                self._accumulate_usage(usage)

            text = "".join(buffer).strip()
            parsed = parse_react_step(text)

            if parsed["final_answer"]:
                final_text = parsed["final_answer"]
                break

            if parsed["action"]:
                scratch.append({"role": "assistant", "content": text})
                observation = await self._execute_tool(parsed["action"], parsed["action_input"])
                preview = (
                    observation[:OBSERVATION_PREVIEW_CHARS] + "……"
                    if len(observation) > OBSERVATION_PREVIEW_CHARS
                    else observation
                )
                yield {"type": "trace", "delta": f"\n\nObservation: {preview}\n\n"}
                scratch.append({"role": "user", "content": f"Observation: {observation}"})
                continue

            # 容错：既无 Final Answer 也无 Action 时，把整段输出当作最终回答
            if text:
                final_text = text
            break

        if not final_text and scratch:
            # 收尾轮：步数耗尽但已检索到信息时，要求模型基于现有 Observation 直接作答，
            # 避免开放多步任务（如「推荐适合我的辅修」）直接落入「抱歉」兜底
            yield {"type": "trace", "delta": "\n\n（检索步数已耗尽，正在基于已检索信息整理最终回答…）\n\n"}
            buffer: list[str] = []
            try:
                async for delta in self.llm.chat_stream(
                    messages
                    + scratch
                    + [
                        {
                            "role": "user",
                            "content": (
                                "检索步数预算已耗尽。请立即停止检索，"
                                "基于以上已经获得的 Observation 信息直接输出 Final Answer，不要再调用任何工具。"
                            ),
                        }
                    ],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                ):
                    buffer.append(delta)
                    yield {"type": "trace", "delta": delta}
            finally:
                self._accumulate_usage(usage)
            text = "".join(buffer).strip()
            parsed = parse_react_step(text)
            if parsed["final_answer"]:
                final_text = parsed["final_answer"]
            elif not parsed["action"] and text:
                final_text = text

        if not final_text:
            final_text = "抱歉，我在限定步数内未能完成检索与推理，请稍后再试，或换一种问法。"

        for piece in chunk_text(final_text):
            yield {"type": "answer", "delta": piece}
        yield {"type": "done", "usage": usage}

    async def run(self, history: list[dict]) -> tuple[str, dict]:
        """非流式便捷方法：返回 (最终回答, usage)。"""
        answer: list[str] = []
        usage: dict = {}
        async for ev in self.run_stream(history):
            if ev["type"] == "answer":
                answer.append(ev["delta"])
            elif ev["type"] == "done":
                usage = ev["usage"]
        return "".join(answer), usage

    def _accumulate_usage(self, usage: dict) -> None:
        last = self.llm.last_usage or {}
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            if last.get(key):
                usage[key] = usage.get(key, 0) + last[key]

    async def _execute_tool(self, name: str, raw_input: Optional[str]) -> str:
        """执行工具并返回 Observation 文本；工具异常也作为 Observation 回传，让模型继续。"""
        tool = self.tools.get((name or "").strip())
        if tool is None:
            return f"错误：未知工具「{name}」。可用工具：{', '.join(self.tools)}"
        try:
            kwargs = self._parse_action_input(tool, raw_input)
            return await tool.func(**kwargs)
        except Exception as exc:  # noqa: BLE001 工具自身异常 → 回传 Observation
            return f"错误：工具执行失败（{type(exc).__name__}: {exc}）"

    @staticmethod
    def _parse_action_input(tool: Tool, raw_input: Optional[str]) -> dict:
        """解析 Action Input：优先 JSON；失败时若工具只有一个必填参数，把原文绑定到该参数。"""
        raw = (raw_input or "").strip()
        if raw:
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass
        required = (tool.parameters.get("required") or []) if tool.parameters else []
        if len(required) == 1:
            return {required[0]: raw}
        return {}
