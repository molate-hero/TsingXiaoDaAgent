"""ReAct 智能体：Thought → Action → Observation 循环，全程流式产出事件。

事件协议（供服务层映射为 SSE）：
- {"type": "trace", "delta": str}   思维链增量（用 __Thought__ 等标记切段并硬编码剥离标记，
                                    仅含人类可读推理；工具结果以精简【检索结果】摘要出现）
- {"type": "answer", "delta": str}  最终回答增量（__Final_Answer__ 之后的文本，只出现一次）
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

# 唯一性标记（与 prompts.py 的 ReAct 格式保持一致）
TK_THOUGHT = "__Thought__"
TK_ACTION = "__Action__"
TK_ACTION_INPUT = "__Action_Input__"
TK_FINAL = "__Final_Answer__"
TK_OBSERVATION = "__Observation__"

# 新式标记的段落切分：__Xxx__ 后跟可选冒号（ASCII : 或全角 ：）与空白
_SEG_RE = re.compile(r"__(?P<kind>Thought|Action_Input|Action|Final_Answer)__(?:\s*[:：])?\s*")

# 流式路由：检测到 __Final_Answer__（或旧式 Final Answer:）后，其后的内容改路由为 answer
_FINAL_STREAM_RE = re.compile(r"__Final_Answer__(?:\s*[:：])?\s*|Final\s+Answer\s*[:：]?")
# 实时 answer 的最小分块长度，避免出现过多极小的 SSE 帧
ANSWER_CHUNK_SIZE = 12

# 旧式行首标签（仅用于兼容展示/解析，不参与流式路由）
_LEGACY_LABEL_RE = re.compile(
    r"(?m)^[ \t]*(?:Action\s+Input|Action|Thought|Final\s+Answer|Observation)\s*[:：]?\s*"
)


def _normalize_legacy_markers(text: str) -> str:
    """把旧式行首标签（Thought:/Action:/Action Input:/Final Answer:）统一转为新式标记，便于解析。"""
    t = text
    t = re.sub(r"(?m)^[ \t]*Action\s+Input\s*[:：]\s*", TK_ACTION_INPUT + ": ", t)
    t = re.sub(r"(?m)^[ \t]*Action\s*[:：]\s*", TK_ACTION + ": ", t)
    t = re.sub(r"(?m)^[ \t]*Final\s+Answer\s*[:：]\s*", TK_FINAL + ": ", t)
    t = re.sub(r"(?m)^[ \t]*Thought\s*[:：]\s*", TK_THOUGHT + ": ", t)
    return t


def parse_react_step(text: str) -> dict:
    """解析模型一轮输出，返回 {thought, action, action_input, final_answer}。

    按 __Thought__/__Action__/__Action_Input__/__Final_Answer__ 标记切段（兼容旧式标签）；
    无任何标记时全部字段为空。
    """
    text = _normalize_legacy_markers((text or "").strip())
    result = {"thought": None, "action": None, "action_input": None, "final_answer": None}
    if not text:
        return result

    matches = list(_SEG_RE.finditer(text))
    if not matches:
        return result
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[m.end():end].strip()
        if not content:
            continue
        kind = m.group("kind")
        if kind == "Thought":
            result["thought"] = content
        elif kind == "Action":
            result["action"] = content
        elif kind == "Action_Input":
            result["action_input"] = content
        elif kind == "Final_Answer":
            result["final_answer"] = content
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


def _clean_reasoning(text: str) -> str:
    """清洗推理展示文本：硬编码剥离 __Xxx__ 标记，只留人类可读内容。

    呈现规则（仅用于「显示」，喂回给模型的 scratch 仍保留原始带标记文本）：
    - __Thought__         直接去掉标记，保留其后推理内容；
    - __Action__          仅硬编码提取工具名，呈现为「\n\n调用工具<工具名>中...\n\n」（前后各一个空行）；
    - __Action_Input__    参数（JSON）不再向用户展示；
    - __Observation__     检索结果不再向用户展示（完整结果仍回填给模型）；
    - __Final_Answer__    不应出现在推理中，稳妥起见也去掉。
    """
    t = (text or "").strip()
    tool_m = re.search(TK_ACTION + r"\s*[:：]?\s*([^\n]+)", t)
    tool = tool_m.group(1).strip() if tool_m else None
    # 剥离 __Action_Input__ 参数段（不展示给用户）
    t = re.sub(TK_ACTION_INPUT + r"\s*[:：]?\s*[^\n]*", "", t)
    # 用「调用工具<工具名>中...」替代 __Action__ 段（前面留一个空行）
    t = re.sub(
        TK_ACTION + r"\s*[:：]?\s*[^\n]*",
        (lambda m: f"\n\n调用工具{tool}中..." if tool else ""),
        t,
    )
    # 去掉 __Thought__/__Observation__/__Final_Answer__ 标记（保留内容）
    for mk in (TK_THOUGHT, TK_OBSERVATION, TK_FINAL):
        t = re.sub(mk + r"\s*[:：]?\s*", "", t)
    # 旧式行首标签兜底
    t = _LEGACY_LABEL_RE.sub("", t)
    # 合并可能因替换产生的多余连续空行
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = t.strip()
    # 工具调用后保留一个空行，与下一步思考分隔
    if tool:
        t = t.rstrip("\n") + "\n\n"
    return t


class ReActAgent:
    def __init__(self, llm: LLMClient, knowledge: KnowledgeBase, config: Config):
        self.llm = llm
        self.knowledge = knowledge
        self.config = config
        self.tools: dict[str, Tool] = build_tools(knowledge)

    def system_prompt(self) -> str:
        return build_system_prompt(self.tools)

    async def run_stream(self, history: list[dict]) -> AsyncIterator[dict]:
        """流式运行 ReAct 循环（见模块 docstring 的事件协议）。

        采用「标记边界路由」：
        - 模型输出在检测到 `__Final_Answer__`（或旧式 Final Answer:）之前的 token → trace，
          并用 _clean_reasoning 硬编码剥离 __Thought__/__Action__/__Action_Input__/等标记；
        - 一旦检测到该边界，其后 token 即时改路由为 answer（最终回答），不再进入思考区，
          因此循环末尾也不再重放回答，正文只出现一次。
        - 工具结果以 __Observation__ 形式回填给模型，思维区仅展示精简的【检索结果】摘要。
        - 容错：若无 __Action__ 也无 __Final_Answer__，整段当作最终回答，且不再作为思考展示。
        """
        messages: list[dict] = [{"role": "system", "content": self.system_prompt()}]
        messages.extend(history)
        scratch: list[dict] = []  # 已发生的 Thought/Action → Observation 对
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        final_text: Optional[str] = None
        answer_streamed = False  # 回答是否已在流中作为 answer 发出（避免末尾重放）

        for _ in range(self.config.max_react_steps):
            buffer: list[str] = []      # 本步完整原始输出（用于解析 & 回填 scratch）
            reasoning: list[str] = []   # 边界前的推理原文（仅用于展示）
            in_answer = False
            answer_chunk = ""           # 实时缓存 answer，凑满 ANSWER_CHUNK_SIZE 再发
            answer_started = False      # 是否已出现首个非空白字符（用于裁剪回答开头的空白）

            try:
                async for delta in self.llm.chat_stream(
                    messages + scratch,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                ):
                    buffer.append(delta)
                    if not in_answer:
                        reasoning.append(delta)
                        joined = "".join(reasoning)
                        m = _FINAL_STREAM_RE.search(joined)
                        if m:
                            # 越过 __Final_Answer__ 边界：其前为推理，其后为回答正文
                            for piece in chunk_text(_clean_reasoning(joined[: m.start()])):
                                if piece:
                                    yield {"type": "trace", "delta": piece}
                            answer_chunk = joined[m.end():]
                            in_answer = True
                    else:
                        answer_chunk += delta
                    # 裁剪回答开头的空白/残留冒号（ASCII 或全角）：无论来自边界拆分还是后续 token
                    if in_answer and not answer_started:
                        answer_chunk = answer_chunk.lstrip(" \t\r\n\u3000：:")
                        if answer_chunk:
                            answer_started = True
                    if in_answer:
                        while len(answer_chunk) >= ANSWER_CHUNK_SIZE:
                            yield {"type": "answer", "delta": answer_chunk[:ANSWER_CHUNK_SIZE]}
                            answer_chunk = answer_chunk[ANSWER_CHUNK_SIZE:]
            finally:
                self._accumulate_usage(usage)

            # 兜底：剩余的 answer 缓存
            if answer_chunk:
                yield {"type": "answer", "delta": answer_chunk}

            text = "".join(buffer).strip()
            parsed = parse_react_step(text)

            if parsed["final_answer"]:
                final_text = parsed["final_answer"]
                answer_streamed = in_answer
                break

            if parsed["action"]:
                # 动作步骤：把本步推理清洗后作为思考展示（工具调用呈现为「调用工具<工具名>中...」）
                for piece in chunk_text(_clean_reasoning("".join(reasoning))):
                    if piece:
                        yield {"type": "trace", "delta": piece}
                scratch.append({"role": "assistant", "content": text})
                observation = await self._execute_tool(parsed["action"], parsed["action_input"])
                # 完整检索结果仅回填给模型（用于推理）；不在此处向用户展示
                scratch.append({"role": "user", "content": f"{TK_OBSERVATION}: {observation}"})
                continue

            # 容错：既无标记（无 __Action__ 也无 __Final_Answer__）时，把整段当作最终回答处理。
            # 注意：此时不应再把本步作为「思考」展示，否则会与最终回答重复。
            if text:
                final_text = _clean_reasoning(text)
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

        if not answer_streamed:
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
