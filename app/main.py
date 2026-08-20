"""FastAPI 服务层：OpenAI 兼容 /v1 端点（清小搭广场标准协议接入）。

- GET  /v1/models            连通性与凭证校验
- POST /v1/chat/completions  对话；stream=true 时返回 SSE（text/event-stream）

SSE 帧序列（协议要求）：
    1. role 帧    delta: {"role": "assistant"}
    2. reasoning 帧 delta: {"reasoning": ...}   ← ReAct/CoT 思维链增量
    3. content 帧 delta: {"content": ...}       ← 最终回答增量
    4. stop 帧    delta: {} + finish_reason:"stop" + usage
    5. data: [DONE]
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from agent.config import Config, get_config
from agent.knowledge import KnowledgeBase
from agent.llm import LLMClient, LLMUpstreamError
from agent.react import ReActAgent
from agent.schemas import ChatMessage

logger = logging.getLogger("tsingxiaoda")

# 让 INFO 级运行日志（知识库加载、请求链路）在容器/终端可见
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = get_config()
    kb = KnowledgeBase(cfg.knowledge_dir).load()
    llm = LLMClient(cfg)
    agent = ReActAgent(llm, kb, cfg)
    app.state.config = cfg
    app.state.knowledge = kb
    app.state.agent = agent
    logger.info("知识库已加载：%d 份文档（%s）", len(kb.docs), cfg.knowledge_dir)
    yield


app = FastAPI(
    title="TsingXiaoDa Agent（清华大学辅修顾问）",
    version="0.1.0",
    lifespan=lifespan,
)


def _check_auth(cfg: Config, request: Request) -> None:
    # API_KEY 为空 → 关闭鉴权，任意请求放行（便于本地/内网部署）
    if not (cfg.api_key or "").strip():
        return
    header = request.headers.get("authorization", "")
    token = header[len("Bearer "):].strip() if header.lower().startswith("bearer ") else ""
    if token != cfg.api_key:
        # 开发默认凭证下允许免凭证本地调试；正式凭证必须严格匹配
        if not (cfg.is_dev_default_auth and token == ""):
            raise HTTPException(status_code=401, detail="invalid bearer credential")


def _normalize_history(messages: list) -> list[dict]:
    """校验并规范化消息列表（协议要求 role ∈ {system,user,assistant}，content 为字符串）。"""
    history: list[dict] = []
    for m in messages:
        if not isinstance(m, dict) or "role" not in m or "content" not in m:
            raise HTTPException(status_code=400, detail="invalid message: need role and content")
        role = m["role"]
        if role not in ("system", "user", "assistant"):
            raise HTTPException(status_code=400, detail=f"unsupported role: {role}")
        history.append(ChatMessage(role=role, content=m["content"]).to_dict())
    return history


@app.get("/v1/models")
async def list_models(request: Request):
    _check_auth(app.state.config, request)
    # 对外暴露本智能体自身的模型名，而非上游模型名
    model = app.state.config.model_name or "default"
    return {
        "object": "list",
        "data": [{"id": model, "object": "model", "owned_by": "tsingxiaoda"}],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    _check_auth(app.state.config, request)
    try:
        raw = await request.json()
    except Exception:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="invalid JSON body")

    # 严格按 JSON 布尔解析 stream（协议要求：不要把字符串 "false" 当真）
    stream_flag = raw.get("stream", False)
    if not isinstance(stream_flag, bool):
        stream_flag = False

    messages = raw.get("messages")
    if not isinstance(messages, list) or not messages:
        raise HTTPException(status_code=400, detail="messages is required and must be a non-empty list")
    history = _normalize_history(messages)

    agent: ReActAgent = app.state.agent

    if stream_flag:
        return StreamingResponse(
            _sse_events(agent, history),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # 非流式：OpenAI 兼容 JSON
    try:
        content, usage = await agent.run(history)
    except LLMUpstreamError as exc:
        raise HTTPException(status_code=502, detail=f"上游模型调用失败: {exc}")
    created = int(time.time())
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": created,
        "model": app.state.config.model_name,
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
        ],
        "usage": usage,
    }


def _sse_frame(payload: dict) -> str:
    return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"


async def _sse_events(agent: ReActAgent, history: list[dict]):
    """把 ReAct 事件流映射为协议规定的 OpenAI 兼容 SSE 帧序列。"""
    cfg = agent.config
    cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    def chunk(delta: dict, finish_reason=None, usage=None, error=None) -> str:
        body = {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": cfg.model_name,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        if usage is not None:
            body["usage"] = usage
        if error is not None:
            body["error"] = error
        return _sse_frame(body)

    emitted = False
    try:
        yield chunk({"role": "assistant"})  # 1. role 帧
        emitted = True
        async for ev in agent.run_stream(history):
            if ev["type"] == "trace":
                yield chunk({"reasoning": ev["delta"]})  # 2. 思维链增量
            elif ev["type"] == "answer":
                yield chunk({"content": ev["delta"]})  # 3. 最终回答增量
            elif ev["type"] == "done":
                yield chunk({}, finish_reason="stop", usage=ev.get("usage"))  # 4. stop 帧
        yield "data: [DONE]\n\n"  # 5. 终止哨兵
    except LLMUpstreamError as exc:
        logger.exception("上游模型调用失败")
        yield chunk({}, finish_reason="stop", error={"type": "upstream_error", "message": str(exc)[:300]})
        yield "data: [DONE]\n\n"
    except Exception as exc:  # noqa: BLE001
        logger.exception("agent 运行异常")
        if not emitted:  # 未产出任何内容就失败 → 让 FastAPI 返回 5xx（协议要求）
            raise
        yield chunk({}, finish_reason="stop", error={"type": "agent_error", "message": str(exc)[:300]})
        yield "data: [DONE]\n\n"


if __name__ == "__main__":
    import uvicorn

    cfg = get_config()
    uvicorn.run("app.main:app", host=cfg.host, port=cfg.port, reload=False)
