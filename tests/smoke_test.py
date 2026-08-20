"""离线冒烟测试：用 httpx.MockTransport 模拟上游 LLM，端到端验证 ReAct 循环、知识库与 SSE 输出。

运行：./.venv/bin/python tests/smoke_test.py
无需真实 LLM API Key，也不访问网络。
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.config import get_config  # noqa: E402
from agent.knowledge import KnowledgeBase  # noqa: E402
from agent.llm import LLMClient  # noqa: E402
from agent.react import ReActAgent, chunk_text, parse_react_step  # noqa: E402

USAGE_MOCK = {"prompt_tokens": 120, "completion_tokens": 80, "total_tokens": 200}


def make_sse(text: str, usage: dict | None = None) -> str:
    """把一段模型输出编码为 OpenAI 兼容 SSE 文本（role 帧 + content 帧 + stop 帧 + [DONE]）。"""
    lines = [
        'data: {"id":"c","object":"chat.completion.chunk","created":1,'
        '"choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}'
    ]
    for i in range(0, len(text), 5):
        piece = text[i : i + 5]
        lines.append(
            "data: "
            + json.dumps(
                {
                    "id": "c",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
                },
                ensure_ascii=False,
            )
        )
    lines.append(
        "data: "
        + json.dumps(
            {
                "id": "c",
                "object": "chat.completion.chunk",
                "created": 1,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": usage or USAGE_MOCK,
            }
        )
    )
    lines.append("data: [DONE]")
    return "\n\n".join(lines) + "\n\n"


def mock_llm_handler(request: httpx.Request) -> httpx.Response:
    """模拟上游模型：第一次调用让模型"思考并调用工具"，第二次收到 Observation 后给出 Final Answer。"""
    body = json.loads(request.content)
    obs_count = sum(
        1
        for m in body["messages"]
        if m["role"] == "user" and str(m.get("content", "")).startswith("Observation")
    )
    if obs_count == 0:
        text = (
            "Thought: 学生询问计算机辅修的学分要求，需要先获取培养方案。\n"
            "Action: get_minor_detail\n"
            'Action Input: {"name": "计算机科学与技术"}'
        )
    else:
        text = (
            "Thought: 培养方案显示至少修完36学分（12门课程），必修28学分。\n"
            "Final Answer: 计算机辅修需修满至少 **36 学分**（12 门课程），"
            "其中必修 28 学分、限选不少于 2 学分、选修不少于 6 学分。"
        )
    return httpx.Response(200, headers={"Content-Type": "text/event-stream"}, text=make_sse(text))


def _build_agent() -> ReActAgent:
    cfg = get_config()
    kb = KnowledgeBase(cfg.knowledge_dir).load()
    llm = LLMClient(cfg, transport=httpx.MockTransport(mock_llm_handler))
    return ReActAgent(llm, kb, cfg)


def test_parse_react_step() -> None:
    p = parse_react_step(
        "Thought: 需要检索。\nAction: get_minor_detail\nAction Input: {\"name\":\"计算机科学与技术\"}"
    )
    assert p["action"] == "get_minor_detail"
    assert p["action_input"] == '{"name":"计算机科学与技术"}'
    assert p["final_answer"] is None

    p2 = parse_react_step("Thought: 完毕。\nFinal Answer: 答案是 42。")
    assert p2["final_answer"] == "答案是 42。"
    assert p2["action"] is None

    p3 = parse_react_step("  ")  # 空输出
    assert p3["final_answer"] is None and p3["action"] is None
    print("[OK] parse_react_step")


def test_knowledge() -> None:
    cfg = get_config()
    kb = KnowledgeBase(cfg.knowledge_dir).load()
    assert len(kb.docs) >= 45, f"知识库文档数异常: {len(kb.docs)}"

    doc = kb.get_minor("计算机科学与技术")
    assert doc is not None and "36学分" in doc.content.replace(" ", "")

    hits = kb.search("先修 微积分")
    assert hits, "关键词检索应命中"

    reg = kb.get_regulations()
    assert reg is not None and "管理办法" in reg.key

    groups = kb.list_minors()
    assert groups and all(g["minors"] for g in groups)
    print(f"[OK] knowledge：{len(kb.docs)} 份文档 / {len(groups)} 个院系")


async def test_react_loop() -> None:
    agent = _build_agent()
    history = [{"role": "user", "content": "计算机辅修要修多少学分？"}]
    events = [ev async for ev in agent.run_stream(history)]

    trace = "".join(ev["delta"] for ev in events if ev["type"] == "trace")
    answer = "".join(ev["delta"] for ev in events if ev["type"] == "answer")
    done = next(ev for ev in events if ev["type"] == "done")

    assert "get_minor_detail" in trace, "思维链应包含工具调用"
    assert "Observation" in trace, "思维链应包含工具观察结果"
    assert "36 学分" in answer, "最终回答应包含检索到的事实"
    assert done["usage"]["completion_tokens"] > 0, "usage 应被累计"
    assert answer == "".join(chunk_text(answer)), "chunk_text 应无损拼接"
    print(f"[OK] react loop：trace={len(trace)}字 / answer={len(answer)}字 / usage={done['usage']}")


def test_sse_endpoint() -> None:
    from fastapi.testclient import TestClient

    from app.main import app

    def install_mock_agent() -> str:
        """lifespan 每次进入 TestClient 都会重建 agent，需在每个块内重新注入 mock 上游。"""
        cfg = get_config()
        llm = LLMClient(cfg, transport=httpx.MockTransport(mock_llm_handler))
        app.state.agent = ReActAgent(llm, app.state.knowledge, cfg)
        return cfg.api_key

    with TestClient(app) as client:
        api_key = install_mock_agent()

        # 流式
        r = client.post(
            "/v1/chat/completions",
            json={"stream": True, "messages": [{"role": "user", "content": "计算机辅修要修多少学分？"}]},
            headers={"Authorization": f"Bearer {api_key}"},
        )
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        body = r.text
        assert "data: [DONE]" in body
        assert '"reasoning"' in body, "SSE 应包含思维链字段"
        assert "36 学分" in body, "SSE 应包含最终回答"
        assert body.count('"role": "assistant"') >= 1, "应有 role 帧"
        assert f'"model": "{app.state.config.model_name}"' in body, "SSE 帧应携带本智能体模型名"
        assert app.state.config.llm_model not in body, "SSE 帧不得暴露上游模型名"
        print(f"[OK] sse endpoint：{body.count('data: {')} 个 data 帧")

    with TestClient(app) as client:
        api_key = install_mock_agent()
        # 非流式
        r = client.post(
            "/v1/chat/completions",
            json={"stream": False, "messages": [{"role": "user", "content": "计算机辅修要修多少学分？"}]},
            headers={"Authorization": f"Bearer {api_key}"},
        )
        data = r.json()
        assert r.status_code == 200
        assert data["choices"][0]["message"]["content"].startswith("计算机辅修")
        assert data["model"] == app.state.config.model_name, "非流式响应应携带本智能体模型名"
        assert data["model"] != app.state.config.llm_model, "非流式响应不得暴露上游模型名"
        print("[OK] non-stream endpoint")

    with TestClient(app) as client:
        # 鉴权失败
        r = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": "Bearer wrong-key"},
        )
        assert r.status_code == 401
        print("[OK] auth 401")

    with TestClient(app) as client:
        api_key = install_mock_agent()
        # 严格布尔解析：字符串 "false" 不应触发流式
        r = client.post(
            "/v1/chat/completions",
            json={"stream": "false", "messages": [{"role": "user", "content": "计算机辅修要修多少学分？"}]},
            headers={"Authorization": f"Bearer {api_key}"},
        )
        assert r.headers["content-type"].startswith("application/json"), "字符串 false 应按非流式处理"
        print("[OK] strict bool stream")


def test_models_endpoint() -> None:
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as client:
        r = client.get("/v1/models", headers={"Authorization": f"Bearer {app.state.config.api_key}"})
        assert r.status_code == 200
        data = r.json()["data"][0]
        assert data["id"] == app.state.config.model_name, "对外应暴露本智能体模型名"
        assert data["id"] != app.state.config.llm_model, "不得暴露上游模型名"
        print(f"[OK] /v1/models（对外 {data['id']}，上游 {app.state.config.llm_model} 已隐藏）")


def test_no_auth_when_api_key_empty() -> None:
    """API_KEY 为空时无需鉴权：无凭证 / 任意凭证均应放行。"""
    import os
    from dataclasses import replace

    from fastapi.testclient import TestClient

    from app.main import _check_auth, app

    # 单元级：_check_auth 对空 API_KEY 配置不做任何拒绝
    class _FakeRequest:
        def __init__(self, headers: dict):
            self.headers = headers

    cfg_empty = replace(get_config(), api_key="")
    for hdrs in ({}, {"authorization": "Bearer whatever"}, {"authorization": "Bearer wrong-key"}):
        _check_auth(cfg_empty, _FakeRequest(hdrs))  # 不应抛 HTTPException(401)

    # 端到端：环境变量 API_KEY="" 时 /v1/models 无凭证、错误凭证均 200
    old = os.environ.get("API_KEY")
    os.environ["API_KEY"] = ""
    try:
        with TestClient(app) as client:
            assert client.get("/v1/models").status_code == 200, "无凭证应放行"
            r = client.get("/v1/models", headers={"Authorization": "Bearer whatever"})
            assert r.status_code == 200, "任意凭证应放行"
    finally:
        if old is None:
            os.environ.pop("API_KEY", None)
        else:
            os.environ["API_KEY"] = old
    print("[OK] API_KEY 为空时无需鉴权（单元 + 端到端）")


if __name__ == "__main__":
    test_parse_react_step()
    test_knowledge()
    asyncio.run(test_react_loop())
    test_sse_endpoint()
    test_models_endpoint()
    test_no_auth_when_api_key_empty()
    print("\n全部冒烟测试通过 ✓")
