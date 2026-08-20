# TsingXiaoDa Agent —— 清华大学辅修顾问智能体

基于 **Python (FastAPI)** 的辅修建议与支持智能体：帮助学生选辅修、解读培养方案、规划修读路径。
采用 **ReAct**（Reason + Act）推理框架，以 **CoT（思维链）** 优化推理过程，并提供 **流式输出**（SSE，OpenAI 兼容协议，可直接接入清小搭广场）。

## 目录结构

```
TsingXiaoDaAgent/
├── run.py                 # 启动入口（uvicorn）
├── requirements.txt       # fastapi / uvicorn / httpx / pydantic
├── .env.example           # 环境变量样例（复制为 .env 使用）
├── agent/                 # 智能体核心
│   ├── config.py          # 配置加载（env / .env）
│   ├── schemas.py         # Pydantic 数据模型（OpenAI 兼容消息）
│   ├── llm.py             # 上游大模型客户端（httpx 流式，OpenAI 兼容）
│   ├── knowledge.py       # 知识库：加载 minors/ 培养方案，目录/检索/详情
│   ├── tools.py           # ReAct 工具集（list_minors / get_minor_detail / search_minors / get_regulations）
│   ├── prompts.py         # 提示词工程：人设 + ReAct 格式 + 初始思维链（CoT）+ 少样本
│   └── react.py           # ReAct 循环（Thought→Action→Observation，流式事件）
├── app/
│   └── main.py            # FastAPI 服务层：/v1/models + /v1/chat/completions（SSE）
├── minors/                # 知识库：各院系辅修培养方案 Markdown
├── tests/
│   └── smoke_test.py      # 离线冒烟测试（MockTransport 模拟上游 LLM，无需真实 Key）
└── openai.md              # 清小搭广场接入协议参考
```

## 快速开始

```bash
# 1. 安装依赖
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. 配置上游模型（复制样例并按需修改，至少填 LLM_API_KEY）
cp .env.example .env

# 3. 启动
python run.py
# 或 uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 验证

```bash
# 连通性 / 凭证
curl -s http://localhost:8000/v1/models \
  -H "Authorization: Bearer sk-tsinghua-minor-dev"

# 流式对话（SSE）：思维链走 delta.reasoning，最终回答走 delta.content
curl -N http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer sk-tsinghua-minor-dev" \
  -H "Content-Type: application/json" \
  -d '{
    "stream": true,
    "messages": [{"role": "user", "content": "我是经管学院大二学生，想辅修计算机，要修多少学分？"}]
  }'

# 非流式
curl -s http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer sk-tsinghua-minor-dev" \
  -H "Content-Type: application/json" \
  -d '{"stream": false, "messages": [{"role": "user", "content": "有哪些文科类辅修？"}]}'
```

## 离线冒烟测试（无需真实 LLM Key）

`tests/smoke_test.py` 用 `httpx.MockTransport` 模拟上游模型（第一次调用让模型「思考并调用工具」，第二次返回 Final Answer），端到端验证 ReAct 循环、知识库检索与 SSE 帧序列：

```bash
./.venv/bin/python tests/smoke_test.py
```

## 架构与设计

### ReAct 循环（agent/react.py）

每轮对话按以下循环推进，最多 `MAX_REACT_STEPS` 轮：

```
Thought（思考）→ Action（工具调用）→ Observation（工具结果）→ 再次 Thought … → Final Answer
```

- 模型的每一步原始输出**即时流式推送**给前端（`delta.reasoning`），用户可看到思考与检索过程；
- 解析出 `Final Answer` 后停止循环，最终回答以 `delta.content` 逐段回放；
- 工具异常、未知工具均作为 Observation 回传，让模型自行修正；无标记输出时容错为最终回答；
- 多轮历史与 Observation 以交替的 assistant/user 消息喂给模型，保持对话上下文。

### CoT 初始思维链（agent/prompts.py）

`INITIAL_COT` 规定了五阶段固定推理路径，保证每次回答都"先想清楚再查、查完再下结论"：

1. **理解诉求** — 提取主修、年级、目标、精力约束；信息不足时在结尾给出澄清问题；
2. **规划检索** — 决定用哪个工具（全貌/细节/关键词/管理办法），避免盲目检索；
3. **检索核对** — 只从 Observation 提取事实：学分、课程结构、先修、接纳人数、主修限制、申请流程；
4. **交叉分析** — 先修课是否满足、学分负担、是否与主修学位重复授予、时间线是否可行；
5. **综合输出** — 推荐排序 + 基于事实的理由 + 关键提醒（申请节点/学分/先修/名额）+ 信息来源。

配合少样本示例（FEW_SHOT）演示一次完整的「思考→工具→观察→回答」轨迹，约束输出格式稳定。

### 工具集（agent/tools.py）

| 工具 | 用途 | 参数 |
| --- | --- | --- |
| `list_minors` | 按院系列出全部辅修专业 | 无 |
| `get_minor_detail` | 获取某专业完整培养方案 | `name` |
| `search_minors` | 关键词检索全部方案，返回摘要 | `query` |
| `get_regulations` | 获取辅修管理办法 | 无 |

新增工具只需在 `build_tools()` 中追加一个 `Tool`（名称 + 描述 + JSON Schema 参数 + 异步实现），提示词会自动带上工具清单。

### 流式输出协议（app/main.py）

`POST /v1/chat/completions`，`stream: true` 时返回 `text/event-stream`，帧序列严格遵循清小搭广场协议：

```
data: {"choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}   # role 帧
data: {"choices":[{"index":0,"delta":{"reasoning":"Thought: …"},"finish_reason":null}]}  # 思维链
data: {"choices":[{"index":0,"delta":{"content":"最终回答增量"},"finish_reason":null}]}   # 回答
data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{...}}          # stop 帧
data: [DONE]
```

要点：`stream` 严格按 JSON 布尔解析（字符串 `"false"` 不触发流式）；Bearer 鉴权（`API_KEY` 留空则关闭鉴权、任意请求放行；开发默认凭证下可免凭证本地调试；设置正式凭证则必须严格匹配）；上游失败时输出 error 帧后以 `[DONE]` 正常收尾。

> 对外暴露的模型名（`/v1/models` 与响应中的 `model` 字段）固定为本智能体自身的 `MODEL_NAME`（默认 `tsingxiaoda-minor`），**不暴露上游模型名**；上游模型仍由 `LLM_MODEL` 内部指定。

## 配置项（.env）

| 变量 | 说明 | 默认 |
| --- | --- | --- |
| `API_KEY` | 服务鉴权凭证；留空则关闭鉴权 | `sk-tsinghua-minor-dev` |
| `LLM_API_KEY` | 上游模型 Key | 空 |
| `LLM_BASE_URL` | 上游端点（到版本段为止） | `https://api.deepseek.com/v1` |
| `LLM_MODEL` | 上游模型名（内部使用） | `deepseek-v4-flash` |
| `MODEL_NAME` | 对外暴露的模型名 | `tsingxiaoda-minor` |
| `LLM_TEMPERATURE` | 采样温度 | `0.3` |
| `LLM_MAX_TOKENS` | 单步最大 token | `2048` |
| `LLM_INCLUDE_USAGE` | 是否请求 usage | `1` |
| `MAX_REACT_STEPS` | ReAct 最大轮数 | `6` |
| `TOOL_RESULT_MAX_CHARS` | 工具结果截断长度 | `6000` |
| `KNOWLEDGE_DIR` | 知识库目录 | `minors` |
| `HOST` / `PORT` | 监听地址 | `0.0.0.0:8000` |
