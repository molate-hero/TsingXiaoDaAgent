# Tsinghua Minor Advisor — 清华大学辅修专业规划助手

基于 DeepSeek API 构建的智能 Agent，为清华本科生提供辅修专业咨询与修读规划服务。兼容 OpenAI `/v1/chat/completions` 格式，支持流式输出。

---

## 快速开始

### 前置要求

- Python 3.10+
- DeepSeek API Key（或兼容 OpenAI 格式的其他 API）

### 本地运行

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 API Key（二选一）
export DEEPSEEK_API_KEY=sk-xxx          # 环境变量
# 或将 Key 写入 .config 文件（自动读取）

# 3. 启动 CLI 交互模式
python run.py

# 或启动 API 服务
python run.py --mode api --port 8000
```

### Docker 部署

```bash
# 构建镜像
docker build -t tsinghua-minor-advisor .

# 运行
docker run -d --name minor-advisor \
  -p 8000:8000 \
  -e DEEPSEEK_API_KEY=sk-xxx \
  -v ./data:/app/data \
  tsinghua-minor-advisor
```

或使用 docker compose：

```bash
export DEEPSEEK_API_KEY=sk-xxx
docker compose up -d
```

---

## API 文档

### OpenAI 兼容接口

**对话：** `POST /v1/chat/completions`

```json
{
  "model": "tsinghua-minor-advisor",
  "messages": [
    {"role": "user", "content": "我是计算机系大一学生，想辅修经济学"}
  ],
  "stream": true,
  "user": "会话ID（可选）"
}
```

可用任何 OpenAI SDK 调用：

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
resp = client.chat.completions.create(
    model="tsinghua-minor-advisor",
    messages=[{"role": "user", "content": "..."}],
    stream=True
)
for chunk in resp:
    print(chunk.choices[0].delta.content or "", end="")
```

**模型列表：** `GET /v1/models`

### 辅修数据接口

| 端点 | 说明 |
|------|------|
| `GET /minors` | 获取所有辅修专业列表 |
| `GET /minors/{名称}` | 获取某辅修详细信息 |

---

## 项目架构

```
TsingXiaoDaAgent/
├── agent/                  # Agent 核心
│   ├── core.py             # 会话管理、LLM 调用、ReAct 工具调度
│   ├── data_loader.py      # 长期记忆：解析辅养方案 → 结构化数据
│   ├── memory.py           # 短期记忆（对话历史）& 长期记忆（辅修数据库）
│   ├── tools.py            # 工具集：搜索、详情、资格检查
│   ├── course_catalog.py   # 已整理课程资料的本地检索目录
│   ├── llm_client.py       # 统一的模型调用、超时与瞬时失败重试
│   ├── prompts.py          # 系统提示词模板
│   └── planner.py          # 修读计划生成
├── api/
│   └── main.py             # FastAPI 服务（OpenAI 兼容格式）
├── data/                   # 解析缓存（自动生成）
├── Dockerfile & docker-compose.yml
├── run.py                  # 统一入口
└── 本科辅修培养方案2025版.md
```

### 设计要点

| 组件 | 说明 |
|------|------|
| **推理机制** | ReAct 模式：LLM 输出 `ACTION` 触发工具调用，结果回填后二次推理 |
| **短期记忆** | 每个会话独立的对话历史（最近 20 轮），以 `user` 字段区分 |
| **长期记忆** | 44 个辅修专业培养方案，以及 1000+ 门已整理的课程资料 |
| **规划能力** | LLM 自主推理 + 专用 Planner 双通道，考虑先修关系、开课学期、学分均衡 |
| **工具集** | 辅修搜索、资格检查、课程搜索、课程详情和课程列表 |

### 课程资料检索

`data/curated_courses.json` 中保存了已匹配到辅修培养方案的课程资料。Agent 会在涉及具体课程的提问中优先查询这些本地资料，能够回答课程内容、学分、先修要求、考核方式、成绩构成和教材等问题。

内部工具包括：

- `search_courses`：按课程号、课程名称、院系或课程主题搜索。
- `get_course_detail`：查询单门课程的详细资料。
- `list_minor_courses`：列出某辅修培养方案中已收录详情的课程。

---

## 使用示例

### CLI 模式

```
$ python run.py

清华大学辅修专业规划助手 v1.0
输入 'quit' 退出 | 'clear' 清空对话 | 'plan' 生成修读计划

你 > 我是计算机系大二学生，想辅修经济学
助手 > [根据你的情况推荐经济学辅修，并给出课程安排...]

你 > plan 机械工程 大二 计算机科学与技术
助手 > [生成按学期的详细修读计划...]
```

### API 模式

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tsinghua-minor-advisor",
    "messages": [{"role": "user", "content": "经济学辅修有哪些必修课？"}],
    "stream": true
  }'
```

---

## 配环境

依赖：`fastapi`、`uvicorn`、`httpx`、`pydantic`

```bash
pip install fastapi uvicorn httpx pydantic
```

---

## License

MIT
