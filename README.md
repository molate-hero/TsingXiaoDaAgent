<div align="center">
  <img src="icon.svg" width="128" height="128" alt="Tsinghua Minor Advisor Icon">
  <h1 align="center">Tsinghua Minor Advisor</h1>
  <p align="center"><strong>清华大学辅修专业规划助手</strong></p>
  <p align="center">基于 DeepSeek API + 词嵌入语义搜索的智能 Agent<br>
  为清华本科生提供辅修专业咨询与修读规划服务</p>
  <p align="center">
    <img src="https://img.shields.io/badge/python-3.10+-blue" alt="Python">
    <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
    <img src="https://img.shields.io/badge/API-OpenAI%20Compatible-orange" alt="API">
  </p>
</div>

---

## 目录

- [快速开始](#快速开始)
- [项目架构](#项目架构)
- [设计要点](#设计要点)
- [使用示例](#使用示例)
- [API 文档](#api-文档)
- [配置说明](#配置说明)
- [License](#license)

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

## 项目架构

```
TsingXiaoDaAgent/
├── agent/                      # Agent 核心
│   ├── core.py                 # 会话管理、LLM 调用、ReAct 工具调度
│   ├── data_loader.py          # 长期记忆：解析辅养方案 → 结构化数据
│   ├── embedding.py            # 词嵌入引擎：语义搜索（sentence-transformers）
│   ├── course_catalog.py       # 已整理课程资料的本地检索目录
│   ├── llm_client.py           # 统一的模型调用、超时与瞬时失败重试
│   ├── memory.py               # 短期记忆（对话历史）& 长期记忆（数据库）
│   ├── tools.py                # 工具集：搜索、详情、资格检查
│   ├── prompts.py              # 系统提示词模板
│   └── planner.py              # 修读计划生成
├── api/
│   └── main.py                 # FastAPI 服务（OpenAI 兼容格式）
├── data/                       # 解析缓存（自动生成）
├── Dockerfile & docker-compose.yml
├── run.py                      # 统一入口
├── icon.svg                    # 项目图标
├── curated_courses.json        # 课程介绍
├── requirements.txt            # 依赖列表
└── 本科辅修培养方案2025版.md     # 原始培养方案数据
```

---

## 设计要点

| 组件 | 说明 |
|------|------|
| **推理机制** | ReAct 模式：LLM 输出 `ACTION` 触发工具调用，结果回填后二次推理 |
| **短期记忆** | 每个会话独立的对话历史（最近 20 轮），以 `user` 字段区分 |
| **长期记忆** | 44 个辅修专业培养方案 + 1000+ 门已整理的课程资料 |
| **词嵌入** | 基于 `shibing624/text2vec-base-chinese` 的语义搜索，余弦相似度排序 |
| **规划能力** | LLM 自主推理 + 专用 Planner 双通道，考虑先修关系、开课学期、学分均衡 |
| **工具集** | `list_minors` · `search_minors` · `semantic_search` · `get_minor_detail` · `check_eligibility` · `multi_agent_search` · `search_courses` · `get_course_detail` · `list_minor_courses` |
| **API 格式** | 兼容 OpenAI `/v1/chat/completions`，支持流式 SSE |

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

| 端点 | 方法 | 说明 |
|------|------|------|
| `/v1/chat/completions` | POST | 对话（支持流式） |
| `/v1/models` | GET | 模型列表 |
| `/minors` | GET | 所有辅修专业列表 |
| `/minors/{name}` | GET | 某辅修详细信息 |
| `/` | GET | 服务健康检查 |

---

## 配置说明

### 依赖

```bash
pip install fastapi uvicorn httpx pydantic sentence-transformers
```

### 词嵌入模型

首次运行 `semantic_search` 时自动下载 `shibing624/text2vec-base-chinese`（约 400MB）。

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `HF_ENDPOINT` | `https://hf-mirror.com` | HuggingFace 镜像源（国内加速） |
| `DEEPSEEK_API_KEY` | — | DeepSeek API Key |
| `SENTENCE_TRANSFORMERS_HOME` | `~/.cache` | 模型缓存目录 |

---

## License

MIT
