"""配置管理：从环境变量与 .env 文件读取服务与上游 LLM 配置。

约定（与 .env.example 一致）：
- API_KEY        服务鉴权凭证（清小搭广场 Bearer Token）；留空则关闭鉴权（任意请求放行）；
                 未设置时使用开发默认值 sk-tsinghua-minor-dev（本地调试可免凭证）
- LLM_API_KEY    上游大模型 API Key（OpenAI 兼容）
- LLM_BASE_URL   上游端点，填到版本段为止，如 https://api.deepseek.com/v1
- LLM_MODEL      上游模型名
- MODEL_NAME     对外暴露的模型名（/v1/models 与响应 model 字段），不暴露上游模型名
- KNOWLEDGE_DIR  知识库目录（默认仓库内 minors/）
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 开发默认凭证：.env.example 中 API_KEY 留空时使用的值
DEV_API_KEY = "sk-tsinghua-minor-dev"

DEFAULTS = {
    "API_KEY": DEV_API_KEY,
    "LLM_API_KEY": "",
    "LLM_BASE_URL": "https://api.deepseek.com/v1",
    "LLM_MODEL": "deepseek-v4-flash",
    # 对外暴露的模型名（/v1/models 与响应中的 model 字段）；不暴露上游模型名
    "MODEL_NAME": "tsinghua-minor-advisor",
    "LLM_TIMEOUT": "60",
    "LLM_INCLUDE_USAGE": "1",
    "LLM_TEMPERATURE": "0.3",
    "LLM_MAX_TOKENS": "2048",
    "MAX_REACT_STEPS": "15",
    "TOOL_RESULT_MAX_CHARS": "6000",
    "KNOWLEDGE_DIR": "",
    "HOST": "0.0.0.0",
    "PORT": "8000",
}


def _load_dotenv(path: Path) -> None:
    """极简 .env 加载器（不依赖 python-dotenv）：KEY=VALUE 每行一条，支持 # 注释。"""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass(frozen=True)
class Config:
    api_key: str
    llm_api_key: str
    llm_base_url: str
    llm_model: str
    model_name: str
    llm_timeout: float
    llm_include_usage: bool
    temperature: float
    max_tokens: int
    max_react_steps: int
    tool_result_max_chars: int
    knowledge_dir: Path
    host: str
    port: int

    @property
    def is_dev_default_auth(self) -> bool:
        """是否使用开发默认凭证（此时本地调试可免凭证访问）。"""
        return self.api_key == DEV_API_KEY


def get_config() -> Config:
    _load_dotenv(PROJECT_ROOT / ".env")
    env = os.environ

    def _get(key: str):
        return env.get(key, DEFAULTS[key])

    knowledge_dir_raw = _get("KNOWLEDGE_DIR")
    knowledge_dir = (
        Path(knowledge_dir_raw).resolve() if knowledge_dir_raw else PROJECT_ROOT / "minors"
    )
    return Config(
        api_key=_get("API_KEY"),
        llm_api_key=_get("LLM_API_KEY"),
        llm_base_url=_get("LLM_BASE_URL").rstrip("/"),
        llm_model=_get("LLM_MODEL"),
        model_name=_get("MODEL_NAME"),
        llm_timeout=float(_get("LLM_TIMEOUT")),
        llm_include_usage=_get("LLM_INCLUDE_USAGE") not in ("", "0", "false", "False"),
        temperature=float(_get("LLM_TEMPERATURE")),
        max_tokens=int(_get("LLM_MAX_TOKENS")),
        max_react_steps=int(_get("MAX_REACT_STEPS")),
        tool_result_max_chars=int(_get("TOOL_RESULT_MAX_CHARS")),
        knowledge_dir=knowledge_dir,
        host=_get("HOST"),
        port=int(_get("PORT")),
    )
