"""启动入口。

用法：
    python run.py
或：
    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""
import uvicorn

from agent.config import get_config

if __name__ == "__main__":
    cfg = get_config()
    uvicorn.run("app.main:app", host=cfg.host, port=cfg.port, reload=False)
