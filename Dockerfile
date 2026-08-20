# TsingXiaoDa Agent —— 清华大学辅修顾问智能体
# 构建：distrobox-host-exec docker build -t tsingxiaoda-minor:latest .
# 运行：distrobox-host-exec docker run -d --name tsingxiaoda-minor \
#         -p 8000:8000 \
#         -e LLM_API_KEY="$LLM_API_KEY" \
#         -e API_KEY="${API_KEY:-sk-tsinghua-minor-dev}" \
#         tsingxiaoda-minor:latest
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 先装依赖，利用层缓存；优先国内镜像（清华），失败则回退官方 PyPI
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple \
    || pip install --no-cache-dir -r requirements.txt

# 复制应用代码与知识库（.env 已被 .dockerignore 排除，密钥仅由运行时环境变量注入）
COPY agent ./agent
COPY app ./app
COPY minors ./minors
COPY run.py .

# 非 root 运行
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# 轻量健康检查：TCP 探测 8000 端口（避免鉴权边界问题）
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import socket; s=socket.create_connection(('127.0.0.1', 8000), 3); s.close()"

CMD ["python", "run.py"]
