FROM python:3.12-slim

WORKDIR /app

# 安装依赖（利用 Docker 层缓存）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 创建非 root 用户
RUN addgroup --system --gid 1001 appuser \
    && adduser --system --uid 1001 --gid 1001 appuser

# 复制源码
COPY agent/ agent/
COPY api/ api/
COPY run.py .
COPY 本科辅修培养方案2026版.md .

# 创建数据缓存目录，设为全局可写以便兼容宿主机 volume 挂载
RUN mkdir -p /app/data && chmod 777 /app/data

USER appuser

EXPOSE 8000

ENV PYTHONUNBUFFERED=1
ENV LANG=C.UTF-8

CMD ["python", "run.py", "--mode", "api", "--host", "0.0.0.0", "--port", "8000"]
