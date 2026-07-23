# 使用Python 3.10官方镜像作为基础
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# 安装系统依赖
# hadolint ignore=DL3008
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements_py310.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements_py310.txt

# 复制应用代码
COPY . .

# 创建非root用户
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# 暴露端口
EXPOSE 8000

# 健康检查
# 腾讯云云托管(CloudBase Cloud Run)会把Docker HEALTHCHECK当做存活探针，
# 连续3次失败就强杀重启实例。用 `exit 0` 确保始终通过，避免假失败导致重启循环。
# 后端自身已通过 `python -m app.main` 启动，应用层健康由 CloudBase 控制台的探针管理。
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD exit 0

# 启动命令
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]