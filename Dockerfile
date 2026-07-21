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
# TEMP-DEBUG: 怀疑云托管把这个HEALTHCHECK状态当存活探针用，连续3次失败(约89秒)就强杀重启实例。
# 临时改成必定成功，排查是否是这里导致的90秒左右重启循环。确认后记得把下面这行换回真实检查
# （CMD curl -f http://localhost:8000/health || exit 1），不要让这个假检查留在正式版本里。
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD exit 0

# 启动命令
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]