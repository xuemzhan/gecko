# 生产环境部署指南

Gecko 是为高并发环境设计的。以下是将 Gecko 应用部署到生产环境的最佳实践。

## 1. 并发模型选择

Gecko 核心是异步的 (`async/await`)。在部署 Web 服务（如 FastAPI 封装 Agent）时，**必须使用异步 Worker**。

### ❌ 不要使用
*   同步的 Gunicorn Worker (`sync`)
*   Flask 自带的开发服务器

### ✅ 推荐使用
*   **Uvicorn**: 单进程高性能。
*   **Gunicorn + Uvicorn Workers**: 多进程管理 + 异步处理。

```bash
# 安装依赖
pip install gunicorn uvicorn

# 启动命令 (假设你的应用入口在 main.py 的 app 对象)
# -k uvicorn.workers.UvicornWorker 是关键！
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## 2. 存储后端选择

虽然 Gecko 对 SQLite 做了大量优化（WAL 模式 + 文件锁），但在极高并发或容器化环境下，建议切换后端。

### 单机 / 少量并发
*   **推荐**: SQLite (`sqlite:///./data.db`)
*   **注意**: Gecko 内置的 `FileLock` 可以保证 Gunicorn 多 Worker 写入 SQLite 时不损坏数据库，但会牺牲部分性能。

### 集群 / 高并发
*   **推荐**: Redis (`redis://host:6379`)
*   **优势**: 无锁竞争，极高的读写吞吐，天生支持 TTL（会话过期）。

## 3. Docker 部署示例

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# 1. 安装系统依赖 (如果需要编译库)
RUN apt-get update && apt-get install -y build-essential curl

# 2. 安装 Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt "gecko-ai[redis]"

# 3. 复制源码
COPY . .

# 4. 设置环境变量
ENV GECKO_LOG_FORMAT=json
ENV GECKO_LOG_LEVEL=INFO

# 5. 启动服务
CMD ["gunicorn", "main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8000"]
```