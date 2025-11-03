#!/bin/bash
# 这个脚本用于生成 gecko 项目重构后的目录结构。
# 它会创建所有必要的目录和空的 __init__.py 文件，
# 以确保所有模块都被正确识别为Python包。

set -e # 如果任何命令失败，脚本将立即退出

echo "Creating refactored directory structure for gecko..."

# --- 核心组件 ---
echo "Creating core components..."
CORE_COMPONENTS=("agent" "team" "workflow" "models" "tools" "knowledge" "guardrails" "eval" "session" "utils")
for component in "${CORE_COMPONENTS[@]}"; do
    mkdir -p "$component"
    touch "$component/__init__.py"
done
touch exceptions.py

# --- API (FastAPI 应用层) ---
echo "Creating api layer..."
mkdir -p api/routers api/security api/interfaces
touch api/__init__.py api/app.py api/settings.py api/config.py
touch api/security/__init__.py api/security/auth.py
touch api/interfaces/__init__.py
ROUTERS=("health" "home" "session" "memory" "knowledge" "evals" "metrics")
for router in "${ROUTERS[@]}"; do
    touch "api/routers/${router}.py"
done
touch api/routers/__init__.py

# --- Client (内部遥测客户端) ---
echo "Creating internal client..."
mkdir -p client
touch client/__init__.py client/telemetry.py client/schemas.py client/routes.py

# --- Components (高级业务逻辑) ---
echo "Creating high-level components..."
mkdir -p components
touch components/__init__.py components/culture_manager.py components/memory_manager.py components/session_summary.py

# --- Storage (数据持久化层) ---
echo "Creating database layer..."
mkdir -p storage/schemas storage/backends
touch storage/__init__.py storage/base.py storage/utils.py
# DB Schemas
touch storage/schemas/__init__.py storage/schemas/culture.py storage/schemas/evals.py storage/schemas/knowledge.py storage/schemas/memory.py
# DB Backends
touch storage/backends/__init__.py
touch storage/backends/sqlalchemy_base.py
BACKENDS=("sqlite" "postgres" "mongo" "redis" "mysql" "gcs_json" "in_memory")
for backend in "${BACKENDS[@]}"; do
    touch "storage/backends/${backend}.py"
done

echo "Structure created successfully!"
ls -R ./