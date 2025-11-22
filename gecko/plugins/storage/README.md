# Gecko Storage Plugin

**Gecko Storage** 是 Gecko 框架的统一持久化层，旨在为 AI 智能体提供高性能、异步非阻塞且并发安全的存储能力。它统一了 **KV 会话存储 (Session)** 和 **向量知识库 (Vector)** 的访问接口。

## 🌟 核心特性

*   **异步优先 (Async First)**: 所有接口均为异步 (`async/await`)。内部通过 `ThreadOffloadMixin` 将同步 I/O（如 SQLite, Chroma 操作）卸载至线程池，杜绝 Event Loop 阻塞。
*   **并发与进程安全**:
    *   **线程安全**: 内置 `asyncio.Lock` 保证协程间互斥。
    *   **进程安全**: 引入 `FileLock` 机制，确保在多进程环境（如 Gunicorn/Uvicorn Workers）下操作 SQLite/文件存储时的数据完整性。
*   **统一接口**:
    *   `SessionInterface`: 用于存储对话历史、Agent 状态 (Get/Set/Delete)。
    *   `VectorInterface`: 用于 RAG 知识库检索 (Upsert/Search)。
*   **健壮性**:
    *   统一抛出 `StorageError`，屏蔽底层驱动（Redis, SQLite, LanceDB）的差异化异常。
    *   自动处理 `metadata` 为空的边缘情况，防止底层数据库崩溃。
*   **高级检索**: 支持向量检索时的 **Metadata Filtering**（元数据过滤）。
*   **插件化扩展**: 支持通过 Python EntryPoints (`gecko.storage.backends`) 自动发现第三方存储后端。

## 📦 安装

Storage 模块依赖具体的后端驱动，请根据需求安装：

```bash
# 基础 (仅接口)
pip install gecko-ai

# SQLite (内置支持，推荐开发/测试)
pip install sqlalchemy sqlmodel filelock

# Redis (推荐生产环境 Session 存储)
pip install redis

# ChromaDB (本地向量库)
pip install chromadb

# LanceDB (高性能本地向量库)
pip install lancedb
```

## 🚀 快速开始

### 1. 初始化存储

使用 `create_storage` 工厂函数，通过 URL 自动加载后端：

```python
from gecko.plugins.storage.factory import create_storage

# SQLite Session 存储
session_store = await create_storage("sqlite:///./sessions.db")

# LanceDB 向量存储
vector_store = await create_storage("lancedb://./knowledge.db")
```

### 2. 会话存储 (Session Storage)

适用于存储 Agent 的短期记忆或状态。

```python
# 写入状态
await session_store.set("user_123", {
    "name": "Alice",
    "history": ["Hi", "Hello"],
    "balance": 100
})

# 读取状态
data = await session_store.get("user_123")
print(data["name"])  # Alice

# 删除
await session_store.delete("user_123")
```

### 3. 向量检索 (Vector Storage & RAG)

适用于 RAG（检索增强生成）场景。

```python
# 写入文档 (Upsert)
documents = [
    {"id": "1", "text": "Apple is a fruit", "embedding": [0.1, 0.1, ...], "metadata": {"type": "fruit"}},
    {"id": "2", "text": "Python is a language", "embedding": [0.9, 0.8, ...], "metadata": {"type": "tech"}}
]
await vector_store.upsert(documents)

# 向量搜索 (Search)
query_vec = [0.1, 0.1, ...]  # 你的 Embedding 向量
results = await vector_store.search(query_vec, top_k=3)

# [高级] 带过滤的搜索 (Metadata Filtering)
# 仅检索 type="tech" 的文档
tech_results = await vector_store.search(
    query_vec, 
    top_k=3, 
    filters={"type": "tech"}
)
```

## 🔌 支持的后端 (Backends)

| Scheme | 后端 | 类型 | URL 示例 | 适用场景 |
| :--- | :--- | :--- | :--- | :--- |
| `sqlite` | SQLite | Session | `sqlite:///./data.db` | 开发、单机部署、中低并发 |
| `redis` | Redis | Session | `redis://localhost:6379/0?ttl=3600` | **生产环境**、分布式、高并发 |
| `chroma` | ChromaDB | Vector + Session | `chroma://./chroma_db?collection=my_app` | 本地 RAG、原型开发 |
| `lancedb` | LanceDB | Vector | `lancedb://./lance_db?table=vectors` | **生产环境**、高性能本地向量检索 |

## 🛠️ 高级架构说明

### 混合存储 (Mixins)

Gecko Storage 的强大功能源于 Mixin 组合模式：

1.  **`ThreadOffloadMixin`**:
    *   **作用**: 将同步 I/O 操作（如 `sqlite3.connect`, `lance.write`）自动封装到 `anyio.to_thread.run_sync` 中运行。
    *   **收益**: 即使使用同步数据库驱动，也不会阻塞主线程的 Event Loop，保证高并发下的响应能力。

2.  **`AtomicWriteMixin`**:
    *   **作用**: 提供双层锁机制。
        *   **Async Lock**: 协程级互斥。
        *   **File Lock**: 进程级互斥（基于 `filelock` 库）。
    *   **收益**: 彻底解决 SQLite/JSON 文件在多 Worker（如 Gunicorn）并发写入时的 `Database is locked` 或数据损坏问题。

### 自定义后端扩展

你可以通过继承 `AbstractStorage` 并组合 Mixin 来快速实现自定义后端。

**步骤 1**: 实现存储类

```python
from gecko.plugins.storage.abc import AbstractStorage
from gecko.plugins.storage.interfaces import SessionInterface
from gecko.plugins.storage.mixins import ThreadOffloadMixin
from gecko.plugins.storage.registry import register_storage

@register_storage("myfile")  # 注册 URL scheme
class MyFileStorage(AbstractStorage, SessionInterface, ThreadOffloadMixin):
    async def initialize(self):
        # ... 初始化逻辑
        pass

    async def get(self, session_id: str):
        # 在线程池中执行同步读取
        return await self._run_sync(self._read_from_disk, session_id)
    
    def _read_from_disk(self, sid):
        # 同步 IO 代码
        ...
```

**步骤 2**: (可选) 通过 `pyproject.toml` 发布插件

Gecko 支持自动发现安装在环境中的第三方插件：

```toml
[project.entry-points."gecko.storage.backends"]
myfile = "my_package.storage:MyFileStorage"
```

## ⚠️ 常见问题

1.  **`sqlite3.OperationalError: database is locked`**:
    *   确保你安装了 `filelock`：`pip install filelock`。
    *   Gecko 会自动启用 WAL 模式和文件锁来解决此问题。

2.  **Vector Search 报错 `metadata` 为空**:
    *   Gecko V0.1+ 已修复此问题。现在的后端会自动将 `None` 的元数据转换为空字典或数据库接受的格式。

3.  **Redis 连接失败**:
    *   请检查 URL 格式。Gecko 会捕获连接异常并抛出统一的 `gecko.core.exceptions.StorageError`，方便上层业务处理。