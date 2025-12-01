# 存储后端参考

Gecko 使用统一的 URL Scheme 管理所有存储连接。

| 协议头 | 后端 | 类型 | 特性 | 依赖 |
| :--- | :--- | :--- | :--- | :--- |
| `sqlite://` | **SQLite** | KV | 默认后端。支持 WAL 模式和 **跨进程文件锁 (FileLock)**，适合单机高并发。 | `sqlmodel` |
| `redis://` | **Redis** | KV | 生产环境推荐。支持 TTL 自动过期，极高性能。 | `redis` |
| `chroma://` | **ChromaDB**| Vector | 功能全面的向量数据库，支持复杂的元数据过滤。 | `chromadb` |
| `lancedb://`| **LanceDB** | Vector | 基于文件的向量库，无需服务端，读取速度极快。 | `lancedb` |

## 配置示例

### SQLite (推荐单机)
```python
storage = await create_storage("sqlite:///./data/sessions.db")
```

### Redis (推荐集群)
```python
# 设置 1 小时过期
storage = await create_storage("redis://localhost:6379/0?ttl=3600")
```

### ChromaDB (RAG)
```python
vector_store = await create_storage("chroma:///./data/chroma_db?collection=knowledge")
```

### LanceDB (RAG)
```python
# 指定维度以便自动建表
vector_store = await create_storage("lancedb:///./data/lance_db?table=vectors&dim=1536")
```