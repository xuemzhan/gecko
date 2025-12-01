# 安装指南

Gecko 尚未发布到 PyPI，目前建议通过源码安装。

## 环境要求

*   **Python**: 3.9 或更高版本
*   **OS**: Linux, macOS, Windows

## 安装步骤

### 1. 基础安装
仅包含核心 Agent、Workflow 和 SQLite 支持，体积最小。

```bash
pip install gecko-ai
```

### 2. 安装 RAG 支持
如果你需要使用知识库功能（向量数据库、Embedding），请安装 `rag` 选项。这将安装 `chromadb`, `lancedb`, `pypdf` 等依赖。

```bash
pip install "gecko-ai[rag]"
```

### 3. 安装所有功能
包含 Redis、OpenTelemetry、YAML 解析等所有可选依赖。

```bash
pip install "gecko-ai[all]"
```

## 验证安装

```bash
gecko --version
# 输出示例: Gecko CLI, version 0.3.1
```