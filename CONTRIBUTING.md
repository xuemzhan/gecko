# 贡献指南 (Contributing to Gecko)

感谢你对 Gecko Framework 感兴趣！我们致力于构建一个工业级、高可靠的 Agent 开发框架。

## 🚀 快速开始

### 1. 环境准备

Gecko 使用 `rye` 或标准 `pip` 进行依赖管理。建议使用 Python 3.9+。

```bash
# Fork 本仓库并 Clone 到本地
git clone https://github.com/YOUR_USERNAME/gecko.git
cd gecko

# 创建虚拟环境并安装依赖 (开发模式)
pip install -e ".[dev,full]"

# 或者使用 rye
rye sync
```

### 2. 代码规范

我们使用以下工具保证代码质量：
*   **Lint**: `ruff`
*   **Type Check**: `mypy` (Strict mode)
*   **Format**: `ruff format`

在提交代码前，请务必运行检查：

```bash
# 运行所有检查
ruff check .
mypy gecko
```

### 3. 运行测试

Gecko 拥有完善的测试套件。

```bash
# 运行单元测试
pytest tests/core

# 运行所有测试 (包含集成测试)
pytest
```

## 📝 开发流程

1.  在 GitHub 上 Fork 本仓库。
2.  基于 `main` 分支创建一个新分支：`git checkout -b feature/my-feature`。
3.  编写代码并添加对应的测试用例。
4.  确保所有测试通过，且覆盖率没有明显下降。
5.  提交 Pull Request (PR)。

## 📐 设计原则

在贡献代码时，请遵循 Gecko 的核心设计哲学：

*   **Async First**: 尽量避免同步阻塞操作，I/O 密集型任务请使用 `ThreadOffloadMixin`。
*   **Protocol Driven**: 使用 `Protocol` 定义接口，而非强制继承。
*   **No Magic**: 避免隐式的黑魔法，显式优于隐式。
*   **Type Safety**: 所有公共 API 必须有完整的类型注解。

## 🐛 提交 Bug

请使用 GitHub Issues 提交 Bug，并包含以下信息：
*   Gecko 版本
*   Python 版本
*   最小复现代码 (Minimal Reproducible Example)
*   错误堆栈信息

---
Happy Coding! 🦎