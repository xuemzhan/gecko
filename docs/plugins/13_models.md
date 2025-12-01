# 模型适配 (Models)

Gecko 通过插件系统支持几乎所有主流的大语言模型。底层依赖 `LiteLLM` 进行协议标准化。

## 配置与初始化

所有模型均通过 `ModelConfig` 进行配置。

```python
from gecko.plugins.models.config import ModelConfig
from gecko.plugins.models.factory import create_model

# 通用配置
config = ModelConfig(
    model_name="gpt-4o",
    api_key="sk-...",
    timeout=60.0,
    max_retries=2
)

model = create_model(config)
```

## 预设类 (Presets)

为了方便使用，Gecko 提供了常见 Provider 的预设类：

### ZhipuChat (智谱 AI)
针对智谱 GLM 系列模型优化。
```python
from gecko.plugins.models import ZhipuChat
model = ZhipuChat(api_key="...", model="glm-4-flash")
```

### OpenAIChat
标准 OpenAI 接口模型（包括 DeepSeek 等兼容接口）。
```python
from gecko.plugins.models import OpenAIChat
model = OpenAIChat(api_key="...", model="gpt-3.5-turbo")
```

### OllamaChat (本地模型)
连接本地运行的 Ollama 服务。
```python
from gecko.plugins.models import OllamaChat
model = OllamaChat(model="llama3", base_url="http://localhost:11434")
```

## Embedding 模型

用于 RAG 场景的向量化模型。

```python
from gecko.plugins.models.embedding import LiteLLMEmbedder

embedder = LiteLLMEmbedder(
    config=ModelConfig(model_name="text-embedding-3-small", api_key="..."),
    dimension=1536
)
```

## Token 计数策略

Gecko 的 `LiteLLMDriver` 实现了智能的 Token 计数回退策略，以确保性能与准确性的平衡：

1.  **Tiktoken (精确)**: 如果环境中有 `tiktoken` 且模型匹配 (如 GPT 系列)，优先使用。
2.  **LiteLLM (适配)**: 如果未匹配，尝试调用 `litellm.encode`。
3.  **估算 (兜底)**: 如果上述均失败或依赖缺失，按 `char_length / 3` 进行快速估算，防止阻塞。
```

---

### 5. API 参考 (docs/api/public_api.md)

```markdown
# 公共 API 参考 (L1 Stable)

以下 API 被视为 Gecko v1.0 稳定接口，在 v1.x 版本周期内保证向后兼容。

建议直接从顶层包 `gecko` 导入使用。

## 核心组件

| 类/函数 | 描述 |
| :--- | :--- |
| `Agent` | 智能体门面，组装模型与工具 |
| `AgentBuilder` | 构建 Agent 的流式接口 |
| `Workflow` | DAG 工作流引擎 |
| `step` | 装饰器，将函数标记为工作流节点 |
| `Next` | 控制流指令，用于动态跳转 |
| `Team` | 多智能体并行执行引擎 |

## 数据模型

| 类 | 描述 |
| :--- | :--- |
| `Message` | 统一消息对象 (`role`, `content`, `tool_calls`) |
| `AgentOutput` | Agent 执行结果，包含文本、工具调用和 Usage |
| `TokenUsage` | Token 消耗统计 |
| `Role` | 角色常量 (`USER`, `ASSISTANT`, `SYSTEM`) |

## 记忆与结构化

| 类 | 描述 |
| :--- | :--- |
| `TokenMemory` | 基于 Token 计数的滑动窗口记忆 |
| `SummaryTokenMemory` | 支持自动摘要的记忆模块 |
| `StructureEngine` | 结构化输出解析引擎 |

## 版本信息

```python
import gecko
print(gecko.__version__)  # e.g. "0.3.1"
```

## 使用示例

```python
from gecko import AgentBuilder, Message, Workflow, step, Next

# 1. Agent
agent = AgentBuilder().with_model(...).build()

# 2. Workflow Node
@step("node_1")
async def my_node(ctx):
    return Next("node_2", input="data")
```