# Gecko Plugins Models

`gecko.plugins.models` 是 Gecko 框架的统一模型接入层。它基于 [LiteLLM](https://github.com/BerriAI/litellm) 构建，为上层 Agent 和 RAG 模块提供了一致、健壮且类型安全的模型调用接口。

## 🌟 核心特性

*   **全能接入**：统一支持 OpenAI、Anthropic、Gemini 等主流 SaaS 模型，以及 DeepSeek、智谱 GLM 等国产大模型。
*   **云端 & 本地双轨**：无缝切换云端 API 和本地离线模型（Ollama, vLLM, SGLang），仅需修改配置。
*   **职责分离**：严格区分 **Chat Model**（对话/多模态）与 **Embedder**（向量化），避免接口混用。
*   **协议驱动**：完全遵循 Gecko Core 的 `StreamableModelProtocol` 和 `EmbedderProtocol`，支持流式输出和 Function Calling。
*   **多模态支持**：通过能力标识（Capability Flags）支持视觉（Vision）等多模态输入。

## 📂 目录结构

```text
gecko/plugins/models/
├── __init__.py          # 导出常用类
├── config.py            # 统一配置对象 (ModelConfig)
├── base.py              # 抽象基类定义 (BaseChatModel, BaseEmbedder)
├── chat.py              # Chat 模型通用实现 (LiteLLMChatModel)
├── embedding.py         # Embedding 模型通用实现 (LiteLLMEmbedder)
└── presets/             # 厂商预设配置
    ├── openai.py
    ├── zhipu.py
    ├── ollama.py
    └── ...
```

## 🚀 快速开始

### 1. 使用 Chat 模型 (对话/推理)

#### 方式 A：使用厂商预设 (推荐)

Gecko 预置了主流厂商的配置类，简化初始化流程。

```python
from gecko.plugins.models.presets.openai import OpenAIChat
from gecko.plugins.models.presets.zhipu import ZhipuChat

# 1. OpenAI
model_openai = OpenAIChat(
    api_key="sk-...", 
    model="gpt-4o"
)

# 2. 智谱 AI (GLM-4)
model_zhipu = ZhipuChat(
    api_key="...", 
    model="glm-4-plus"
)

# 调用 (支持异步)
response = await model_zhipu.acompletion([{"role": "user", "content": "你好"}])
print(response.choices[0].message.content)
```

#### 方式 B：连接本地模型 (Ollama / vLLM)

支持完全离线的本地推理，适合隐私敏感或无网环境。

```python
from gecko.plugins.models.presets.ollama import OllamaChat

# 连接本地 Ollama 服务
local_model = OllamaChat(
    model="llama3",                 # 对应 `ollama run llama3`
    base_url="http://localhost:11434",
    timeout=120.0                   # 本地推理可能较慢，建议增加超时
)

# 集成到 Agent
agent = Agent(model=local_model, ...)
```

#### 方式 C：通用配置 (自定义厂商)

对于未预设的厂商（如 DeepSeek、Moonshot），可使用通用适配器。

```python
from gecko.plugins.models.config import ModelConfig
from gecko.plugins.models.chat import LiteLLMChatModel

# 连接 DeepSeek (通过 OpenAI 兼容接口)
config = ModelConfig(
    model_name="deepseek-chat",
    api_key="sk-...",
    base_url="https://api.deepseek.com",
    max_retries=3
)

model = LiteLLMChatModel(config)
```

### 2. 使用 Embedding 模型 (RAG)

Embedding 模型用于将文本转换为向量，是 RAG 系统的核心组件。

```python
from gecko.plugins.models.presets.openai import OpenAIEmbedder
from gecko.plugins.models.presets.ollama import OllamaEmbedder

# 1. OpenAI Embedding
embedder_cloud = OpenAIEmbedder(
    api_key="sk-...",
    model="text-embedding-3-small",
    dimension=1536
)

# 2. 本地 Embedding (Ollama)
embedder_local = OllamaEmbedder(
    model="nomic-embed-text",
    base_url="http://localhost:11434",
    dimension=768  # 需手动指定维度以便向量库初始化
)

# 使用
vectors = await embedder_local.embed_documents(["Gecko 是一个 AI 框架"])
```

## ⚙️ 配置详解 (ModelConfig)

`ModelConfig` 是所有模型初始化的核心配置对象，支持以下参数：

| 参数 | 类型 | 描述 | 默认值 |
| :--- | :--- | :--- | :--- |
| `model_name` | str | 模型名称 (如 `gpt-4o`, `ollama/llama3`) | 必填 |
| `api_key` | str | API 密钥 | None |
| `base_url` | str | API 基础地址 (SaaS 可空，本地必填) | None |
| `timeout` | float | 请求超时时间 (秒) | 60.0 |
| `max_retries` | int | 失败重试次数 | 2 |
| `supports_vision` | bool | 是否支持视觉输入 | False |
| `supports_function_calling` | bool | 是否支持工具调用 | True |
| `extra_kwargs` | dict | 透传给 LiteLLM 的额外参数 | {} |

## 🔌 高级用法

### 多模态支持 (Vision)

Gecko 的 `Message` 对象支持多模态内容。要启用此功能，请确保模型配置了 `supports_vision=True`。

```python
from gecko.core.message import Message

# 初始化支持视觉的模型
model = OpenAIChat(model="gpt-4o", api_key="...")

# 发送图片
msg = Message.user(
    text="这张图片里有什么？",
    images=["https://example.com/image.jpg"]
)

response = await model.acompletion([msg.to_openai_format()])
```

### 流式输出 (Streaming)

所有 Chat 模型均实现了 `astream` 接口，返回标准化的 `StreamChunk`。

```python
async for chunk in model.astream(messages):
    if chunk.content:
        print(chunk.content, end="", flush=True)
```

## 🛠️ 扩展指南

如果您需要支持新的模型厂商，只需继承 `LiteLLMChatModel` 或 `LiteLLMEmbedder` 并预设配置即可：

```python
# 示例：添加 Moonshot (Kimi) 支持
from gecko.plugins.models.chat import LiteLLMChatModel
from gecko.plugins.models.config import ModelConfig

class MoonshotChat(LiteLLMChatModel):
    def __init__(self, api_key: str, model: str = "moonshot-v1-8k", **kwargs):
        config = ModelConfig(
            model_name=model,
            api_key=api_key,
            base_url="https://api.moonshot.cn/v1",
            **kwargs
        )
        super().__init__(config)
```