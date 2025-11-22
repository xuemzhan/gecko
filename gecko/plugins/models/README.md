# Gecko Models Plugin

`gecko.plugins.models` 是 Gecko 框架的核心模型接入层。它采用 **驱动器模式 (Driver Pattern)** 和 **注册表模式 (Registry Pattern)** 设计，为上层应用提供了一个统一、健壮且高度可扩展的模型调用接口。

## 🌟 核心特性

*   **驱动器架构 (Driver Architecture)**：核心层与具体实现解耦。默认内置 `LiteLLMDriver` 以支持 100+ 种模型，同时支持扩展原生 SDK 驱动（如 `NativeOpenAIDriver`, `NativeGeminiDriver`）。
*   **全能接入**：无缝支持 OpenAI, Anthropic, Zhipu (智谱AI) 等 SaaS 服务，以及 Ollama, vLLM 等本地离线模型。
*   **健壮性设计**：内置 **防腐层 (Anti-Corruption Layer)**，通过 `LiteLLMAdapter` 自动清洗上游响应，彻底解决 Pydantic 版本冲突和序列化警告问题。
*   **协议驱动**：严格遵循 Gecko Core 的 `StreamableModelProtocol` 和 `EmbedderProtocol`。
*   **多模态与流式**：原生支持视觉输入 (Vision) 和 Token 级流式输出 (Streaming)。

## 📂 目录结构

```text
gecko/plugins/models/
├── __init__.py                  # 模块入口 (导出常用类)
├── config.py                    # 统一配置对象 (ModelConfig)
├── base.py                      # 抽象基类 (BaseChatModel, BaseEmbedder)
├── factory.py                   # 工厂方法 (create_model)
├── registry.py                  # 驱动注册表 (@register_driver)
├── adapter.py                   # 响应清洗适配器 (ACL)
├── embedding.py                 # Embedding 模型通用实现
├── drivers/                     # 驱动器实现目录
│   ├── __init__.py
│   └── litellm_driver.py        # [默认] LiteLLM 通用驱动
└── presets/                     # 厂商预设配置 (简化初始化)
    ├── __init__.py
    ├── ollama.py
    ├── openai.py
    └── zhipu.py
```

## 🚀 快速开始

### 1. 使用厂商预设 (推荐)

预设类 (Presets) 封装了复杂的配置细节，是使用特定厂商模型的最佳方式。

#### 智谱 AI (ZhipuGLM)

```python
import os
from gecko.plugins.models.presets.zhipu import ZhipuChat
from gecko.core.message import Message

# 初始化 (使用 OpenAI 兼容协议连接)
model = ZhipuChat(
    api_key=os.getenv("ZHIPU_API_KEY"), 
    model="glm-4-flash"
)

# 调用
msg = Message.user("你好，请介绍一下 Gecko 框架")
response = await model.acompletion([msg.to_openai_format()])
print(response.choices[0].message["content"])
```

#### 本地模型 (Ollama)

适合离线环境或隐私敏感场景。

```python
from gecko.plugins.models.presets.ollama import OllamaChat

# 连接本地 Ollama (默认端口 11434)
local_model = OllamaChat(
    model="llama3",  
    base_url="http://localhost:11434",
    timeout=120.0  # 本地推理建议增加超时
)

response = await local_model.acompletion([...])
```

### 2. 使用通用配置 (Factory 模式)

对于未提供预设的厂商（如 DeepSeek、Moonshot），或者需要动态加载配置的场景，使用 `ModelConfig` 和 `create_model`。

```python
from gecko.plugins.models.config import ModelConfig
from gecko.plugins.models.factory import create_model

# 配置 DeepSeek (通过 LiteLLM 驱动)
config = ModelConfig(
    model_name="deepseek-chat",
    driver_type="litellm",  # 指定驱动
    api_key="sk-...",
    base_url="https://api.deepseek.com",
    max_retries=3
)

model = create_model(config)
```

### 3. 使用 Embedding 模型 (RAG)

```python
from gecko.plugins.models.presets.openai import OpenAIEmbedder

embedder = OpenAIEmbedder(
    api_key="sk-...",
    model="text-embedding-3-small",
    dimension=1536
)

vectors = await embedder.embed_documents(["Gecko 是一个 AI 智能体框架"])
```

## ⚙️ 配置详解 (ModelConfig)

| 参数 | 类型 | 说明 | 默认值 |
| :--- | :--- | :--- | :--- |
| `model_name` | str | 模型名称 (如 `gpt-4o`, `ollama/qwen2`) | **必填** |
| `driver_type` | str | 驱动类型 (`litellm`, `openai_native` 等) | `"litellm"` |
| `api_key` | str | API 密钥 | `None` |
| `base_url` | str | API 基础地址 (SaaS 可空，本地必填) | `None` |
| `timeout` | float | 请求超时时间 (秒) | `60.0` |
| `max_retries` | int | 失败重试次数 | `2` |
| `supports_vision` | bool | 启用视觉支持 | `False` |
| `extra_kwargs` | dict | 透传给驱动底层的额外参数 | `{}` |

## 🔌 高级特性

### 流式输出 (Streaming)

所有 Chat 模型均实现了 `astream` 接口，返回标准化的 `StreamChunk`。

```python
async for chunk in model.astream(messages):
    if chunk.content:
        print(chunk.content, end="", flush=True)
```

### 多模态 (Vision)

支持发送图片 URL 或 Base64 数据。需确保 `supports_vision=True`。

```python
vision_model = ZhipuChat(api_key="...", model="glm-4v-flash")

msg = Message.user(
    text="这张图里有什么？",
    images=["https://example.com/photo.jpg"]
)

await vision_model.acompletion([msg.to_openai_format()])
```

## 🛠️ 架构扩展指南

Gecko 的模型层设计支持无限扩展。如果您需要接入特殊的 SDK（例如 Google 原生 SDK 以支持 Video 输入），可以编写自定义驱动。

### 如何增加新的驱动？

1.  **创建驱动文件**：在 `gecko/plugins/models/drivers/` 下创建 `my_custom_driver.py`。
2.  **继承基类**：继承 `BaseChatModel`。
3.  **注册驱动**：使用 `@register_driver("my_driver_name")` 装饰器。

```python
# gecko/plugins/models/drivers/my_custom_driver.py
from gecko.plugins.models.base import BaseChatModel
from gecko.plugins.models.registry import register_driver

@register_driver("my_native_sdk")
class MyNativeDriver(BaseChatModel):
    async def acompletion(self, messages, **kwargs):
        # 调用原生 SDK 逻辑
        native_response = await my_sdk.chat(...)
        # 转换为 Gecko 标准 CompletionResponse
        return CompletionResponse(...)

    async def astream(self, messages, **kwargs):
        # 实现流式逻辑
        ...
```

4.  **使用新驱动**：

```python
config = ModelConfig(
    model_name="my-model",
    driver_type="my_native_sdk",  # 指定新驱动
    ...
)
model = create_model(config)
```

---

**Gecko Team**