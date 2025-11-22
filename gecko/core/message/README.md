# Gecko Core Message

`gecko.core.message` 定义了 Gecko 框架中通用的消息数据结构。它旨在提供一个统一、类型安全且支持多模态（Multimodal）的消息表示，同时保持与 OpenAI Chat Completion API 的高度兼容性。

## 🌟 核心特性

*   **标准化格式**：统一封装 User, Assistant, System 和 Tool 消息。
*   **多模态支持**：内置对文本和图片混合内容的支持（基于 `ContentBlock`）。
*   **资源管理**：提供 `MediaResource` 用于处理本地文件、URL 和 Base64 编码，支持异步加载。
*   **OpenAI 兼容**：提供 `to_openai_format()` 和 `from_openai()` 方法，实现无缝转换。
*   **类型安全**：基于 Pydantic 构建，确保存储和传输过程中的数据完整性。

## 📂 模块结构

该包包含以下核心组件：

| 模块文件 | 类/组件 | 描述 |
| :--- | :--- | :--- |
| **`model.py`** | `Message` | 消息的主体定义，包含角色、内容、工具调用等字段，以及便捷的工厂方法。 |
| **`resources.py`** | `MediaResource`<br>`ContentBlock` | 定义多模态资源（如图片）的数据结构，负责文件读取和 Base64 编码。 |

## 🚀 使用指南

### 1. 基础文本消息
使用工厂方法快速创建不同角色的消息。

```python
from gecko.core.message import Message

# 用户消息
msg_user = Message.user("Tell me a joke.")

# 助手消息
msg_ai = Message.assistant("Why did the chicken cross the road?")

# 系统消息
msg_sys = Message.system("You are a helpful comedian.")
```

### 2. 多模态消息（图片支持）
Gecko 支持在一条消息中混合文本和图片。支持本地路径（自动转 Base64）和 URL。

**同步方式：**
```python
# 自动读取本地文件并编码为 Base64
msg = Message.user(
    text="What's in this image?",
    images=["./photo.jpg", "https://example.com/logo.png"]
)
```

**异步方式（推荐用于 Web 服务）：**
防止大文件读取阻塞 Event Loop。
```python
msg = await Message.user_async(
    text="Analyze this document",
    images=["./large_scan.png"]
)
```

### 3. 工具调用与结果
处理 Agent 的工具调用流程。

```python
# 1. 模型返回的工具调用消息
msg_call = Message.assistant(
    content="",
    # 这里的结构通常由 LLM 驱动自动生成
    tool_calls=[{
        "id": "call_123",
        "function": {"name": "search", "arguments": "..."}
    }]
)

# 2. 工具执行后的结果消息
msg_result = Message.tool_result(
    tool_call_id="call_123",
    tool_name="search",
    content={"status": "success", "data": "..."}  # 支持字典自动序列化
)
```

### 4. OpenAI 格式转换
方便与 LiteLLM 或其他 OpenAI 兼容接口交互。

```python
# 导出为 OpenAI 格式字典
payload = msg.to_openai_format()
# Result: {'role': 'user', 'content': 'Hello'}

# 从 OpenAI 格式导入
raw_data = {"role": "assistant", "content": "Hi"}
msg = Message.from_openai(raw_data)
```

## 📦 核心类详解

### `Message`
主消息对象。

*   **属性**:
    *   `role`: 角色 (`user`, `assistant`, `system`, `tool`)
    *   `content`: 内容，可以是字符串或 `ContentBlock` 列表。
    *   `tool_calls`: 工具调用列表（可选）。
    *   `name`: 发送者名称（可选）。

*   **方法**:
    *   `get_text_content()`: 提取纯文本内容（忽略图片）。
    *   `truncate_content(length)`: 截断文本内容（保留图片）。
    *   `is_empty()`: 检查内容是否为空。

### `MediaResource`
媒体资源封装。

*   **支持源**:
    *   `url`: 网络图片地址。
    *   `base64_data`: 图片的 Base64 编码字符串。
    *   `path`: 本地文件路径（通过 `from_file` 转换）。

*   **方法**:
    *   `from_file(path)`: 同步读取本地文件。
    *   `from_file_async(path)`: 异步读取本地文件（线程池卸载）。
    *   `to_openai_image_url()`: 生成 OpenAI API 所需的 `image_url` 结构。