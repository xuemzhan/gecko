# Gecko Core Engine Module

## 概述

`gecko.core.engine` 模块是 Gecko 多智能体框架的核心推理引擎，负责协调 LLM 调用、工具执行和上下文管理。该模块实现了 ReAct (Reasoning + Acting) 范式，支持流式输出、结构化响应和自动错误恢复。

## 架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        CognitiveEngine (ABC)                     │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐            │
│  │ ModelProtocol│ │   ToolBox    │ │ TokenMemory  │            │
│  └──────┬───────┘ └──────┬───────┘ └──────┬───────┘            │
│         │                │                │                     │
│         └────────────────┼────────────────┘                     │
│                          ▼                                      │
│  ┌────────────────────────────────────────────────────────────┐│
│  │                     ReActEngine                             ││
│  │  ┌─────────────┐  ┌──────────────────┐  ┌───────────────┐  ││
│  │  │StreamBuffer │  │ ExecutionContext │  │ ExecutionStats│  ││
│  │  └─────────────┘  └──────────────────┘  └───────────────┘  ││
│  └────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

## 模块结构

```
gecko/core/engine/
├── __init__.py          # 模块导出
├── base.py              # 基础引擎类和统计组件
├── buffer.py            # 流式响应缓冲区
└── react.py             # ReAct 推理引擎实现
```

## 核心组件

### CognitiveEngine (base.py)

抽象基类，定义引擎的核心接口和生命周期。

```python
from gecko.core.engine import CognitiveEngine

class CustomEngine(CognitiveEngine):
    async def step(self, input_messages: List[Message], **kwargs) -> AgentOutput:
        # 实现推理逻辑
        ...
```

**主要特性：**
- 生命周期管理 (`initialize`, `cleanup`)
- 钩子系统 (`before_step_hook`, `after_step_hook`, `on_error_hook`)
- 执行统计 (`ExecutionStats`)
- 成本追踪 (`record_cost`)
- 事件发布 (`EventBus` 集成)

### ExecutionStats (base.py)

线程安全的执行统计收集器。

```python
from gecko.core.engine import ExecutionStats

stats = ExecutionStats()
stats.add_step(duration=1.5, input_tokens=100, output_tokens=50)
stats.add_tool_call(n=2)
stats.add_cost(0.005)

print(stats.to_dict())
# {
#     "total_steps": 1,
#     "total_time": 1.5,
#     "avg_step_time": 1.5,
#     "input_tokens": 100,
#     "output_tokens": 50,
#     "total_tokens": 150,
#     "tool_calls": 2,
#     "errors": 0,
#     "error_rate": 0.0,
#     "estimated_cost": 0.005
# }
```

### StreamBuffer (buffer.py)

处理 LLM 流式响应的缓冲区，支持增量内容和工具调用解析。

```python
from gecko.core.engine.buffer import StreamBuffer

buffer = StreamBuffer(
    max_content_chars=200_000,
    max_argument_chars=100_000,
    max_tool_index=1000
)

for chunk in stream_response:
    text_delta = buffer.add_chunk(chunk)
    if text_delta:
        print(text_delta, end="", flush=True)

message = buffer.build_message()
```

**特性：**
- 增量内容累积
- 工具调用参数流式组装
- 自动 JSON 清理（处理尾随逗号、单引号等）
- 内存保护（可配置上限）

### ReActEngine (react.py)

实现 ReAct 循环的主引擎。

```python
from gecko.core.engine import ReActEngine, ReActConfig

config = ReActConfig(
    max_reflections=2,
    tool_error_threshold=3,
    loop_repeat_threshold=2,
    max_context_chars=100_000
)

engine = ReActEngine(
    model=llm_client,
    toolbox=toolbox,
    memory=memory,
    max_turns=10,
    config=config
)

# 同步调用
output = await engine.step([Message.user("查询天气")])

# 流式调用
async for event in engine.step_stream([Message.user("查询天气")]):
    if event.type == "token":
        print(event.content, end="")
    elif event.type == "tool_output":
        print(f"\n[Tool] {event.data['tool_name']}: {event.content}")

# 结构化输出
class WeatherInfo(BaseModel):
    city: str
    temperature: float
    condition: str

result = await engine.step_structured(
    [Message.user("北京天气如何？")],
    response_model=WeatherInfo
)
```

### ExecutionContext (react.py)

管理对话上下文和执行状态。

```python
from gecko.core.engine.react import ExecutionContext

context = ExecutionContext(
    messages=[Message.system("You are helpful.")],
    max_history=50,
    max_chars=100_000
)

context.add_message(Message.user("Hello"))
context.add_message(Message.assistant("Hi there!"))

# 自动裁剪过长上下文
# 保持工具调用/结果配对完整性
```

## 配置选项

### 模型定价配置

```python
from gecko.core.engine import MODEL_PRICING, get_pricing_for_model

# 查询定价
pricing = get_pricing_for_model("gpt-4-turbo")
# {"input": 10.0, "output": 30.0}  (单位: USD/1M tokens)

# 自定义定价
MODEL_PRICING["custom-model"] = {"input": 1.0, "output": 2.0}
```

**外部配置文件：**

```bash
# 环境变量指定
export GECKO_PRICING_FILE=/path/to/pricing.json

# 或用户配置
# ~/.gecko/pricing.json
```

```json
{
  "custom-model-v1": {"input": 0.5, "output": 1.0},
  "custom-model-v2": {"input": 0.8, "output": 1.5}
}
```

### ReActConfig 参数

| 参数                    | 类型 | 默认值  | 说明                   |
| ----------------------- | ---- | ------- | ---------------------- |
| `max_reflections`       | int  | 2       | 错误反思最大次数       |
| `tool_error_threshold`  | int  | 3       | 触发反思的连续错误阈值 |
| `loop_repeat_threshold` | int  | 2       | 循环检测重复阈值       |
| `max_context_chars`     | int  | 100,000 | 上下文最大字符数       |

### ReActEngine 参数

| 参数                     | 类型               | 默认值   | 说明             |
| ------------------------ | ------------------ | -------- | ---------------- |
| `max_turns`              | int                | 10       | 最大推理轮次     |
| `max_observation_length` | int                | 2000     | 工具输出截断长度 |
| `system_prompt`          | str/PromptTemplate | 内置模板 | 系统提示词       |
| `on_turn_start`          | Callable           | None     | 轮次开始回调     |
| `on_turn_end`            | Callable           | None     | 轮次结束回调     |

## 使用示例

### 基础使用

```python
import asyncio
from gecko.core.engine import ReActEngine, create_engine
from gecko.core.message import Message
from gecko.core.toolbox import ToolBox
from gecko.core.memory import TokenMemory

async def main():
    # 初始化组件
    model = MyLLMClient()
    toolbox = ToolBox()
    memory = TokenMemory(session_id="demo")
    
    # 创建引擎
    engine = ReActEngine(
        model=model,
        toolbox=toolbox,
        memory=memory,
        max_turns=5
    )
    
    # 使用上下文管理器
    async with engine:
        result = await engine.step([
            Message.user("帮我查询北京的天气")
        ])
        print(result.content)
        
        # 查看统计
        print(engine.get_stats_summary())

asyncio.run(main())
```

### 流式输出

```python
async def stream_example(engine, query):
    events = []
    async for event in engine.step_stream([Message.user(query)]):
        events.append(event)
        
        if event.type == "token":
            print(event.content, end="", flush=True)
        elif event.type == "tool_input":
            print(f"\n🔧 调用工具: {event.data['tools']}")
        elif event.type == "tool_output":
            print(f"📤 工具结果: {event.content[:100]}...")
        elif event.type == "error":
            print(f"❌ 错误: {event.content}")
    
    print()  # 换行
```

### 结构化输出

```python
from pydantic import BaseModel, Field
from typing import List

class TaskPlan(BaseModel):
    goal: str = Field(description="任务目标")
    steps: List[str] = Field(description="执行步骤")
    estimated_time: int = Field(description="预计时间(分钟)")

async def structured_example(engine):
    plan = await engine.step_structured(
        [Message.user("制定一个学习 Python 的计划")],
        response_model=TaskPlan,
        max_retries=2
    )
    
    print(f"目标: {plan.goal}")
    for i, step in enumerate(plan.steps, 1):
        print(f"  {i}. {step}")
    print(f"预计时间: {plan.estimated_time} 分钟")
```

### 自定义钩子

```python
async def log_step_start(messages, **kwargs):
    print(f"📝 开始处理 {len(messages)} 条消息")

async def log_step_end(messages, output, **kwargs):
    print(f"✅ 完成，输出 {len(output.content)} 字符")

async def handle_error(error, messages, **kwargs):
    print(f"🚨 发生错误: {error}")
    # 可以发送告警通知

engine.before_step_hook = log_step_start
engine.after_step_hook = log_step_end
engine.on_error_hook = handle_error
engine.hooks_fail_fast = False  # 钩子失败不中断主流程
```

### 事件总线集成

```python
from gecko.core.events import EventBus

event_bus = EventBus()

@event_bus.on("step_started")
async def on_start(data):
    print(f"引擎 {data['engine']} 开始处理")

@event_bus.on("step_completed")
async def on_complete(data):
    print(f"处理完成，工具调用: {data['has_tool_calls']}")

@event_bus.on("step_error")
async def on_error(data):
    print(f"错误: {data['error_type']} - {data['error_message']}")

engine = ReActEngine(
    model=model,
    toolbox=toolbox,
    memory=memory,
    event_bus=event_bus
)
```

## 扩展指南

### 自定义引擎

```python
from gecko.core.engine import CognitiveEngine
from gecko.core.output import AgentOutput

class ChainOfThoughtEngine(CognitiveEngine):
    """实现 Chain-of-Thought 推理的引擎"""
    
    async def step(self, input_messages, **kwargs):
        await self.before_step(input_messages, **kwargs)
        
        start_time = time.time()
        try:
            # 第一步：生成推理链
            reasoning = await self._generate_reasoning(input_messages)
            
            # 第二步：基于推理生成答案
            answer = await self._generate_answer(reasoning)
            
            output = AgentOutput(
                content=answer,
                metadata={"reasoning": reasoning}
            )
            
            await self.after_step(input_messages, output, **kwargs)
            return output
            
        except Exception as e:
            await self.on_error(e, input_messages, **kwargs)
            raise
        finally:
            self.record_step(duration=time.time() - start_time)
    
    async def _generate_reasoning(self, messages):
        # 实现推理逻辑
        ...
    
    async def _generate_answer(self, reasoning):
        # 实现答案生成
        ...
```

### 自定义流式处理

```python
class CustomStreamEngine(CognitiveEngine):
    
    async def step_stream(self, input_messages, **kwargs):
        self.validate_input(input_messages)
        await self.before_step(input_messages, **kwargs)
        
        buffer = StreamBuffer()
        
        try:
            async for chunk in self.model.astream(messages=input_messages):
                text = buffer.add_chunk(chunk)
                if text:
                    yield AgentStreamEvent(type="token", content=text)
            
            message = buffer.build_message()
            output = AgentOutput(content=message.content)
            
            yield AgentStreamEvent(type="result", data={"output": output})
            await self.after_step(input_messages, output, **kwargs)
            
        except Exception as e:
            yield AgentStreamEvent(type="error", content=str(e))
            await self.on_error(e, input_messages, **kwargs)
            raise
```

## 性能优化

### 内存优化

1. **`__slots__` 优化**: `ExecutionStats`, `StreamBuffer`, `ExecutionContext` 使用 `__slots__` 减少内存占用

2. **消息长度缓存**: `ExecutionContext` 缓存消息长度避免重复计算

3. **定价查询缓存**: `get_pricing_for_model()` 使用前缀匹配缓存

### 上下文裁剪

使用 O(n) 算法裁剪上下文：
- 保持系统消息
- 保持工具调用/结果配对完整性
- 优先移除较早的消息

### 资源保护

| 限制           | 默认值  | 配置方式                               |
| -------------- | ------- | -------------------------------------- |
| 内容最大字符   | 200,000 | `StreamBuffer(max_content_chars=...)`  |
| 参数最大字符   | 100,000 | `StreamBuffer(max_argument_chars=...)` |
| 工具索引上限   | 1,000   | `StreamBuffer(max_tool_index=...)`     |
| 上下文字符上限 | 100,000 | `ReActConfig(max_context_chars=...)`   |
| 历史消息上限   | 50      | `ExecutionContext(max_history=...)`    |

## 错误处理

### 自动循环检测

引擎自动检测以下循环模式：
- 连续相同工具调用
- A-B-A 振荡调用

### 错误反思机制

当连续工具调用失败达到阈值时，引擎会：
1. 注入反思消息
2. 重置错误计数
3. 继续执行

超过最大反思次数后停止执行。

### 超时处理

```python
async for event in engine.step_stream(
    messages,
    timeout=30.0  # 30秒超时
):
    ...
```

## API 参考

### 导出

```python
from gecko.core.engine import (
    # 基类
    CognitiveEngine,
    
    # 主引擎
    ReActEngine,
    
    # 配置
    ReActConfig,
    ExecutionContext,
    
    # 统计
    ExecutionStats,
    
    # 缓冲区
    StreamBuffer,
    
    # 工厂
    create_engine,
    
    # 定价
    MODEL_PRICING,
    load_model_pricing,
    get_model_pricing,
    get_pricing_for_model,
    
    # 常量
    DEFAULT_REACT_TEMPLATE,
    STRUCTURE_TOOL_PREFIX,
    MAX_RETRY_DELAY_SECONDS,
)
```

## 依赖

- Python 3.8+
- pydantic >= 2.0
- asyncio (标准库)

## 测试

```bash
# 运行引擎模块测试
rye run pytest tests/core/test_engine_base.py -v
rye run pytest tests/core/test_engine_buffer.py -v
rye run pytest tests/core/test_engine_react.py -v

# 运行全部测试
rye run pytest tests/core/ -v --cov=gecko.core.engine
```