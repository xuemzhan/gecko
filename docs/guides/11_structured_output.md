# 结构化输出 (Structured Output)

让 LLM 输出稳定的 JSON 数据是构建 Agent 的关键。Gecko 的 `StructureEngine` 提供了多重保障策略。

## 基础用法

定义一个 Pydantic 模型，然后请求 Agent 返回该结构。

```python
from pydantic import BaseModel, Field
from gecko.core.structure import StructureEngine

class UserInfo(BaseModel):
    name: str = Field(description="姓名")
    age: int = Field(description="年龄")
    tags: list[str] = Field(description="标签")

# 模拟 LLM 输出的文本 (可能是 Markdown 包裹的)
llm_output = """
```json
{"name": "Gecko", "age": 1, "tags": ["AI", "Framework"]}
```
"""

# 解析
user = await StructureEngine.parse(llm_output, UserInfo)
print(user.name) # Gecko
```

## 鲁棒性与自动修复

`StructureEngine` 内部会自动尝试以下策略，直到解析成功：

1.  **Tool Call 解析**: 如果 LLM 使用了 Function Calling，优先解析参数。
2.  **Markdown 提取**: 自动提取 ` ```json ... ``` ` 代码块。
3.  **JSON 修复**: 自动处理常见的 JSON 错误，如：
    *   移除尾部多余的逗号 (Trailing commas)
    *   移除注释 (`//...`)
    *   寻找最外层的 `{...}` 括号对
4.  **YAML 回退** (可选): 如果安装了 `PyYAML`，支持解析 YAML 格式的输出。

## Agent 集成

在 Agent 的 `run` 方法中直接指定 `response_model`，Gecko 会自动处理 Prompt 注入（告知模型需要 JSON）和结果解析。

```python
result = await agent.run(
    "分析这段文本的情感：'今天天气真好'", 
    response_model=SentimentAnalysis
)
```