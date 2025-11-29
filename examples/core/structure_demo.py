# examples/structure_demo.py
"""
Gecko 结构化输出示例（基于最新 structure 子模块实现）

演示内容：
1. 将 Pydantic 模型转换为 OpenAI Tool Schema
2. 从纯 JSON 文本解析为模型
3. 从 Markdown 代码块中提取 JSON 并解析
4. 从 LLM Tool Call（function 调用）结构中解析
5. 自动清理“脏 JSON”（注释、尾逗号等）
6. 使用 Pydantic 自带的校验功能验证数据
7. 对比实际数据与 Schema 的差异（缺字段 / 多字段 / 类型不匹配）
8. 使用 YAML 策略（如果环境安装了 PyYAML）
9. 使用自定义 Strategy 插件扩展解析能力
10. 捕获并打印详细的解析错误信息
11. 使用轻量工具函数从文本中提取 JSON 片段
"""

import asyncio
from typing import Any, Dict

from pydantic import BaseModel, Field, RootModel

from gecko.core.structure import (
    StructureEngine,
    StructureParseError,
    extract_json_from_text,
    ExtractionStrategy,
    register_extraction_strategy,
)


# ==========================
# 定义数据模型
# ==========================

class UserProfile(BaseModel):
    name: str = Field(description="用户名")
    age: int = Field(description="年龄", ge=0, le=150)
    email: str = Field(description="邮箱地址")
    interests: list[str] = Field(default_factory=list, description="兴趣爱好")


class SearchQuery(BaseModel):
    query: str = Field(description="搜索关键词")
    max_results: int = Field(default=5, description="最大结果数")
    filters: Dict[str, Any] = Field(default_factory=dict, description="过滤条件")


# RootModel 示例：用于解析“顶层就是数组”的 JSON，如 [{"name":...}, ...]
class UserList(RootModel[list[UserProfile]]):
    """示例：解析用户列表的 RootModel"""


# ==========================
# 示例主逻辑
# ==========================

async def main() -> None:
    print("=== Gecko 结构化输出示例（StructureEngine / Strategy / YAML）===\n")

    # ------------------------------------------------------------------
    # 1. 生成 OpenAI Tool Schema
    # ------------------------------------------------------------------
    print("1. 生成 OpenAI Tool Schema")
    tool_schema = StructureEngine.to_openai_tool(SearchQuery)
    print(f"   工具类型: {tool_schema['type']}")
    print(f"   工具名:   {tool_schema['function']['name']}")
    params = tool_schema["function"]["parameters"]
    print(f"   参数列表: {list(params.get('properties', {}).keys())}\n")

    # ------------------------------------------------------------------
    # 2. 从纯 JSON 文本解析
    # ------------------------------------------------------------------
    print("2. 从纯 JSON 文本解析")
    json_text = """
    {
        "name": "Alice",
        "age": 25,
        "email": "alice@example.com",
        "interests": ["AI", "Python", "Music"]
    }
    """
    try:
        user = await StructureEngine.parse(json_text, UserProfile)
        print(f"   解析成功: {user.name}, {user.age} 岁")
        print(f"   兴趣: {', '.join(user.interests)}\n")
    except StructureParseError as e:
        print(f"   解析失败: {e.get_detailed_error()}\n")

    # ------------------------------------------------------------------
    # 3. 从 Markdown 代码块解析
    # ------------------------------------------------------------------
    print("3. 从 Markdown 代码块解析")
    markdown_text = """
    这是用户信息：

    ```json
    {
        "name": "Bob",
        "age": 30,
        "email": "bob@example.com"
    }
    ```

    以上是提取的数据。
    """
    try:
        user = await StructureEngine.parse(markdown_text, UserProfile)
        print(f"   解析成功: {user.name}, {user.age} 岁\n")
    except StructureParseError as e:
        print(f"   解析失败: {e.get_detailed_error()}\n")

    # ------------------------------------------------------------------
    # 4. 从工具调用（Tool Call）解析
    # ------------------------------------------------------------------
    print("4. 从工具调用解析")
    tool_calls = [
        {
            "id": "call_123",
            "function": {
                "name": "search",
                "arguments": '{"query": "Python tutorials", "max_results": 10}'
            },
        }
    ]
    try:
        query = await StructureEngine.parse(
            content="",  # 有 raw_tool_calls 时，content 会被忽略
            model_class=SearchQuery,
            raw_tool_calls=tool_calls,
        )
        print(f"   查询关键词: {query.query}")
        print(f"   最大结果数: {query.max_results}")
        print(f"   过滤条件:   {query.filters}\n")
    except StructureParseError as e:
        print(f"   解析失败: {e.get_detailed_error()}\n")

    # ------------------------------------------------------------------
    # 5. 处理“脏 JSON”（自动清理）
    # ------------------------------------------------------------------
    print("5. 处理格式问题（自动清理脏 JSON）")
    dirty_json = """
    {
        "name": "Charlie",
        "age": 35,
        "email": "charlie@example.com",  // 这是注释
        "interests": ["reading", "coding",],  // 尾部逗号
    }
    """
    try:
        user = await StructureEngine.parse(
            dirty_json,
            UserProfile,
            auto_fix=True,
        )
        print(f"   清理后解析成功: {user.name}, {user.age} 岁")
        print(f"   兴趣: {user.interests}\n")
    except StructureParseError as e:
        print(f"   解析失败: {e.get_detailed_error()}\n")

    # ------------------------------------------------------------------
    # 6. 使用 Pydantic 自带的校验功能
    # ------------------------------------------------------------------
    print("6. 使用 Pydantic 自带校验（model_validate）")
    data = {
        "name": "David",
        "age": 28,
        "email": "david@example.com",
        "interests": ["reading"],
    }
    try:
        # 推荐使用 Pydantic v2 的 model_validate，而不是自定义 validate 封装
        user = UserProfile.model_validate(data)
        print(f"   校验成功: {user}\n")
    except Exception as e:
        print(f"   校验失败: {e}\n")

    # ------------------------------------------------------------------
    # 7. Schema 差异检查
    # ------------------------------------------------------------------
    print("7. Schema 差异检查")
    incomplete_data = {
        "name": "Eve",
        # 缺少 age 和 email
        "extra_field": "should not be here",
    }
    diff = StructureEngine.get_schema_diff(incomplete_data, UserProfile)
    print(f"   缺少字段: {diff['missing_required']}")
    print(f"   额外字段: {diff['extra_fields']}")
    print(f"   类型不匹配: {diff['type_mismatches']}\n")

    # ------------------------------------------------------------------
    # 8. YAML 策略示例（仅当安装了 PyYAML 时）
    # ------------------------------------------------------------------
    print("8. YAML 策略解析示例（可选）")
    try:
        import yaml  # noqa: F401

        yaml_text = """
        query: "battery life prediction"
        max_results: 3
        filters: [bms, test]
        """
        # 注意：YAML 策略是作为“所有 JSON 策略失败后的兜底”
        #       所以文本中包含 '['，可以绕过 fast-fail，然后交给 yaml.safe_load。
        try:
            yaml_query = await StructureEngine.parse(yaml_text, SearchQuery)
            print("   YAML 解析成功：")
            print(f"     query:       {yaml_query.query}")
            print(f"     max_results: {yaml_query.max_results}")
            print(f"     filters:     {yaml_query.filters}\n")
        except StructureParseError as e:
            print("   YAML 文本解析失败：")
            print(e.get_detailed_error(), "\n")
    except ImportError:
        print("   未安装 PyYAML，跳过 YAML 策略示例。\n")

    # ------------------------------------------------------------------
    # 9. 自定义 Strategy 插件示例
    # ------------------------------------------------------------------
    print("9. 自定义 Strategy 插件示例")

    # 自定义解析规则：
    # 假设 LLM 有时候输出这种格式：
    #   "Some prefix [ignored] QUERY:python max=3"
    # 我们用插件来兜底解析这个格式为 SearchQuery。
    def custom_query_strategy(text: str, model_class: type[SearchQuery]) -> SearchQuery:
        """
        简单示例：从特定模式文本中提取 query 和 max_results。
        注意：这里只是 Demo，真实项目可根据业务自定义更复杂逻辑。
        """
        marker = "QUERY:"
        if marker not in text:
            raise ValueError("custom_query_strategy: not applicable")

        # 截取 QUERY: 之后的部分，例如 "python max=3"
        payload = text.split(marker, 1)[1].strip()
        parts = payload.split()
        query_value = parts[0] if parts else "default"
        max_results = 5

        for p in parts[1:]:
            if p.startswith("max="):
                try:
                    max_results = int(p.split("=", 1)[1])
                except ValueError:
                    pass

        data = {
            "query": query_value,
            "max_results": max_results,
            "filters": {},
        }
        return model_class.model_validate(data)  # 使用 Pydantic 校验

    # 注册插件到全局 Strategy 列表（会在所有 JSON 策略失败后尝试）
    register_extraction_strategy(
        ExtractionStrategy(
            name="custom_query",
            func=custom_query_strategy, # type: ignore
        )
    )

    plugin_text = "LLM output prefix [xxx] QUERY:battery_health max=3"

    try:
        plugin_query = await StructureEngine.parse(plugin_text, SearchQuery)
        print("   自定义策略解析成功：")
        print(f"     query:       {plugin_query.query}")
        print(f"     max_results: {plugin_query.max_results}\n")
    except StructureParseError as e:
        print("   自定义策略解析失败：")
        print(e.get_detailed_error(), "\n")

    # ------------------------------------------------------------------
    # 10. 捕获并输出详细错误信息
    # ------------------------------------------------------------------
    print("10. 详细错误信息")
    invalid_text = "This is not JSON at all, just plain text."
    try:
        await StructureEngine.parse(invalid_text, UserProfile)
    except StructureParseError as e:
        print("   捕获到 StructureParseError：")
        print(e.get_detailed_error(), "\n")

    # ------------------------------------------------------------------
    # 11. 轻量 JSON 提取工具：extract_json_from_text
    # ------------------------------------------------------------------
    print("11. 轻量 JSON 提取工具 extract_json_from_text")
    mixed_text = """
    模型输出：

    这是一段说明文字，下面是 JSON：

    ```json
    {"foo": "bar", "answer": 42}
    ```

    以上。
    """
    extracted = extract_json_from_text(mixed_text)
    if extracted:
        print("   提取到 JSON：", extracted, "\n")
    else:
        print("   未能提取到有效 JSON\n")


if __name__ == "__main__":
    asyncio.run(main())
