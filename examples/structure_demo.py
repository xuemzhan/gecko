# examples/structure_demo.py
import asyncio
from pydantic import BaseModel, Field
from gecko.core.structure import StructureEngine, StructureParseError


# 定义数据模型
class UserProfile(BaseModel):
    name: str = Field(description="用户名")
    age: int = Field(description="年龄", ge=0, le=150)
    email: str = Field(description="邮箱地址")
    interests: list[str] = Field(default_factory=list, description="兴趣爱好")


class SearchQuery(BaseModel):
    query: str = Field(description="搜索关键词")
    max_results: int = Field(default=5, description="最大结果数")
    filters: dict = Field(default_factory=dict, description="过滤条件")


async def main():
    print("=== Gecko 结构化输出示例 ===\n")
    
    # 1. 生成 OpenAI Tool Schema
    print("1. 生成 OpenAI Tool Schema")
    schema = StructureEngine.to_openai_tool(SearchQuery)
    print(f"   工具名: {schema['function']['name']}")
    print(f"   参数: {list(schema['function']['parameters']['properties'].keys())}\n")
    
    # 2. 从 JSON 文本解析
    print("2. 从纯 JSON 文本解析")
    json_text = '''
    {
        "name": "Alice",
        "age": 25,
        "email": "alice@example.com",
        "interests": ["AI", "Python", "Music"]
    }
    '''
    
    try:
        user = await StructureEngine.parse(json_text, UserProfile)
        print(f"   解析成功: {user.name}, {user.age} 岁")
        print(f"   兴趣: {', '.join(user.interests)}\n")
    except StructureParseError as e:
        print(f"   解析失败: {e}\n")
    
    # 3. 从 Markdown 代码块解析
    print("3. 从 Markdown 代码块解析")
    markdown_text = '''
    这是用户信息：
    
    ```json
    {
        "name": "Bob",
        "age": 30,
        "email": "bob@example.com"
    }
    ```
    
    以上是提取的数据。
    '''
    
    try:
        user = await StructureEngine.parse(markdown_text, UserProfile)
        print(f"   解析成功: {user.name}\n")
    except StructureParseError as e:
        print(f"   解析失败: {e}\n")
    
    # 4. 从工具调用解析
    print("4. 从工具调用解析")
    tool_calls = [
        {
            "id": "call_123",
            "function": {
                "name": "search",
                "arguments": '{"query": "Python tutorials", "max_results": 10}'
            }
        }
    ]
    
    try:
        query = await StructureEngine.parse(
            content="",
            model_class=SearchQuery,
            raw_tool_calls=tool_calls
        )
        print(f"   查询: {query.query}")
        print(f"   最大结果数: {query.max_results}\n")
    except StructureParseError as e:
        print(f"   解析失败: {e}\n")
    
    # 5. 处理格式问题（自动清理）
    print("5. 处理格式问题（自动清理）")
    dirty_json = '''
    {
        "name": "Charlie",
        "age": 35,
        "email": "charlie@example.com",  // 这是注释
        "interests": ["reading", "coding",],  // 尾部逗号
    }
    '''
    
    try:
        user = await StructureEngine.parse(
            dirty_json,
            UserProfile,
            auto_fix=True
        )
        print(f"   清理后解析成功: {user.name}\n")
    except StructureParseError as e:
        print(f"   解析失败: {e.get_detailed_error()}\n")
    
    # 6. 验证数据
    print("6. 验证数据")
    data = {
        "name": "David",
        "age": 28,
        "email": "david@example.com"
    }
    
    try:
        user = StructureEngine.validate(data, UserProfile)
        print(f"   验证成功: {user}\n")
    except Exception as e:
        print(f"   验证失败: {e}\n")
    
    # 7. Schema 差异检查
    print("7. Schema 差异检查")
    incomplete_data = {
        "name": "Eve",
        # 缺少 age 和 email
        "extra_field": "should not be here"
    }
    
    diff = StructureEngine.get_schema_diff(incomplete_data, UserProfile)
    print(f"   缺少字段: {diff['missing_required']}")
    print(f"   额外字段: {diff['extra_fields']}\n")
    
    # 8. 详细错误信息
    print("8. 详细错误信息")
    invalid_text = "This is not JSON at all, just plain text."
    
    try:
        user = await StructureEngine.parse(invalid_text, UserProfile)
    except StructureParseError as e:
        print("   捕获到详细错误:")
        print(e.get_detailed_error())


if __name__ == "__main__":
    asyncio.run(main())