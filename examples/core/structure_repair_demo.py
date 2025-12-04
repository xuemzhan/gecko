import asyncio
import os
from pydantic import BaseModel
from gecko.core.structure import StructureEngine, StructureParseError
from gecko.plugins.models import ZhipuChat

class User(BaseModel):
    name: str
    age: int

async def main():
    api_key = os.getenv("ZHIPU_API_KEY")
    if not api_key: return
    
    model = ZhipuChat(api_key=api_key, model="glm-4-flash")

    # 1. 模拟一个极其糟糕的 JSON (LLM 常见错误：Markdown 包裹 + 尾部逗号 + 缺失引号)
    bad_json = """
    Here is the data:
    ```json
    {
        name: "Gecko",  // Missing quotes on key
        "age": 10,      // Trailing comma
    }
    ```
    """
    
    print("🔴 尝试解析错误 JSON...")
    try:
        # 2. 传入 model 参数开启自愈功能
        user = await StructureEngine.parse(
            content=bad_json, 
            model_class=User, 
            model=model  # [v0.4 核心] 传入模型以启用 LLM Repair
        )
        print(f"🟢 自愈成功! Result: {user}")
    except StructureParseError as e:
        print(f"❌ 最终失败: {e}")

if __name__ == "__main__":
    asyncio.run(main())