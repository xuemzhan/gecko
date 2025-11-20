# tests/core/test_structure.py
import pytest
from pydantic import BaseModel
from gecko.core.structure import StructureEngine

class UserInfo(BaseModel):
    name: str
    age: int

@pytest.mark.asyncio
async def test_structure_parsing_strategies():
    # Case 1: Tool Calls (最高优先级)
    tool_calls = [{
        "function": {
            "name": "extract_data", 
            "arguments": '{"name": "Alice", "age": 30}'
        }
    }]
    res1 = await StructureEngine.parse("some text", UserInfo, raw_tool_calls=tool_calls)
    assert res1.name == "Alice"
    
    # Case 2: Pure JSON
    res2 = await StructureEngine.parse('{"name": "Bob", "age": 25}', UserInfo)
    assert res2.name == "Bob"
    
    # Case 3: Markdown JSON (带 ```json)
    md_text = """Here is the data:
    ```json
    {
        "name": "Charlie",
        "age": 40
    }
    ```
    """
    res3 = await StructureEngine.parse(md_text, UserInfo)
    assert res3.name == "Charlie"

    # Case 4: Markdown (不带 json 标记)
    md_text_raw = """
    ```
    {"name": "Dave", "age": 20}
    ```
    """
    res4 = await StructureEngine.parse(md_text_raw, UserInfo)
    assert res4.name == "Dave"

    # Case 5: 暴力查找 (Brute Force)
    messy_text = "Sure! output is { \"name\": \"Eve\", \"age\": 18 } thanks."
    res5 = await StructureEngine.parse(messy_text, UserInfo)
    assert res5.name == "Eve"

    # Case 6: 失败场景
    with pytest.raises(ValueError) as exc:
        await StructureEngine.parse("No json here", UserInfo)
    assert "No valid JSON found" in str(exc.value)