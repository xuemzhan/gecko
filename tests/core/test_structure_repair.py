# tests/core/test_structure_repair.py
"""
结构化输出修复逻辑单元测试

覆盖范围：
1. repair_json_with_llm 的成功路径
2. repair_json_with_llm 的失败路径（类型错误、语法错误）
"""
import pytest
from unittest.mock import AsyncMock, MagicMock
from gecko.core.structure.repair import repair_json_with_llm
from gecko.core.protocols import ModelProtocol

@pytest.mark.asyncio
async def test_repair_json_success():
    """
    验证 LLM 成功修复 JSON 的场景
    输入：格式错误的 JSON 字符串
    输出：LLM 返回修正后的 JSON，函数返回解析后的 dict
    """
    # Mock LLM 模型
    mock_model = MagicMock(spec=ModelProtocol)
    mock_model.acompletion = AsyncMock()
    
    # 模拟 LLM 返回修复后的合法 JSON
    mock_response = MagicMock()
    # 修复了引号和括号
    mock_response.choices[0].message = {"content": '{"fixed": true}'}
    mock_model.acompletion.return_value = mock_response
    
    broken_text = "{ fixed: true " # 错误输入
    
    # 执行修复
    result = await repair_json_with_llm(broken_text, "Parse Error", mock_model)
    
    # 验证结果
    assert result == {"fixed": True}
    
    # 验证 Prompt 包含了原始错误信息，帮助 LLM 定位问题
    call_args = mock_model.acompletion.call_args[1] # 获取 kwargs
    messages = call_args.get("messages", []) or mock_model.acompletion.call_args[0][0]
    prompt_content = messages[0]["content"]
    assert "Parse Error" in prompt_content
    assert broken_text in prompt_content

@pytest.mark.asyncio
async def test_repair_json_failure_returns_list():
    """
    验证 LLM 返回了 List 而非 Dict 时抛出异常
    (repair_json_with_llm 契约要求返回 Dict)
    """
    mock_model = MagicMock(spec=ModelProtocol)
    mock_model.acompletion = AsyncMock()
    
    # 模拟 LLM 返回了一个数组
    mock_response = MagicMock()
    mock_response.choices[0].message = {"content": '[1, 2, 3]'}
    mock_model.acompletion.return_value = mock_response
    
    # 执行修复，预期抛出 ValueError
    # 使用 match 匹配部分错误信息
    with pytest.raises(ValueError, match="Repair failed"):
        await repair_json_with_llm("broken", "error", mock_model)

@pytest.mark.asyncio
async def test_repair_json_failure_invalid_syntax():
    """
    验证 LLM 返回的内容仍然无法解析为 JSON 的情况
    """
    mock_model = MagicMock(spec=ModelProtocol)
    mock_model.acompletion = AsyncMock()
    
    # 模拟 LLM 返回了自然语言而非 JSON
    mock_response = MagicMock()
    mock_response.choices[0].message = {"content": 'Sorry, I cannot fix this.'}
    mock_model.acompletion.return_value = mock_response
    
    # 执行修复，预期抛出 ValueError
    with pytest.raises(ValueError, match="Repair failed"):
        await repair_json_with_llm("broken", "error", mock_model)