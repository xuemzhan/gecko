from __future__ import annotations

from typing import Any, Dict, Optional

from gecko.core.logging import get_logger
from gecko.core.protocols import ModelProtocol
from gecko.core.structure.json_extractor import extract_structured_data

logger = get_logger(__name__)

REPAIR_TEMPLATE = """
The following text was intended to be a valid JSON object but failed parsing:

```text
{broken_text}
```

Parser Error:
{error_msg}

Task:
Please fix the JSON formatting errors (e.g., missing quotes, trailing commas, unescaped characters).
Output ONLY the valid, minified JSON object. Do not add any markdown, explanations, or extra text.
"""

async def repair_json_with_llm(
    broken_text: str,
    error_msg: str,
    model: ModelProtocol,
    max_length: int = 2000
) -> Dict[str, Any]:
    """
    使用 LLM 修复损坏的 JSON 字符串
    
    Args:
        broken_text: 解析失败的原始文本
        error_msg: 解析器报错信息
        model: 用于修复的模型实例
        max_length: 发送给修复模型的最大文本长度 (防止 Context 爆炸)
    
    Returns:
        修复后的字典数据
        
    Raises:
        ValueError: 如果修复后仍然无法解析，或者解析结果不是字典
    """
    # 截断过长文本，保留头部信息作为修复依据
    if len(broken_text) > max_length:
        broken_text = broken_text[:max_length] + "...(truncated)"

    prompt = REPAIR_TEMPLATE.format(
        broken_text=broken_text,
        error_msg=error_msg
    )

    try:
        logger.info("Triggering LLM-based JSON repair...")
        
        # 调用模型 (构造简单的 OpenAI 格式消息)
        # temperature=0.0 对于格式修复至关重要
        response = await model.acompletion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0, 
            json_mode=True   # 如果底层模型支持，开启 JSON 模式能显著提高成功率
        )
        
        fixed_content = response.choices[0].message.get("content", "")
        
        # 尝试从修复后的内容中提取
        # 我们复用 extract_structured_data 的提取逻辑 (正则/括号匹配)
        # 此时传入 dict 作为 model_class 可以绕过 Pydantic 校验，仅做 JSON Load
        fixed_data = extract_structured_data(
            text=fixed_content,
            model_class=dict, # type: ignore 
            auto_fix=True
        )
        
        # [优化] 强类型检查：确保返回的是字典
        if not isinstance(fixed_data, dict):
            raise ValueError(f"Repair returned {type(fixed_data)}, expected dict")
        
        logger.info("JSON repair successful")
        return fixed_data

    except Exception as e:
        logger.warning(f"JSON repair failed: {e}")
        raise ValueError(f"Repair failed: {e}") from e