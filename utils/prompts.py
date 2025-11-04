# agno/utils/prompts.py

"""
提示工程工具模块

该模块提供了用于生成结构化提示的辅助函数，主要用于指导大语言模型（LLM）
以特定的格式（如 JSON）返回输出。

主要功能：
- get_json_output_prompt: 生成 JSON 格式输出提示
- get_response_model_format_prompt: 生成文本格式输出提示
"""

import json
import logging
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from pydantic import BaseModel

# 常量定义
JSON_START_MARKER = "{"
JSON_END_MARKER = "}"
DEFS_KEY = "$defs"
PROPERTIES_KEY = "properties"
ENUM_KEY = "enum"
TITLE_KEY = "title"
REF_KEY = "$ref"
ALL_OF_KEY = "allOf"
DESCRIPTION_KEY = "description"
ENUM_REF_KEY = "enum_ref"

# 提示模板
JSON_OUTPUT_BASE_PROMPT = "Provide your output as a JSON containing the following fields:"
JSON_FIELDS_TAG_START = "<json_fields>"
JSON_FIELDS_TAG_END = "</json_fields>"
JSON_PROPERTIES_TAG_START = "<json_field_properties>"
JSON_PROPERTIES_TAG_END = "</json_field_properties>"

JSON_FORMAT_INSTRUCTIONS = """
Your response MUST start with `{` and end with `}`.
Your output will be passed to a JSON parser (e.g., json.loads() in Python).
Ensure it contains only valid JSON and nothing else."""

RESPONSE_MODEL_BASE_PROMPT = (
    "Make sure your response is a valid string (NOT JSON) that mentions the following topics:"
)

# 日志配置
logger = logging.getLogger(__name__)


def log_warning(message: str) -> None:
    """记录警告级别日志"""
    logger.warning(message)


# --- JSON 输出提示 ---

def get_json_output_prompt(
    output_schema: Union[str, List, BaseModel, Type[BaseModel]]
) -> str:
    """
    根据输出 schema 生成指导 LLM 输出 JSON 格式的提示
    
    该函数支持多种 schema 格式：
    - 字符串：直接作为字段描述
    - 列表：字段名列表
    - Pydantic 模型：完整的类型定义
    
    Args:
        output_schema: 定义期望输出格式的 schema
        
    Returns:
        完整的 JSON 输出指令字符串
        
    Examples:
        >>> prompt = get_json_output_prompt(["name", "age"])
        >>> "name" in prompt and "age" in prompt
        True
        
        >>> class User(BaseModel):
        ...     name: str
        ...     age: int
        >>> prompt = get_json_output_prompt(User)
        >>> "name" in prompt
        True
    """
    fields_str, properties_str = _format_schema_for_prompt(output_schema)
    
    # 构建基础提示
    if fields_str:
        prompt = _build_json_prompt_with_schema(fields_str, properties_str)
    else:
        prompt = "Provide the output as JSON."
    
    # 添加格式化指令
    prompt += JSON_FORMAT_INSTRUCTIONS
    
    return prompt


def _build_json_prompt_with_schema(
    fields_str: str,
    properties_str: Optional[str] = None
) -> str:
    """
    构建包含 schema 信息的 JSON 提示
    
    Args:
        fields_str: 字段列表字符串
        properties_str: 属性详情字符串（可选）
        
    Returns:
        格式化的提示字符串
    """
    prompt = (
        f"{JSON_OUTPUT_BASE_PROMPT}\n"
        f"{JSON_FIELDS_TAG_START}\n{fields_str}\n{JSON_FIELDS_TAG_END}"
    )
    
    if properties_str:
        prompt += (
            f"\n\nHere are the properties for each field:\n"
            f"{JSON_PROPERTIES_TAG_START}\n{properties_str}\n{JSON_PROPERTIES_TAG_END}"
        )
    
    return prompt


def _format_schema_for_prompt(
    output_schema: Any
) -> Tuple[Optional[str], Optional[str]]:
    """
    根据 schema 类型分派到相应的格式化函数
    
    Args:
        output_schema: 各种类型的 schema
        
    Returns:
        (字段字符串, 属性字符串) 元组
    """
    # 处理字符串 schema
    if isinstance(output_schema, str):
        # 空字符串视为无效 schema
        if not output_schema.strip():
            log_warning("Schema 字符串为空")
            return None, None
        return output_schema, None
    
    # 处理列表 schema
    if isinstance(output_schema, list):
        # 空列表视为无效 schema
        if not output_schema:
            log_warning("Schema 列表为空")
            return None, None
        
        try:
            return json.dumps(output_schema, ensure_ascii=False), None
        except (TypeError, ValueError) as e:
            log_warning(f"无法序列化列表 schema: {e}")
            return str(output_schema), None
    
    # 处理 Pydantic 模型 schema
    if _is_pydantic_model(output_schema):
        return _format_pydantic_schema_for_prompt(output_schema)
    
    log_warning(f"无法为 schema 类型 '{type(output_schema).__name__}' 构建 JSON 提示")
    return None, None


def _is_pydantic_model(obj: Any) -> bool:
    """检查对象是否为 Pydantic 模型或模型类"""
    if isinstance(obj, BaseModel):
        return True
    try:
        if isinstance(obj, type) and issubclass(obj, BaseModel):
            return True
    except TypeError:
        pass
    return False


def _format_pydantic_schema_for_prompt(
    schema_obj: Union[Type[BaseModel], BaseModel]
) -> Tuple[Optional[str], Optional[str]]:
    """
    为 Pydantic 模型生成字段列表和属性详情
    
    Args:
        schema_obj: Pydantic 模型实例或类
        
    Returns:
        (字段列表 JSON, 属性详情 JSON) 元组
    """
    # 获取 JSON Schema
    try:
        json_schema = schema_obj.model_json_schema()
    except Exception as e:
        logger.error(f"从 Pydantic 模型生成 JSON Schema 失败: {e}")
        return None, None
    
    if not json_schema:
        log_warning("生成的 JSON Schema 为空")
        return None, None
    
    # 提取和格式化属性
    properties = _extract_and_format_properties(json_schema)
    
    if not properties:
        return None, None
    
    # 创建字段列表（排除 $defs）
    fields_list = [key for key in properties.keys() if key != DEFS_KEY]
    
    if not fields_list:
        log_warning("没有可用的字段")
        return None, None
    
    try:
        fields_str = json.dumps(fields_list, ensure_ascii=False)
        properties_str = json.dumps(properties, indent=2, ensure_ascii=False)
        return fields_str, properties_str
    except (TypeError, ValueError) as e:
        logger.error(f"序列化 schema 数据失败: {e}")
        return None, None


def _extract_and_format_properties(json_schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    从 Pydantic JSON Schema 中提取、清理和格式化属性定义
    
    Args:
        json_schema: Pydantic 生成的 JSON Schema
        
    Returns:
        格式化后的属性字典
    """
    all_properties: Dict[str, Any] = {}
    defs = json_schema.get(DEFS_KEY, {})
    
    # 处理顶层属性
    if PROPERTIES_KEY in json_schema:
        for name, props in json_schema[PROPERTIES_KEY].items():
            all_properties[name] = _clean_property_dict(props, defs)
    
    # 处理定义（$defs）
    formatted_defs = _format_definitions(defs)
    if formatted_defs:
        all_properties[DEFS_KEY] = formatted_defs
    
    return all_properties


def _format_definitions(defs: Dict[str, Any]) -> Dict[str, Any]:
    """
    格式化 $defs 中的定义（枚举和嵌套模型）
    
    Args:
        defs: $defs 字典
        
    Returns:
        格式化后的定义字典
    """
    formatted_defs: Dict[str, Any] = {}
    
    for def_name, def_props in defs.items():
        # 处理枚举类型
        if ENUM_KEY in def_props:
            formatted_defs[def_name] = {
                "type": "string",
                ENUM_KEY: def_props[ENUM_KEY],
                DESCRIPTION_KEY: def_props.get(DESCRIPTION_KEY, ""),
            }
        # 处理嵌套模型
        elif PROPERTIES_KEY in def_props:
            nested_props = _format_nested_model_properties(def_props, defs)
            if nested_props:
                formatted_defs[def_name] = nested_props
    
    return formatted_defs


def _format_nested_model_properties(
    def_props: Dict[str, Any],
    defs: Dict[str, Any]
) -> Dict[str, Any]:
    """
    格式化嵌套模型的属性
    
    Args:
        def_props: 模型定义属性
        defs: 所有定义的引用
        
    Returns:
        格式化后的属性字典
    """
    formatted = {}
    
    for name, props in def_props[PROPERTIES_KEY].items():
        formatted[name] = _clean_property_dict(props, defs)
    
    return formatted


def _clean_property_dict(props: Dict[str, Any], defs: Dict[str, Any]) -> Dict[str, Any]:
    """
    清理属性字典：移除冗余字段并处理引用
    
    Args:
        props: 原始属性字典
        defs: 定义字典（用于解析引用）
        
    Returns:
        清理后的属性字典
    """
    # 移除 Pydantic 自动生成的 title
    cleaned = {k: v for k, v in props.items() if k != TITLE_KEY}
    
    # 处理枚举引用
    if ALL_OF_KEY in cleaned:
        _process_enum_reference(cleaned, defs)
    
    return cleaned


def _process_enum_reference(cleaned: Dict[str, Any], defs: Dict[str, Any]) -> None:
    """
    处理 allOf 中的枚举引用，原地修改字典
    
    Args:
        cleaned: 要处理的属性字典
        defs: 定义字典
    """
    all_of = cleaned.get(ALL_OF_KEY)
    
    if not isinstance(all_of, list) or not all_of:
        return
    
    ref = all_of[0].get(REF_KEY, "")
    
    if not ref.startswith(f"#{DEFS_KEY}/"):
        return
    
    # 提取枚举名称
    enum_name = ref.split("/")[-1]
    
    # 验证是否为枚举定义
    if defs.get(enum_name, {}).get(ENUM_KEY):
        cleaned[ENUM_REF_KEY] = enum_name
        del cleaned[ALL_OF_KEY]


# --- 文本格式输出提示 ---

def get_response_model_format_prompt(output_schema: Type[BaseModel]) -> str:
    """
    为 Pydantic 模型生成文本格式输出提示
    
    该提示要求 LLM 生成包含特定主题的自然语言字符串（非 JSON）。
    
    Args:
        output_schema: Pydantic 模型类
        
    Returns:
        指导 LLM 生成文本响应的提示
        
    Examples:
        >>> class Report(BaseModel):
        ...     summary: str = "报告摘要"
        ...     findings: str
        >>> prompt = get_response_model_format_prompt(Report)
        >>> "summary" in prompt
        True
    """
    message = RESPONSE_MODEL_BASE_PROMPT
    
    # 遍历模型字段
    for field_name, field_info in output_schema.model_fields.items():
        description = field_info.description or ""
        
        if description:
            message += f"\n- {field_name}: {description}"
        else:
            message += f"\n- {field_name}"
    
    return message


# --- 测试代码 ---

def _test_json_output_prompt_string() -> None:
    """测试字符串 schema"""
    print("\n[1] 测试字符串 schema:")
    
    str_schema = '"name": "string", "age": "integer"'
    prompt = get_json_output_prompt(str_schema)
    
    try:
        assert str_schema in prompt
        assert JSON_START_MARKER in prompt
        assert JSON_END_MARKER in prompt
        print("  ✓ 字符串 schema 处理正确")
    except AssertionError as e:
        print(f"  ✗ 验证失败: {e}")


def _test_json_output_prompt_list() -> None:
    """测试列表 schema"""
    print("\n[2] 测试列表 schema:")
    
    list_schema = ["name", "age", "is_member"]
    prompt = get_json_output_prompt(list_schema)
    
    try:
        for field in list_schema:
            assert field in prompt, f"字段 {field} 未在提示中找到"
        print("  ✓ 列表 schema 处理正确")
        print(f"    字段: {list_schema}")
    except AssertionError as e:
        print(f"  ✗ 验证失败: {e}")


def _test_json_output_prompt_pydantic() -> None:
    """测试 Pydantic 模型 schema"""
    print("\n[3] 测试 Pydantic 模型 schema:")
    
    class UserRole(str, Enum):
        """用户角色"""
        ADMIN = "admin"
        USER = "user"
        GUEST = "guest"
    
    class UserProfile(BaseModel):
        """用户个人资料"""
        name: str
        age: int
        role: UserRole
        hobbies: List[str]
    
    prompt = get_json_output_prompt(UserProfile)
    
    try:
        # 验证字段列表
        assert "name" in prompt
        assert "age" in prompt
        assert "role" in prompt
        assert "hobbies" in prompt
        print("  ✓ 所有字段都在提示中")
        
        # 验证枚举引用
        assert DEFS_KEY in prompt
        assert ENUM_REF_KEY in prompt or ENUM_KEY in prompt
        print("  ✓ 枚举定义正确")
        
        # 验证格式化指令
        assert "json.loads()" in prompt
        print("  ✓ 格式化指令完整")
        
    except AssertionError as e:
        print(f"  ✗ 验证失败: {e}")


def _test_response_model_format() -> None:
    """测试文本格式提示"""
    print("\n[4] 测试文本格式提示:")
    
    class Report(BaseModel):
        """简单报告"""
        summary: str
        key_findings: List[str]
    
    # 设置字段描述
    Report.model_fields["summary"].description = "报告摘要"
    Report.model_fields["key_findings"].description = "关键发现"
    
    prompt = get_response_model_format_prompt(Report)
    
    try:
        assert "summary" in prompt
        assert "key_findings" in prompt
        assert "报告摘要" in prompt
        assert "NOT JSON" in prompt
        print("  ✓ 文本格式提示生成正确")
    except AssertionError as e:
        print(f"  ✗ 验证失败: {e}")


def _test_nested_models() -> None:
    """测试嵌套模型"""
    print("\n[5] 测试嵌套模型:")
    
    class Address(BaseModel):
        """地址信息"""
        street: str
        city: str
    
    class Person(BaseModel):
        """人员信息"""
        name: str
        address: Address
    
    prompt = get_json_output_prompt(Person)
    
    try:
        assert "name" in prompt
        assert "address" in prompt
        assert DEFS_KEY in prompt
        print("  ✓ 嵌套模型处理正确")
    except AssertionError as e:
        print(f"  ✗ 验证失败: {e}")


def _test_edge_cases() -> None:
    """测试边缘情况"""
    print("\n[6] 测试边缘情况:")
    
    passed = 0
    failed = 0
    
    # 测试空列表
    try:
        prompt = get_json_output_prompt([])
        assert "Provide the output as JSON." in prompt
        print("  ✓ 空列表处理正确")
        passed += 1
    except AssertionError as e:
        print(f"  ✗ 空列表测试失败: {e}")
        failed += 1
    except Exception as e:
        print(f"  ✗ 空列表测试异常: {e}")
        failed += 1
    
    # 测试无效类型
    try:
        prompt = get_json_output_prompt(123) # type: ignore
        assert "Provide the output as JSON." in prompt
        print("  ✓ 无效类型处理正确")
        passed += 1
    except AssertionError as e:
        print(f"  ✗ 无效类型测试失败: {e}")
        failed += 1
    except Exception as e:
        print(f"  ✗ 无效类型测试异常: {e}")
        failed += 1
    
    # 测试空字符串
    try:
        prompt = get_json_output_prompt("")
        assert len(prompt) > 0
        assert "Provide the output as JSON." in prompt
        print("  ✓ 空字符串处理正确")
        passed += 1
    except AssertionError as e:
        print(f"  ✗ 空字符串测试失败: {e}")
        failed += 1
    except Exception as e:
        print(f"  ✗ 空字符串测试异常: {e}")
        failed += 1
    
    # 测试 None 值
    try:
        prompt = get_json_output_prompt(None) # type: ignore
        assert "Provide the output as JSON." in prompt
        print("  ✓ None 值处理正确")
        passed += 1
    except AssertionError as e:
        print(f"  ✗ None 值测试失败: {e}")
        failed += 1
    except Exception as e:
        print(f"  ✗ None 值测试异常: {e}")
        failed += 1
    
    # 测试包含空字符串的列表
    try:
        prompt = get_json_output_prompt(["", "valid"])
        assert "valid" in prompt
        print("  ✓ 包含空字符串的列表处理正确")
        passed += 1
    except Exception as e:
        print(f"  ✗ 包含空字符串的列表测试异常: {e}")
        failed += 1
    
    print(f"\n  边缘情况测试: {passed} 通过, {failed} 失败")


def _run_tests() -> None:
    """运行所有测试"""
    print("=" * 60)
    print("运行 agno/utils/prompts.py 测试")
    print("=" * 60)
    
    _test_json_output_prompt_string()
    _test_json_output_prompt_list()
    _test_json_output_prompt_pydantic()
    _test_response_model_format()
    _test_nested_models()
    _test_edge_cases()
    
    print("\n" + "=" * 60)
    print("✓ 测试完成")
    print("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.WARNING,  # 改为 WARNING 避免测试时过多日志
        format='%(levelname)s: %(message)s'
    )
    
    _run_tests()