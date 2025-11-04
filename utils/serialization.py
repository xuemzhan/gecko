# agno/utils/serialization.py

"""
数据序列化与 Schema 工具模块

该模块提供了数据序列化和 JSON Schema 生成的核心功能。
主要职责包括：
- 将 Python 类型提示转换为 JSON Schema 表示
- 支持 Pydantic 模型、dataclasses、Enums 和标准 Python 类型
- 提供自定义 JSON 序列化器，处理 datetime、Enum 等特殊对象
- 包含用于内联和解析复杂 Pydantic Schema 的工具
"""

import dataclasses
import json
import logging
from datetime import date, datetime, time
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union, get_args, get_origin

from pydantic import BaseModel

# 日志配置
logger = logging.getLogger(__name__)

# 类型映射常量
PY_TYPE_TO_JSON_TYPE = {
    "int": "integer",
    "float": "number",
    "str": "string",
    "bool": "boolean",
    "none": "null",
    "nonetype": "null",
    "list": "array",
    "tuple": "array",
    "set": "array",
    "dict": "object",
}

# 序列化支持的日期时间类型
DATETIME_TYPES = (datetime, date, time)


# --- 自定义 JSON 序列化器 ---

def json_serializer(obj: Any) -> Any:
    """
    自定义 JSON 序列化器，处理标准库不支持的对象类型
    
    支持的类型：
    - datetime, date, time: 转换为 ISO 格式字符串
    - Enum: 转换为其值或名称
    - 其他对象: 转换为字符串表示
    
    Args:
        obj: 需要序列化的对象
        
    Returns:
        对象的 JSON 可序列化表示
        
    Examples:
        >>> json.dumps(datetime(2023, 1, 1), default=json_serializer)
        '"2023-01-01T00:00:00"'
    """
    # 处理日期时间类型
    if isinstance(obj, DATETIME_TYPES):
        return obj.isoformat()
    
    # 处理枚举类型
    if isinstance(obj, Enum):
        value = obj.value
        # 如果枚举值是基础类型，直接返回
        if isinstance(value, (str, int, float, bool, type(None))):
            return value
        return obj.name
    
    # 其他类型尝试转换为字符串
    try:
        return str(obj)
    except Exception as e:
        logger.warning(f"无法序列化对象 {type(obj).__name__}: {e}")
        return f"<Unserializable: {type(obj).__name__}>"


# --- JSON Schema 生成 ---

def get_json_schema_for_arg(type_hint: Any) -> Optional[Dict[str, Any]]:
    """
    递归生成 Python 类型提示对应的 JSON Schema
    
    支持的类型：
    - 基础类型: int, str, bool, float, None
    - 容器类型: List, Tuple, Set, Dict
    - 联合类型: Union, Optional
    - 枚举类型: Enum
    - Pydantic 模型: BaseModel
    - 数据类: dataclass
    
    Args:
        type_hint: Python 类型提示
        
    Returns:
        对应的 JSON Schema，如果无法生成则返回 None
        
    Examples:
        >>> get_json_schema_for_arg(int)
        {'type': 'integer'}
        >>> get_json_schema_for_arg(List[str])
        {'type': 'array', 'items': {'type': 'string'}}
    """
    if type_hint is None or type_hint is type(None):
        return {"type": "null"}
    
    origin = get_origin(type_hint)
    args = get_args(type_hint)
    
    # 处理泛型/复合类型
    if origin is not None:
        return _handle_generic_type(origin, args)
    
    # 处理类类型
    if isinstance(type_hint, type):
        schema = _handle_class_type(type_hint)
        if schema is not None:
            return schema
    
    # 处理基础类型
    json_type = _get_json_type(type_hint)
    if json_type:
        return {"type": json_type}
    
    logger.warning(f"无法为类型 '{type_hint}' 生成 JSON Schema")
    return None


def get_json_schema(
    type_hints: Dict[str, Any],
    param_descriptions: Optional[Dict[str, str]] = None,
    strict: bool = False
) -> Dict[str, Any]:
    """
    根据类型提示和参数描述生成完整的 JSON Schema
    
    Args:
        type_hints: 从函数签名获取的类型提示字典
        param_descriptions: 参数的描述文档
        strict: 是否禁止额外属性（additionalProperties: false）
        
    Returns:
        完整的 JSON Schema 对象
        
    Examples:
        >>> from typing import get_type_hints
        >>> def func(name: str, age: int = 0): pass
        >>> hints = get_type_hints(func)
        >>> schema = get_json_schema(hints)
        >>> schema['properties']['name']['type']
        'string'
    """
    json_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {},
    }
    
    if strict:
        json_schema["additionalProperties"] = False
    
    required_fields: List[str] = []
    
    for param_name, type_hint in type_hints.items():
        # 跳过返回值类型
        if param_name == "return":
            continue
        
        # 检查是否为可选参数
        is_optional = _is_optional_type(type_hint)
        if not is_optional:
            required_fields.append(param_name)
        
        # 生成参数的 schema
        arg_schema = get_json_schema_for_arg(type_hint)
        if arg_schema is None:
            continue
        
        # 添加参数描述
        if param_descriptions and param_name in param_descriptions:
            arg_schema["description"] = param_descriptions[param_name]
        
        json_schema["properties"][param_name] = arg_schema
    
    # 只在有必需字段时添加 required 键
    if required_fields:
        json_schema["required"] = required_fields
    
    return json_schema


def inline_pydantic_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    内联 Pydantic Schema 中的 $ref 引用
    
    将所有 $ref 引用替换为实际定义，移除 $defs。
    这对于不支持 JSON Schema 引用的模型提供商很有用。
    
    Args:
        schema: 原始 Pydantic JSON Schema
        
    Returns:
        内联了所有引用的 Schema
        
    Examples:
        >>> schema = {'$ref': '#/$defs/Model', '$defs': {'Model': {'type': 'object'}}}
        >>> inlined = inline_pydantic_schema(schema)
        >>> '$defs' in inlined
        False
    """
    if not isinstance(schema, dict):
        return schema
    
    # 提取并移除定义
    definitions = schema.pop("$defs", {})
    
    def resolve_ref(ref_path: str) -> Dict[str, Any]:
        """解析 $ref 引用"""
        if not ref_path.startswith("#/$defs/"):
            logger.warning(f"无法解析外部引用: {ref_path}")
            return {"type": "object"}
        
        def_name = ref_path.split("/")[-1]
        if def_name not in definitions:
            logger.warning(f"未找到定义: {def_name}")
            return {"type": "object"}
        
        # 递归处理定义内的引用
        return process_schema(definitions[def_name])
    
    def process_schema(sub_schema: Any) -> Any:
        """递归处理 schema 中的引用"""
        if not isinstance(sub_schema, dict):
            return sub_schema
        
        # 如果是引用，解析它
        if "$ref" in sub_schema:
            return resolve_ref(sub_schema["$ref"])
        
        # 递归处理所有字段
        result = {}
        for key, value in sub_schema.items():
            if isinstance(value, dict):
                result[key] = process_schema(value)
            elif isinstance(value, list):
                result[key] = [
                    process_schema(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                result[key] = value
        
        return result
    
    return process_schema(schema)


# --- 内部辅助函数 ---

def _is_union_type(origin: Any) -> bool:
    """检查类型是否为 Union（跨 Python 版本兼容）"""
    if origin is Union:
        return True
    
    # Python 3.10+ 的 types.UnionType
    try:
        import types
        if hasattr(types, "UnionType") and isinstance(origin, type(types.UnionType)):
            return True
    except (ImportError, AttributeError):
        pass
    
    return False


def _is_optional_type(type_hint: Any) -> bool:
    """检查类型是否为 Optional（即 Union[X, None]）"""
    origin = get_origin(type_hint)
    if not _is_union_type(origin):
        return False
    
    args = get_args(type_hint)
    return type(None) in args


def _get_json_type(py_type: Any) -> Optional[str]:
    """将 Python 类型映射到 JSON Schema 类型"""
    type_name = getattr(py_type, "__name__", str(py_type)).lower()
    return PY_TYPE_TO_JSON_TYPE.get(type_name)


def _handle_generic_type(origin: Any, args: Tuple[Any, ...]) -> Optional[Dict[str, Any]]:
    """处理泛型类型（List, Dict, Union 等）"""
    # 处理容器类型
    if origin in (list, tuple, set):
        item_schema = (
            get_json_schema_for_arg(args[0]) if args 
            else {"type": "string"}
        )
        return {"type": "array", "items": item_schema}
    
    # 处理字典类型
    if origin is dict:
        value_schema = (
            get_json_schema_for_arg(args[1]) if len(args) > 1 
            else {}
        )
        schema: Dict[str, Any] = {"type": "object"}
        if value_schema:
            schema["additionalProperties"] = value_schema
        return schema
    
    # 处理 Union 类型
    if _is_union_type(origin):
        return _handle_union_type(args)
    
    return None


def _handle_union_type(args: Tuple[Any, ...]) -> Optional[Dict[str, Any]]:
    """处理 Union 类型"""
    # 过滤掉 NoneType
    non_none_args = [arg for arg in args if arg is not type(None)]
    
    # 生成每个类型的 schema
    schemas = [
        get_json_schema_for_arg(arg) 
        for arg in non_none_args
    ]
    schemas = [s for s in schemas if s is not None]
    
    if not schemas:
        return None
    
    if len(schemas) == 1:
        return schemas[0]
    
    return {"anyOf": schemas}


def _handle_class_type(type_hint: type) -> Optional[Dict[str, Any]]:
    """处理类类型（Enum, BaseModel, dataclass）"""
    try:
        # 处理枚举类型
        if issubclass(type_hint, Enum):
            return {
                "type": "string",
                "enum": [member.value for member in type_hint]
            }
    except TypeError:
        pass
    
    try:
        # 处理 Pydantic 模型
        if issubclass(type_hint, BaseModel):
            schema = type_hint.model_json_schema()
            return inline_pydantic_schema(schema)
    except TypeError:
        pass
    
    # 处理 dataclass
    if dataclasses.is_dataclass(type_hint):
        return _get_schema_for_dataclass(type_hint)
    
    return None


def _get_schema_for_dataclass(dc: type) -> Dict[str, Any]:
    """为 dataclass 生成 JSON Schema"""
    properties: Dict[str, Any] = {}
    required: List[str] = []
    
    for field_name, field_info in dc.__dataclass_fields__.items():
        # 生成字段的 schema
        field_schema = get_json_schema_for_arg(field_info.type)
        if field_schema is None:
            continue
        
        properties[field_name] = field_schema
        
        # 检查是否为必需字段（没有默认值）
        has_default = (
            field_info.default is not dataclasses.MISSING or
            field_info.default_factory is not dataclasses.MISSING
        )
        if not has_default:
            required.append(field_name)
    
    schema: Dict[str, Any] = {
        "type": "object",
        "properties": properties
    }
    
    if required:
        schema["required"] = required
    
    return schema


# --- 测试代码 ---

def _test_json_serializer() -> None:
    """测试 JSON 序列化器"""
    print("\n[1] 测试 json_serializer:")
    
    class Color(Enum):
        RED = "red"
        GREEN = "green"
        BLUE = 1
    
    test_data = {
        "datetime": datetime(2023, 1, 1, 12, 30, 0),
        "date": date(2023, 1, 1),
        "time": time(12, 30, 0),
        "enum_str": Color.RED,
        "enum_int": Color.BLUE,
        "set": {1, 2, 3}
    }
    
    try:
        serialized = json.dumps(test_data, default=json_serializer, indent=2)
        print("  ✓ 序列化成功")
        
        # 验证结果
        deserialized = json.loads(serialized)
        assert deserialized["enum_str"] == "red"
        assert deserialized["enum_int"] == 1
        assert deserialized["datetime"] == "2023-01-01T12:30:00"
        print("  ✓ 枚举和日期时间值验证通过")
        
    except Exception as e:
        print(f"  ✗ 序列化失败: {e}")


def _test_schema_generation() -> None:
    """测试 JSON Schema 生成"""
    print("\n[2] 测试 JSON Schema 生成:")
    
    from typing import get_type_hints
    
    class NestedModel(BaseModel):
        detail: str
        count: int = 0
    
    @dataclasses.dataclass
    class MyDataClass:
        id: int
        items: List[str] = dataclasses.field(default_factory=list)
    
    def my_function(
        name: str,
        age: Optional[int] = 30,
        tags: List[str] = None, # type: ignore
        config: Optional[NestedModel] = None,
        data: Optional[MyDataClass] = None
    ):
        """测试函数"""
        pass
    
    try:
        type_hints = get_type_hints(my_function)
        descriptions = {
            "name": "用户名",
            "age": "用户年龄"
        }
        
        schema = get_json_schema(type_hints, descriptions, strict=True)
        
        print("  ✓ Schema 生成成功")
        
        # 验证关键部分
        assert schema["properties"]["name"]["type"] == "string"
        assert "name" in schema["required"]
        assert "age" not in schema["required"]
        assert schema["properties"]["tags"]["type"] == "array"
        assert schema["additionalProperties"] is False
        
        print("  ✓ Schema 结构验证通过")
        print(f"    必需字段: {schema.get('required', [])}")
        print(f"    属性数量: {len(schema['properties'])}")
        
    except Exception as e:
        print(f"  ✗ Schema 生成失败: {e}")
        import traceback
        traceback.print_exc()


def _test_schema_inlining() -> None:
    """测试 Pydantic Schema 内联"""
    print("\n[3] 测试 Pydantic Schema 内联:")
    
    class Address(BaseModel):
        street: str
        city: str
    
    class Person(BaseModel):
        name: str
        address: Address
    
    try:
        original_schema = Person.model_json_schema()
        has_defs = "$defs" in original_schema
        print(f"  原始 Schema 包含 $defs: {has_defs}")
        
        inlined_schema = inline_pydantic_schema(original_schema)
        print("  ✓ 内联完成")
        
        # 验证
        assert "$defs" not in inlined_schema
        schema_str = json.dumps(inlined_schema)
        assert "$ref" not in schema_str
        
        print("  ✓ 内联验证通过（无 $defs 和 $ref）")
        
        # 验证结构完整性
        assert "properties" in inlined_schema
        assert "name" in inlined_schema["properties"]
        assert "address" in inlined_schema["properties"]
        print("  ✓ Schema 结构完整")
        
    except Exception as e:
        print(f"  ✗ 内联失败: {e}")


def _test_edge_cases() -> None:
    """测试边缘情况"""
    print("\n[4] 测试边缘情况:")
    
    try:
        # 测试 None 类型
        schema = get_json_schema_for_arg(type(None))
        assert schema == {"type": "null"}
        print("  ✓ None 类型处理正确")
        
        # 测试嵌套 Optional
        schema = get_json_schema_for_arg(Optional[Optional[str]])
        assert "string" in str(schema)
        print("  ✓ 嵌套 Optional 处理正确")
        
        # 测试复杂 Union
        schema = get_json_schema_for_arg(Union[int, str, List[bool]])
        assert "anyOf" in schema # type: ignore
        print("  ✓ 复杂 Union 处理正确")
        
        # 测试空字典
        schema = get_json_schema_for_arg(dict)
        assert schema["type"] == "object" # type: ignore
        print("  ✓ 字典类型处理正确")
        
        # 测试 Set 类型
        schema = get_json_schema_for_arg(Set[int])
        assert schema["type"] == "array" # type: ignore
        assert schema["items"]["type"] == "integer" # type: ignore
        print("  ✓ Set 类型处理正确")
        
    except Exception as e:
        print(f"  ✗ 边缘情况测试失败: {e}")


def _run_tests() -> None:
    """运行所有测试"""
    print("=" * 60)
    print("运行 agno/utils/serialization.py 测试")
    print("=" * 60)
    
    _test_json_serializer()
    _test_schema_generation()
    _test_schema_inlining()
    _test_edge_cases()
    
    print("\n" + "=" * 60)
    print("✓ 测试完成")
    print("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )
    
    _run_tests()