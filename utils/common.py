# agno/utils/common.py

"""
通用辅助函数模块

该模块包含与项目特定业务逻辑无关的、跨领域通用的辅助函数。
主要功能包括：
- 复杂的类型检查 (isinstanceany, is_typed_dict, check_type_compatibility)
- 数据结构转换 (dataclass_to_dict, nested_model_dump)
- 数据校验 (is_empty, validate_typed_dict)
"""

from dataclasses import asdict, is_dataclass
from typing import Any, List, Optional, Set, Type, Union, get_args, get_origin, get_type_hints, Optional

# --- 类型与实例检查 ---

def isinstanceany(obj: Any, class_list: List[Type]) -> bool:
    """
    检查一个对象是否是给定类列表中任何一个类的实例。

    Args:
        obj (Any): 要检查的对象。
        class_list (List[Type]): 包含多个类定义的列表。

    Returns:
        bool: 如果对象是列表中任何一个类的实例，则返回 True，否则返回 False。
    """
    for cls in class_list:
        if isinstance(obj, cls):
            return True
    return False


def is_typed_dict(cls: Type[Any]) -> bool:
    """
    通过检查是否存在特定的 dunder 属性来判断一个类是否是 TypedDict。
    兼容 Python 3.8 到 3.13+，不再依赖 __total__（Python 3.12+ 中已移除）。

    Args:
        cls (Type[Any]): 要检查的类。

    Returns:
        bool: 如果该类是 TypedDict，则返回 True，否则返回 False。
    """
    return (
        hasattr(cls, "__annotations__")
        and hasattr(cls, "__required_keys__")
        and hasattr(cls, "__optional_keys__")
    )


# --- 数据校验与转换 ---

def is_empty(val: Any) -> bool:
    """
    检查一个值是否为 None 或为空（例如，空字符串、空列表、空字典）。

    Args:
        val (Any): 要检查的值。

    Returns:
        bool: 如果值为 None 或其长度为 0，则返回 True。
    """
    if val is None:
        return True
    # 检查是否有 __len__ 属性，并判断其长度
    if hasattr(val, '__len__') and len(val) == 0:
        return True
    return False


def dataclass_to_dict(dataclass_object: Any, exclude: Optional[Set[str]] = None, exclude_none: bool = False) -> dict:
    """
    将一个 dataclass 实例转换为字典，并提供排除字段或排除 None 值的功能。

    Args:
        dataclass_object (Any): dataclass 的实例。
        exclude (Optional[set[str]], optional): 一个包含要从结果字典中排除的字段名的集合。默认为 None。
        exclude_none (bool, optional): 如果为 True，将排除所有值为 None 的字段。默认为 False。

    Returns:
        dict: 转换后的字典。
    """
    if not is_dataclass(dataclass_object):
        raise TypeError("Input must be a dataclass instance.")
        
    # 使用标准库的 asdict 进行初步转换
    final_dict = asdict(dataclass_object) # pyright: ignore[reportArgumentType]

    # 排除指定字段
    if exclude:
        for key in exclude:
            final_dict.pop(key, None)
    
    # 排除值为 None 的字段
    if exclude_none:
        final_dict = {k: v for k, v in final_dict.items() if v is not None}
        
    return final_dict


def nested_model_dump(value: Any) -> Any:
    """
    递归地对 Pydantic 模型、字典或列表进行 model_dump 操作。
    这对于将包含 Pydantic 模型的复杂嵌套数据结构完全转换为纯 Python 对象非常有用。

    Args:
        value (Any): 输入的数据，可以是 Pydantic 模型、字典、列表或其他类型。

    Returns:
        Any: 转换后的数据。
    """
    # 动态导入 Pydantic BaseModel 以避免循环依赖或不必要的硬依赖
    try:
        from pydantic import BaseModel
    except ImportError:
        # 如果没有安装 Pydantic，则无法处理 BaseModel，直接返回值
        BaseModel = None

    if BaseModel and isinstance(value, BaseModel):
        # 如果是 Pydantic 模型，调用 model_dump
        return value.model_dump()
    elif isinstance(value, dict):
        # 如果是字典，递归处理它的值
        return {k: nested_model_dump(v) for k, v in value.items()}
    elif isinstance(value, list):
        # 如果是列表，递归处理它的每个元素
        return [nested_model_dump(item) for item in value]
    
    # 对于所有其他类型，直接返回
    return value


def validate_typed_dict(data: dict, schema_cls: Type[Any]) -> dict:
    """
    根据一个 TypedDict schema 来验证输入字典。

    此函数会检查：
    1. 是否缺少必需字段。
    2. 是否存在 schema 中未定义的意外字段。
    3. 已提供字段的类型是否与 schema 定义兼容。

    Args:
        data (dict): 需要验证的输入字典。
        schema_cls (Type[Any]): 作为 schema 的 TypedDict 类。

    Raises:
        ValueError: 如果验证失败，会抛出带有详细错误信息的 ValueError。

    Returns:
        dict: 如果验证通过，返回原始数据字典。
    """
    if not isinstance(data, dict):
        raise ValueError(f"期望输入为字典类型以匹配 TypedDict {schema_cls.__name__}, 但收到了 {type(data)}")

    if not is_typed_dict(schema_cls):
        raise TypeError(f"{schema_cls.__name__} 不是一个有效的 TypedDict。")

    try:
        type_hints = get_type_hints(schema_cls)
    except Exception as e:
        raise ValueError(f"无法获取 TypedDict {schema_cls.__name__} 的类型提示: {e}")

    required_keys: Set[str] = getattr(schema_cls, "__required_keys__", set())
    optional_keys: Set[str] = getattr(schema_cls, "__optional_keys__", set())
    all_keys = required_keys | optional_keys

    # 检查缺失的必需字段
    missing_required = required_keys - set(data.keys())
    if missing_required:
        raise ValueError(f"在 TypedDict {schema_cls.__name__} 中缺少必需字段: {missing_required}")

    # 检查意外的字段
    unexpected_fields = set(data.keys()) - all_keys
    if unexpected_fields:
        raise ValueError(f"在 TypedDict {schema_cls.__name__} 中发现意外字段: {unexpected_fields}")

    # 对已提供的字段进行基本类型检查
    for field_name, value in data.items():
        if field_name in type_hints:
            expected_type = type_hints[field_name]
            if not check_type_compatibility(value, expected_type):
                raise ValueError(
                    f"字段 '{field_name}' 的期望类型是 {expected_type}, 但收到了类型为 {type(value)} 的值: {value!r}"
                )

    return data


# --- 复杂类型兼容性检查 ---

def check_type_compatibility(value: Any, expected_type: Type) -> bool:
    """
    对值和期望类型进行基本的兼容性检查，支持 Optional, Union, List 等复合类型。

    Args:
        value (Any): 要检查的值。
        expected_type (Type): 期望的类型。

    Returns:
        bool: 如果类型兼容，则返回 True。
    """
    origin = get_origin(expected_type)
    args = get_args(expected_type)

    # 1. 处理 Union 类型 (包括 Optional[T]，即 Union[T, None])
    if origin is Union:
        # 检查值是否与 Union 中的任何一个类型兼容
        return any(check_type_compatibility(value, arg) for arg in args)

    # 2. 处理 List 类型
    if origin in (list, List):
        if not isinstance(value, list):
            return False
        # 如果 List 有泛型参数 (如 List[int])，则检查列表中的所有元素
        if args:
            element_type = args[0]
            # 如果元素类型是 Any，则任何元素都可以
            if element_type is Any:
                return True
            return all(check_type_compatibility(item, element_type) for item in value)
        return True  # 如果是裸露的 List 类型，只要值是列表就通过

    # 3. 处理 Any 类型
    if expected_type is Any:
        return True

    # 4. 对于其他简单类型或自定义类，使用常规的 isinstance 检查
    try:
        return isinstance(value, expected_type)
    except TypeError:
        # 如果 expected_type 不是一个可用于 isinstance 的有效类型（例如裸露的 Union），
        # 认为它是一个无法检查的复杂类型，保守地返回 True。
        # 注意：这种情况在上面对 Union 的处理后应该很少见。
        return True


if __name__ == "__main__":
    # --- 测试代码 ---
    from dataclasses import dataclass
    from typing import TypedDict
    from pydantic import BaseModel

    print("--- 正在运行 agno/utils/common.py 的测试代码 ---")

    # 1. 测试 isinstanceany
    print("\n[1] 测试 isinstanceany:")
    class A: pass
    class B: pass
    class C: pass
    instance_a = A()
    print(f"  A() 是 A 或 B 的实例吗? {isinstanceany(instance_a, [A, B])}")  # 应该为 True
    print(f"  A() 是 B 或 C 的实例吗? {isinstanceany(instance_a, [B, C])}")  # 应该为 False
    print(f"  'hello' 是 int 或 str 的实例吗? {isinstanceany('hello', [int, str])}") # 应该为 True

    # 2. 测试 is_empty
    print("\n[2] 测试 is_empty:")
    test_cases = [None, [], {}, "", "hello", [1], {"a": 1}, 0]
    for case in test_cases:
        print(f"  is_empty({case!r}) -> {is_empty(case)}")

    # 3. 测试 dataclass_to_dict
    print("\n[3] 测试 dataclass_to_dict:")
    @dataclass
    class User:
        id: int
        name: str
        email: Optional[str] = None
        password: str = "******"
    
    user_instance = User(id=1, name="Alice")
    print(f"  原始实例: {user_instance}")
    print(f"  默认转换: {dataclass_to_dict(user_instance)}")
    print(f"  排除 'password': {dataclass_to_dict(user_instance, exclude={'password'})}")
    print(f"  排除 None 值: {dataclass_to_dict(user_instance, exclude_none=True)}")

    # 4. 测试 nested_model_dump
    print("\n[4] 测试 nested_model_dump:")
    class Item(BaseModel):
        name: str
        price: float
    class Order(BaseModel):
        order_id: int
        items: List[Item]
    
    order_instance = Order(order_id=101, items=[Item(name="Laptop", price=1200.0)])
    print(f"  Pydantic 实例: {order_instance}")
    print(f"  嵌套 dump 结果: {nested_model_dump(order_instance)}")
    
    # 5. 测试 is_typed_dict 和 validate_typed_dict
    print("\n[5] 测试 TypedDict 相关函数:")
    class MySchema(TypedDict, total=True):
        user_id: int
        username: str
        tags: Optional[List[str]]

    class NotTypedDict:
        pass

    print(f"  MySchema 是 TypedDict 吗? {is_typed_dict(MySchema)}") # 应该为 True
    print(f"  NotTypedDict 是 TypedDict 吗? {is_typed_dict(NotTypedDict)}") # 应该为 False

    valid_data = {"user_id": 123, "username": "Bob"}
    missing_data = {"user_id": 123}
    extra_data = {"user_id": 123, "username": "Bob", "extra_field": "test"}
    wrong_type_data = {"user_id": "123", "username": "Bob"}

    try:
        validate_typed_dict(valid_data, MySchema)
        print("  有效数据验证通过。")
    except ValueError as e:
        print(f"  有效数据验证失败: {e}")

    try:
        validate_typed_dict(missing_data, MySchema)
    except ValueError as e:
        print(f"  缺少必需字段测试: {e}")

    try:
        validate_typed_dict(extra_data, MySchema)
    except ValueError as e:
        print(f"  包含意外字段测试: {e}")
        
    try:
        validate_typed_dict(wrong_type_data, MySchema)
    except ValueError as e:
        print(f"  类型错误测试: {e}")

    # 6. 测试 check_type_compatibility
    print("\n[6] 测试 check_type_compatibility:")
    print(f"  check(10, int) -> {check_type_compatibility(10, int)}") # True
    print(f"  check('a', str) -> {check_type_compatibility('a', str)}") # True
    print(f"  check(None, Optional[str]) -> {check_type_compatibility(None, Optional[str])}") # pyright: ignore[reportArgumentType] # True
    print(f"  check('a', Optional[str]) -> {check_type_compatibility('a', Optional[str])}") # type: ignore # True
    print(f"  check(10, Union[str, int]) -> {check_type_compatibility(10, Union[str, int])}") # type: ignore # True
    print(f"  check([1, 2], List[int]) -> {check_type_compatibility([1, 2], List[int])}") # True
    print(f"  check([1, 'a'], List[int]) -> {check_type_compatibility([1, 'a'], List[int])}") # False
    print(f"  check([1, 'a'], List[Any]) -> {check_type_compatibility([1, 'a'], List[Any])}") # True
    print(f"  check({{'a': 1}}, Any) -> {check_type_compatibility({'a': 1}, Any)}") # type: ignore # True

    print("\n--- 测试结束 ---")