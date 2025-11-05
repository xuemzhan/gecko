# agno/utils/tool_utils.py

"""
工具、函数调用与钩子工具模块

该模块提供了与 Agent 的工具使用、函数调用和钩子执行相关的所有辅助功能。
它整合了工具调用解析、函数执行流程以及钩子函数的动态参数过滤等核心逻辑。

主要功能:
- 从不同格式（如 OpenAI API 响应、XML 标签）中解析和构建 `FunctionCall` 对象。
- 处理钩子函数，包括规范化和动态过滤传递给钩子的参数。
- 提供用于从字符串中提取或移除工具调用 XML 标签的工具。
"""

import ast
import inspect
import json
import re
from typing import Any, Callable, Dict, List, Optional, Union

# 动态导入以避免循环依赖
try:
    from guardrails.base import BaseGuardrail
    from models.response import ToolExecution
    from tools.function import Function, FunctionCall
except ImportError:
    # 定义桩类以供独立测试
    class BaseGuardrail: pass
    class ToolExecution: pass
    class Function: pass
    class FunctionCall:
        def __init__(self, function): self.function = function
        
# 假设日志模块已按规划重构
# from agno.utils.log import log_error, log_warning
# 在独立运行时，使用标准日志库
import logging
logger = logging.getLogger(__name__)

def log_error(message: str): logger.error(message)
def log_warning(message: str): logger.warning(message)


# --- 函数调用解析与构建 ---

def get_function_call_from_tool_call(
    tool_call: Dict[str, Any], functions: Optional[Dict[str, Function]] = None
) -> Optional[FunctionCall]:
    """
    从一个标准的 LLM 工具调用字典（如 OpenAI 格式）中解析出 `FunctionCall` 对象。

    Args:
        tool_call (Dict[str, Any]): LLM 返回的工具调用字典。
        functions (Optional[Dict[str, Function]]): 可用函数的字典。

    Returns:
        Optional[FunctionCall]: 构建好的 `FunctionCall` 对象，如果解析失败则返回 None。
    """
    if tool_call.get("type") != "function":
        return None
    
    func_data = tool_call.get("function")
    if not func_data:
        return None
        
    name = func_data.get("name")
    arguments = func_data.get("arguments", "{}")
    call_id = tool_call.get("id")

    if name:
        return get_function_call(
            name=name,
            arguments=arguments,
            call_id=call_id,
            functions=functions,
        )
    return None


def get_function_call(
    name: str,
    arguments: str = "{}",
    call_id: Optional[str] = None,
    functions: Optional[Dict[str, Function]] = None,
) -> Optional[FunctionCall]:
    """
    根据函数名称和参数字符串，构建一个可执行的 `FunctionCall` 对象。

    此函数会：
    1. 在提供的 `functions` 字典中查找对应的 `Function` 对象。
    2. 安全地解析 `arguments` 字符串（支持 JSON 和 Python 字面量）。
    3. 清理参数值（例如，将 "true" 字符串转换为布尔值 `True`）。
    4. 处理解析错误，并将错误信息附加到 `FunctionCall` 对象上。

    Args:
        name (str): 要调用的函数名称。
        arguments (str): 包含函数参数的字符串，通常是 JSON 格式。
        call_id (Optional[str]): 函数调用的唯一 ID。
        functions (Optional[Dict[str, Function]]): 可用函数的字典。

    Returns:
        Optional[FunctionCall]: 构建好的 `FunctionCall` 对象，如果函数未找到则返回 None。
    """
    if not functions or name not in functions:
        log_error(f"函数 '{name}' 在可用函数列表中未找到。")
        return None

    function_to_call = functions[name]
    function_call = FunctionCall(function=function_to_call)
    if call_id:
        function_call.call_id = call_id

    try:
        # 尝试使用 json.loads()，如果失败则回退到 ast.literal_eval()
        try:
            _arguments = json.loads(arguments or "{}")
        except (json.JSONDecodeError, TypeError):
            _arguments = ast.literal_eval(arguments or "{}")
    except (ValueError, SyntaxError) as e:
        error_msg = f"无法解码函数参数 '{arguments}': {e}"
        log_error(error_msg)
        function_call.error = error_msg
        return function_call

    if not isinstance(_arguments, dict):
        error_msg = "函数参数不是一个有效的 JSON 对象。"
        log_error(f"{error_msg} 收到的参数: {arguments}")
        function_call.error = error_msg
        return function_call

    # 清理参数值
    function_call.arguments = _clean_arguments(_arguments)
    return function_call


def _clean_arguments(args: Dict[str, Any]) -> Dict[str, Any]:
    """清理参数字典中的值，例如将 'true'/'false'/'none' 字符串转换为 Python 对象。"""
    cleaned_args = {}
    for k, v in args.items():
        if isinstance(v, str):
            v_lower = v.strip().lower()
            if v_lower == "true":
                cleaned_args[k] = True
            elif v_lower == "false":
                cleaned_args[k] = False
            elif v_lower in ("none", "null", ""):
                cleaned_args[k] = None
            else:
                cleaned_args[k] = v # 保留原始字符串
        else:
            cleaned_args[k] = v
    return cleaned_args


# --- 工具调用 XML 标签处理 ---

def extract_xml_tags(text: str, tag: str) -> List[str]:
    """从字符串中提取所有被指定 XML 标签包裹的内容块。"""
    return re.findall(f"<{re.escape(tag)}>(.*?)</{re.escape(tag)}>", text, re.DOTALL)

def remove_xml_tags(text: str, tag: str) -> str:
    """从字符串中移除所有被指定 XML 标签包裹的内容块及其标签。"""
    return re.sub(f"<{re.escape(tag)}>.*?</{re.escape(tag)}>", "", text, flags=re.DOTALL).strip()


# --- 钩子 (Hooks) 处理 ---

def normalize_hooks(
    hooks: Optional[List[Union[Callable[..., Any], BaseGuardrail]]],
    async_mode: bool = False,
) -> List[Callable[..., Any]]:
    """
    将包含普通函数和 `BaseGuardrail` 实例的钩子列表规范化为统一的可调用函数列表。

    Args:
        hooks: 原始钩子列表。
        async_mode: 如果为 True，则从 `BaseGuardrail` 中提取 `async_check` 方法。

    Raises:
        ValueError: 如果在同步模式下使用了异步钩子函数。

    Returns:
        一个只包含可调用函数的列表。
    """
    if not hooks:
        return []

    result_hooks: List[Callable[..., Any]] = []
    for hook in hooks:
        if isinstance(hook, BaseGuardrail):
            result_hooks.append(hook.async_check if async_mode else hook.check) # type: ignore
        elif callable(hook):
            # 检查异步函数是否在同步模式下被错误使用
            if not async_mode and inspect.iscoroutinefunction(hook):
                raise ValueError(
                    f"不能在同步模式 `run()` 中使用异步钩子 '{hook.__name__}'。请改用 `arun()`。"
                )
            result_hooks.append(hook)
    return result_hooks


def filter_hook_args(hook: Callable[..., Any], all_args: Dict[str, Any]) -> Dict[str, Any]:
    """
    过滤参数字典，只保留钩子函数签名中声明接受的参数。
    如果钩子函数包含 `**kwargs`，则传递所有参数。
    """
    try:
        sig = inspect.signature(hook)
        
        # 检查是否存在 **kwargs
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            return all_args

        # 只传递函数声明的参数
        accepted_params = set(sig.parameters.keys())
        return {k: v for k, v in all_args.items() if k in accepted_params}

    except (TypeError, ValueError):
        # 如果无法检查签名（例如，对于某些内置函数），则保守地传递所有参数
        log_warning(f"无法检查钩子 '{getattr(hook, '__name__', 'unknown')}' 的签名，将传递所有参数。")
        return all_args


if __name__ == "__main__":
    # --- 测试代码 ---
    logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')

    print("--- 正在运行 agno/utils/tool_utils.py 的测试代码 ---")
    
    # 1. 测试函数调用解析
    print("\n[1] 测试函数调用解析:")
    
    # 模拟 Function 对象
    class MockFunction:
        def __init__(self, name): self.name = name
    
    functions_db = {"search": MockFunction("search")}
    
    # a. 从 tool_call 字典解析
    print("\n  a. 从 tool_call 字典解析:")
    tool_call_dict = {
        "id": "call_abc",
        "type": "function",
        "function": {"name": "search", "arguments": '{"query": "Agno", "limit": "5"}'}
    }
    func_call_1 = get_function_call_from_tool_call(tool_call_dict, functions_db)
    print(f"    解析结果: name={func_call_1.function.name}, args={func_call_1.arguments}, id={func_call_1.call_id}")
    assert func_call_1.arguments == {"query": "Agno", "limit": "5"} # 注意：此时 limit 还是字符串

    # b. 使用 get_function_call 直接解析并清理参数
    print("\n  b. 使用 get_function_call 并清理参数:")
    args_str = '{"query": "Test", "is_active": "true", "count": null, "extra": "none"}'
    func_call_2 = get_function_call("search", args_str, "call_def", functions_db)
    print(f"    原始参数字符串: {args_str}")
    print(f"    清理后的参数: {func_call_2.arguments}")
    assert func_call_2.arguments == {"query": "Test", "is_active": True, "count": None, "extra": None}
    
    # c. 使用 Python 字面量作为参数
    print("\n  c. 使用 Python 字面量作为参数:")
    literal_args_str = "{'query': 'Python Literal', 'is_active': True}"
    func_call_3 = get_function_call("search", literal_args_str, "call_ghi", functions_db)
    print(f"    原始参数字符串: {literal_args_str}")
    print(f"    解析后的参数: {func_call_3.arguments}")
    assert func_call_3.arguments == {"query": "Python Literal", "is_active": True}

    # 2. 测试 XML 标签处理
    print("\n[2] 测试 XML 标签处理:")
    text_with_tags = "思考... <tool>print('hello')</tool> 然后... <tool>a = 1 + 1</tool> 结束。"
    extracted = extract_xml_tags(text_with_tags, "tool")
    print(f"  提取 'tool' 标签: {extracted}")
    assert extracted == ["print('hello')", "a = 1 + 1"]
    
    removed = remove_xml_tags(text_with_tags, "tool")
    print(f"  移除 'tool' 标签后的文本: '{removed}'")
    assert removed == "思考...  然后...  结束。"

    # 3. 测试钩子处理
    print("\n[3] 测试钩子处理:")
    def sync_hook(content: str): print(f"  [Sync Hook] content: {content}")
    async def async_hook(run_id: str): print(f"  [Async Hook] run_id: {run_id}")
    def hook_with_kwargs(content: str, **kwargs): print(f"  [Kwargs Hook] content: {content}, kwargs: {kwargs}")

    all_args = {"content": "test", "run_id": "123", "extra": "data"}
    
    print("  a. 测试参数过滤:")
    filtered_args_sync = filter_hook_args(sync_hook, all_args)
    print(f"    - sync_hook 接收的参数: {filtered_args_sync}")
    assert filtered_args_sync == {"content": "test"}
    
    filtered_args_kwargs = filter_hook_args(hook_with_kwargs, all_args)
    print(f"    - hook_with_kwargs 接收的参数: {filtered_args_kwargs}")
    assert filtered_args_kwargs == all_args

    print("\n  b. 测试钩子规范化:")
    try:
        normalize_hooks([async_hook])
    except ValueError as e:
        print(f"    - 成功捕获在同步模式下使用异步钩子的错误: {e}")

    print("\n--- 测试结束 ---")