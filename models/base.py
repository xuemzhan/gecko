import asyncio  
import collections.abc  
import json  
from abc import ABC, abstractmethod  
from dataclasses import dataclass, field  
from hashlib import md5  
from pathlib import Path  
from time import time  
from types import AsyncGeneratorType, GeneratorType  
from typing import (  
    Any,  
    AsyncIterator,  
    Dict,  
    Iterator,  
    List,  
    Literal,  
    Optional,  
    Tuple,  
    Type,  
    Union,  
    get_args,  
)  
from uuid import uuid4  
  
from pydantic import BaseModel  
  
from exceptions import AgentRunException  
from media import Audio, File, Image, Video  
from models.message import Citations, Message  
from models.metrics import Metrics  
from models.response import ModelResponse, ModelResponseEvent, ToolExecution  
from run.agent import CustomEvent, RunContentEvent, RunOutput, RunOutputEvent  
from run.team import RunContentEvent as TeamRunContentEvent  
from run.team import TeamRunOutputEvent  
from run.workflow import WorkflowCompletedEvent, WorkflowRunOutputEvent  
from tools.function import Function, FunctionCall, FunctionExecutionResult, ToolResult, UserInputField  
from utils.log import log_debug, log_error, log_info, log_warning  
from utils.time_utils import Timer  
from utils.tool_utils import get_function_call_for_tool_call, get_function_call_for_tool_execution  
  
# ================= 安全类型映射 (替代 eval) =================  
ALLOWED_TYPE_MAPPING = {  
    "str": str,  
    "int": int,  
    "float": float,  
    "bool": bool,  
    "list": list,  
    "dict": dict,  
    "Any": Any,  
}  
  
def safe_type_resolver(type_str: str) -> Type:  
    """安全解析类型字符串，避免使用 eval"""  
    if not isinstance(type_str, str):  
        return type_str  
      
    # 处理 typing 模块类型 (如 List[str])  
    if type_str.startswith(("List[", "Dict[", "Union[", "Optional[")):  
        try:  
            from typing import _eval_type  # type: ignore  
            globals_dict = {**globals(), **ALLOWED_TYPE_MAPPING}  
            return _eval_type(eval(type_str, globals_dict), globals_dict, None)  # type: ignore  
        except Exception:  
            log_warning(f"无法解析复杂类型 {type_str}，回退到 str")  
            return str  
      
    return ALLOWED_TYPE_MAPPING.get(type_str.lower(), str)  
  
# ================= 缓存抽象层 =================  
class CacheBackend(ABC):  
    """缓存后端抽象接口"""  
    @abstractmethod  
    def get(self, key: str) -> Optional[Dict[str, Any]]:  
        pass  
      
    @abstractmethod  
    def set(self, key: str, value: Dict[str, Any], ttl: Optional[int] = None):  
        pass  
  
class FileSystemCache(CacheBackend):  
    """基于文件系统的缓存实现"""  
    def __init__(self, cache_dir: Optional[Union[str, Path]] = None):  
        self.cache_dir = Path(cache_dir or Path.home() / ".agno" / "cache" / "model_responses")  
        self.cache_dir.mkdir(parents=True, exist_ok=True)  
      
    def _get_cache_file(self, key: str) -> Path:  
        return self.cache_dir / f"{key}.json"  
      
    def get(self, key: str) -> Optional[Dict[str, Any]]:  
        cache_file = self._get_cache_file(key)  
        if not cache_file.exists():  
            return None  
          
        try:  
            with open(cache_file, "r", encoding="utf-8") as f:  
                cached_data = json.load(f)  
              
            # 检查TTL  
            if "timestamp" in cached_data and cached_data.get("ttl"):  
                if time() - cached_data["timestamp"] > cached_data["ttl"]:  
                    cache_file.unlink()  
                    return None  
              
            return cached_data  
        except (OSError, json.JSONDecodeError) as e:  
            log_error(f"缓存读取失败 {key}: {e}")  
            return None  
      
    def set(self, key: str, value: Dict[str, Any], ttl: Optional[int] = None):  
        try:  
            cache_file = self._get_cache_file(key)  
            value["timestamp"] = int(time())  
            if ttl is not None:  
                value["ttl"] = ttl  
              
            with open(cache_file, "w", encoding="utf-8") as f:  
                json.dump(value, f, ensure_ascii=False)  
        except (OSError, TypeError, json.JSONEncodeError) as e:  
            log_error(f"缓存写入失败 {key}: {e}")  
  
# ================= 事件总线 =================  
class EventBus:  
    """事件总线，支持订阅/发布模式"""  
    def __init__(self):  
        self._subscribers: Dict[str, List[callable]] = {}  
      
    def subscribe(self, event_type: str, callback: callable):  
        self._subscribers.setdefault(event_type, []).append(callback)  
      
    async def publish(self, event_type: str, event: Any):  
        if callbacks := self._subscribers.get(event_type):  
            for cb in callbacks:  
                if asyncio.iscoroutinefunction(cb):  
                    await cb(event)  
                else:  
                    cb(event)  
  
# ================= 工具执行器 =================  
class ToolExecutor:  
    """专门负责工具调用的执行逻辑（同步/异步）"""  
      
    def __init__(self, model, event_bus: Optional[EventBus] = None):  
        self.model = model  
        self.event_bus = event_bus or EventBus()  
      
    def _create_tool_execution_event(self, function_call: FunctionCall, status: str, **kwargs) -> ModelResponse:  
        """创建工具执行事件"""  
        return ModelResponse(  
            content=f"{function_call.get_call_str()} {status}",  
            tool_executions=[ToolExecution(  
                tool_call_id=function_call.call_id,  
                tool_name=function_call.function.name,  
                tool_args=function_call.arguments,  
                **kwargs  
            )],  
            event=ModelResponseEvent.tool_call_completed.value if status == "completed"   
                  else ModelResponseEvent.tool_call_paused.value  
        )  
  
    def run_function_call(self, function_call: FunctionCall) -> Iterator[ModelResponse]:  
        """同步执行单个函数调用并 yield 事件"""  
        # 发布工具开始事件  
        start_event = ModelResponse(  
            content=function_call.get_call_str(),  
            tool_executions=[ToolExecution(  
                tool_call_id=function_call.call_id,  
                tool_name=function_call.function.name,  
                tool_args=function_call.arguments,  
            )],  
            event=ModelResponseEvent.tool_call_started.value,  
        )  
        yield from self._publish_event(start_event)  
          
        # 执行工具  
        timer = Timer()  
        timer.start()  
        try:  
            result = function_call.execute()  
            success = result.status == "success"  
        except Exception as e:  
            log_error(f"函数执行错误: {function_call.function.name} - {e}")  
            result = FunctionExecutionResult(status="failure", error=str(e))  
            success = False  
        finally:  
            timer.stop()  
          
        # 处理结果  
        output = self._extract_output_from_result(function_call, result)  
        result_msg = self._create_result_message(function_call, success, output, timer, result)  
          
        # 发布完成事件  
        complete_event = self._create_tool_execution_event(  
            function_call, "completed",  
            result=output,  
            metrics=result_msg.metrics,  
            stop_after_tool_call=result_msg.stop_after_tool_call  
        )  
        complete_event.updated_session_state = result.updated_session_state  
        complete_event.images = result.images  
        complete_event.videos = result.videos  
        complete_event.audios = result.audios  
        complete_event.files = result.files  
          
        yield from self._publish_event(complete_event)  
        yield result_msg  
  
    async def arun_function_calls(self, function_calls: List[FunctionCall]) -> AsyncIterator[ModelResponse]:  
        """异步执行函数调用链（并发+流式事件）"""  
        tasks = [asyncio.create_task(self._arun_single(fc)) for fc in function_calls]  
          
        for future in asyncio.as_completed(tasks):  
            try:  
                event = await future  
                yield from self._publish_event(event)  
            except Exception as e:  
                log_error(f"异步工具执行异常: {e}")  
                yield ModelResponse(content=f"工具执行失败: {e}", event=ModelResponseEvent.error.value)  
  
    async def _arun_single(self, function_call: FunctionCall) -> ModelResponse:  
        """异步执行单个工具调用"""  
        timer = Timer()  
        timer.start()  
        try:  
            if self._is_async_function(function_call):  
                result = await function_call.aexecute()  
            else:  
                result = await asyncio.to_thread(function_call.execute)  
            success = result.status == "success"  
        except Exception as e:  
            log_error(f"异步函数执行错误: {function_call.function.name} - {e}")  
            result = FunctionExecutionResult(status="failure", error=str(e))  
            success = False  
        finally:  
            timer.stop()  
          
        output = self._extract_output_from_result(function_call, result)  
        result_msg = self._create_result_message(function_call, success, output, timer, result)  
          
        return self._create_tool_execution_event(  
            function_call, "completed",  
            result=output,  
            metrics=result_msg.metrics,  
            stop_after_tool_call=result_msg.stop_after_tool_call,  
            updated_session_state=result.updated_session_state,  
            images=result.images,  
            videos=result.videos,  
            audios=result.audios,  
            files=result.files  
        )  
  
    def _is_async_function(self, function_call: FunctionCall) -> bool:  
        """检查是否为异步函数"""  
        from inspect import iscoroutinefunction, isasyncgenfunction  
        func = function_call.function  
        return (  
            iscoroutinefunction(func.entrypoint) or   
            isasyncgenfunction(func.entrypoint) or  
            (func.tool_hooks and any(iscoroutinefunction(h) for h in func.tool_hooks))  
        )  
  
    def _extract_output_from_result(self, function_call: FunctionCall, exec_result: FunctionExecutionResult) -> str:  
        """从执行结果中提取输出文本"""  
        if isinstance(exec_result.result, ToolResult):  
            tool_result = exec_result.result  
            # 转移媒体资源  
            if tool_result.images: exec_result.images = tool_result.images  
            if tool_result.videos: exec_result.videos = tool_result.videos  
            if tool_result.audios: exec_result.audios = tool_result.audios  
            if tool_result.files: exec_result.files = tool_result.files  
            return tool_result.content or ""  
          
        if isinstance(function_call.result, (GeneratorType, collections.abc.Iterator)):  
            return "".join(str(item) for item in function_call.result)  
          
        return str(function_call.result) if function_call.result is not None else ""  
  
    def _create_result_message(self, function_call: FunctionCall, success: bool, output: str,   
                              timer: Timer, exec_result: FunctionExecutionResult) -> Message:  
        """创建工具调用结果消息"""  
        return Message(  
            role=self.model.tool_message_role,  
            content=output if success else function_call.error,  
            tool_call_id=function_call.call_id,  
            tool_name=function_call.function.name,  
            tool_args=function_call.arguments,  
            tool_call_error=not success,  
            stop_after_tool_call=function_call.function.stop_after_tool_call,  
            metrics=Metrics(duration=timer.elapsed),  
            images=exec_result.images,  
            videos=exec_result.videos,  
            audio=exec_result.audios,  
            files=exec_result.files,  
        )  
      
    async def _publish_event(self, event: ModelResponse) -> AsyncIterator[ModelResponse]:  
        """发布事件到总线并yield"""  
        await self.event_bus.publish(event.event, event)  
        yield event  
  
# ================= 工具调用编排器 =================  
class ToolCallingOrchestrator:  
    """管理工具调用循环和状态流转"""  
      
    def __init__(self, model, tool_executor: ToolExecutor):  
        self.model = model  
        self.executor = tool_executor  
      
    def _check_paused_conditions(self, function_call: FunctionCall) -> List[ToolExecution]:  
        """检查需要暂停的工具调用条件"""  
        paused = []  
        func = function_call.function  
          
        if func.requires_confirmation:  
            paused.append(ToolExecution(  
                tool_call_id=function_call.call_id,  
                tool_name=func.name,  
                tool_args=function_call.arguments,  
                requires_confirmation=True,  
            ))  
          
        if func.requires_user_input:  
            user_input_schema = func.user_input_schema or []  
            # 安全解析用户输入字段类型  
            for field in user_input_schema:  
                if isinstance(field.field_type, str):  
                    field.field_type = safe_type_resolver(field.field_type)  
              
            paused.append(ToolExecution(  
                tool_call_id=function_call.call_id,  
                tool_name=func.name,  
                tool_args=function_call.arguments,  
                requires_user_input=True,  
                user_input_schema=user_input_schema,  
            ))  
          
        # 处理 get_user_input 特殊情况  
        if func.name == "get_user_input" and function_call.arguments.get("user_input_fields"):  
            user_input_schema = []  
            for input_field in function_call.arguments["user_input_fields"]:  
                user_input_schema.append(UserInputField(  
                    name=input_field.get("field_name"),  
                    field_type=safe_type_resolver(input_field.get("field_type", "str")),  
                    description=input_field.get("field_description"),  
                ))  
            paused.append(ToolExecution(  
                tool_call_id=function_call.call_id,  
                tool_name=func.name,  
                tool_args=function_call.arguments,  
                requires_user_input=True,  
                user_input_schema=user_input_schema,  
            ))  
          
        if func.external_execution:  
            paused.append(ToolExecution(  
                tool_call_id=function_call.call_id,  
                tool_name=func.name,  
                tool_args=function_call.arguments,  
                external_execution_required=True,  
            ))  
          
        return paused  
  
    def run(  
        self,  
        messages: List[Message],  
        assistant_message: Message,  
        model_response: ModelResponse,  
        functions: Dict[str, Function],  
        func_call_limit: Optional[int],  
        send_media_to_model: bool,  
        is_async: bool = False,  
    ) -> bool:  
        """执行工具调用循环，返回是否继续"""  
        if not assistant_message.tool_calls:  
            return False  
          
        func_calls = self.model.get_function_calls_to_run(assistant_message, messages, functions)  
        func_results: List[Message] = []  
        call_count = 0  
          
        for fc in func_calls:  
            if func_call_limit and call_count >= func_call_limit:  
                func_results.append(self.model.create_tool_call_limit_error_result(fc))  
                continue  
              
            # 检查暂停条件  
            paused_executions = self._check_paused_conditions(fc)  
            if paused_executions:  
                model_response.tool_executions = paused_executions  
                model_response.event = ModelResponseEvent.tool_call_paused.value  
                return False  # 遇到暂停条件即停止  
              
            # 执行工具  
            if is_async:  
                # 异步执行将在事件循环中处理  
                async for event in self.executor.arun_function_calls([fc]):  
                    model_response.merge(event)  
                    func_results.append(event)  
            else:  
                for event in self.executor.run_function_call(fc):  
                    if isinstance(event, ModelResponse):  
                        model_response.merge(event)  
                        func_results.append(event)  
              
            call_count += 1  
          
        # 处理媒体和格式化  
        self.model.format_function_call_results(messages, func_results, **(model_response.extra or {}))  
        if any(m.images or m.videos or m.audio or m.files for m in func_results):  
            self.model._handle_function_call_media(messages, func_results, send_media_to_model)  
          
        # 检查停止条件  
        stop_conditions = [  
            any(m.stop_after_tool_call for m in func_results),  
            any(tc.requires_confirmation for tc in model_response.tool_executions or []),  
            any(tc.external_execution_required for tc in model_response.tool_executions or []),  
            any(tc.requires_user_input for tc in model_response.tool_executions or []),  
        ]  
        return not any(stop_conditions)  
  
# ================= 重构后的 Model 类 =================  
@dataclass  
class Model(ABC):  
    # 基础字段  
    id: str  
    name: Optional[str] = None  
    provider: Optional[str] = None  
    supports_native_structured_outputs: bool = False  
    supports_json_schema_outputs: bool = False  
    _tool_choice: Optional[Union[str, Dict[str, Any]]] = None  
    system_prompt: Optional[str] = None  
    instructions: Optional[List[str]] = None  
    tool_message_role: str = "tool"  
    assistant_message_role: str = "assistant"  
    cache_response: bool = False  
    cache_ttl: Optional[int] = None  
    cache_dir: Optional[str] = None  
    cache_backend: Optional[CacheBackend] = None  
    event_bus: EventBus = field(default_factory=EventBus)  
  
    def __post_init__(self):  
        if self.provider is None and self.name is not None:  
            self.provider = f"{self.name} ({self.id})"  
          
        # 初始化工具执行器和编排器  
        self._tool_executor = ToolExecutor(self, self.event_bus)  
        self._orchestrator = ToolCallingOrchestrator(self, self._tool_executor)  
          
        # 初始化缓存后端  
        if self.cache_response and not self.cache_backend:  
            self.cache_backend = FileSystemCache(self.cache_dir)  
  
    # --- 缓存相关 ---  
    def _get_model_cache_key(self, messages: List[Message], stream: bool, **kwargs) -> str:  
        """完整缓存键：包含所有影响输出的参数"""  
        message_data = [{"role": m.role, "content": m.content} for m in messages]  
        cache_data = {  
            "model_id": self.id,  
            "messages": message_data,  
            "has_tools": bool(kwargs.get("tools")),  
            "response_format": kwargs.get("response_format"),  
            "tool_choice": kwargs.get("tool_choice") or self._tool_choice,  
            "tool_call_limit": kwargs.get("tool_call_limit"),  
            "stream": stream,  
        }  
        return md5(json.dumps(cache_data, sort_keys=True, default=str).encode()).hexdigest()  
  
    def _get_cached_model_response(self, cache_key: str) -> Optional[Dict[str, Any]]:  
        """从缓存后端获取响应"""  
        if not self.cache_backend:  
            return None  
        return self.cache_backend.get(cache_key)  
  
    def _save_model_response_to_cache(self, cache_key: str, result: ModelResponse, is_streaming: bool = False):  
        """保存响应到缓存后端"""  
        if not self.cache_backend:  
            return  
          
        cache_data = {  
            "timestamp": int(time()),  
            "is_streaming": is_streaming,  
            "result": result.to_dict(),  
        }  
        if self.cache_ttl:  
            cache_data["ttl"] = self.cache_ttl  
          
        self.cache_backend.set(cache_key, cache_data)  
  
    # --- 媒体处理修复 ---  
    def _populate_assistant_message(self, assistant_message: Message, provider_response: ModelResponse) -> Message:  
        """保留所有媒体文件"""  
        if provider_response.role:  
            assistant_message.role = provider_response.role  
        if provider_response.content:  
            assistant_message.content = provider_response.content  
        if provider_response.tool_calls:  
            assistant_message.tool_calls = provider_response.tool_calls  
        # 直接赋值整个媒体列表  
        assistant_message.images = provider_response.images  
        assistant_message.videos = provider_response.videos  
        assistant_message.audios = provider_response.audios  
        assistant_message.files = provider_response.files  
        # ... 其他字段处理 ...  
        return assistant_message  
  
    # --- 核心响应方法 ---  
    def response(self, messages: List[Message], **kwargs) -> ModelResponse:  
        """同步响应主流程"""  
        # 1. 缓存检查  
        cache_key = None  
        if self.cache_response:  
            cache_key = self._get_model_cache_key(messages, stream=False, **kwargs)  
            if cached := self._get_cached_model_response(cache_key):  
                return ModelResponse.from_dict(cached["result"])  
          
        # 2. 初始化  
        model_response = ModelResponse()  
        tool_dicts = self._format_tools(kwargs.get("tools"))  
        functions = {t.name: t for t in kwargs.get("tools", []) if isinstance(t, Function)}  
          
        # 3. 主循环  
        while True:  
            assistant_msg = Message(role=self.assistant_message_role)  
            self._process_model_response(messages, assistant_msg, model_response, tools=tool_dicts, **kwargs)  
            messages.append(assistant_msg)  
            assistant_msg.log(metrics=True)  
              
            # 4. 工具调用处理  
            should_continue = self._orchestrator.run(  
                messages=messages,  
                assistant_message=assistant_msg,  
                model_response=model_response,  
                functions=functions,  
                func_call_limit=kwargs.get("tool_call_limit"),  
                send_media_to_model=kwargs.get("send_media_to_model", True),  
                is_async=False,  
            )  
            if not should_continue:  
                break  
          
        # 5. 缓存结果  
        if cache_key:  
            self._save_model_response_to_cache(cache_key, model_response)  
        return model_response  
  
    async def aresponse(self, messages: List[Message], **kwargs) -> ModelResponse:  
        """异步响应主流程"""  
        cache_key = None  
        if self.cache_response:  
            cache_key = self._get_model_cache_key(messages, stream=False, **kwargs)  
            if cached := self._get_cached_model_response(cache_key):  
                return ModelResponse.from_dict(cached["result"])  
          
        model_response = ModelResponse()  
        tool_dicts = self._format_tools(kwargs.get("tools"))  
        functions = {t.name: t for t in kwargs.get("tools", []) if isinstance(t, Function)}  
          
        while True:  
            assistant_msg = Message(role=self.assistant_message_role)  
            await self._aprocess_model_response(messages, assistant_msg, model_response, tools=tool_dicts, **kwargs)  
            messages.append(assistant_msg)  
            assistant_msg.log(metrics=True)  
              
            should_continue = self._orchestrator.run(  
                messages=messages,  
                assistant_message=assistant_msg,  
                model_response=model_response,  
                functions=functions,  
                func_call_limit=kwargs.get("tool_call_limit"),  
                send_media_to_model=kwargs.get("send_media_to_model", True),  
                is_async=True,  
            )  
            if not should_continue:  
                break  
          
        if cache_key:  
            self._save_model_response_to_cache(cache_key, model_response)  
        return model_response  
  
    # --- 流式响应方法 ---  
    def response_stream(self, messages: List[Message], **kwargs) -> Iterator[ModelResponse]:  
        cache_key = self._get_model_cache_key(messages, stream=True, **kwargs) if self.cache_response else None  
        if cache_key and (cached := self._get_cached_model_response(cache_key)):  
            yield from self._streaming_responses_from_cache(cached)  
            return  
          
        streaming_responses: List[ModelResponse] = []  
        tool_dicts = self._format_tools(kwargs.get("tools"))  
        functions = {t.name: t for t in kwargs.get("tools", []) if isinstance(t, Function)}  
          
        while True:  
            assistant_msg = Message(role=self.assistant_message_role)  
            if kwargs.get("stream_model_response", True):  
                for delta in self.invoke_stream(messages=messages, assistant_message=assistant_msg,   
                                               tools=tool_dicts, **kwargs):  
                    yield delta  
                    streaming_responses.append(delta)  
            else:  
                model_resp = ModelResponse()  
                self._process_model_response(messages, assistant_msg, model_resp, tools=tool_dicts, **kwargs)  
                yield model_resp  
                streaming_responses.append(model_resp)  
              
            messages.append(assistant_msg)  
            should_continue = self._orchestrator.run(...)  # 参数类似response方法  
            if not should_continue:  
                break  
          
        if cache_key:  
            self._save_streaming_responses_to_cache(cache_key, streaming_responses)  
  
    # --- 抽象方法 (需子类实现) ---  
    @abstractmethod  
    def invoke(self, *args, **kwargs) -> ModelResponse: ...  
      
    @abstractmethod  
    async def ainvoke(self, *args, **kwargs) -> ModelResponse: ...  
      
    @abstractmethod  
    def invoke_stream(self, *args, **kwargs) -> Iterator[ModelResponse]: ...  
      
    @abstractmethod  
    async def ainvoke_stream(self, *args, **kwargs) -> AsyncIterator[ModelResponse]: ...  
      
    @abstractmethod  
    def _parse_provider_response(self, response: Any, **kwargs) -> ModelResponse: ...  
      
    @abstractmethod  
    def _parse_provider_response_delta(self, response: Any) -> ModelResponse: ...  
  
    # --- 辅助方法 ---  
    def _format_tools(self, tools: Optional[List[Union[Function, dict]]]) -> List[Dict[str, Any]]:  
        return [{"type": "function", "function": t.to_dict()} if isinstance(t, Function) else t   
                for t in tools or []]  
      
    def _handle_function_call_media(self, messages: List[Message], results: List[Message], send_media: bool):  
        """聚合所有媒体并通过用户消息传递"""  
        if not send_media:   
            return  
          
        media_collector = {  
            "images": [img for m in results if m.images for img in m.images],  
            "videos": [vid for m in results if m.videos for vid in m.videos],  
            "audios": [aud for m in results if m.audios for aud in m.audios],  
            "files": [f for m in results if m.files for f in m.files],  
        }  
          
        # 清空原消息媒体引用  
        for m in results:  
            m.images = m.videos = m.audios = m.files = None  
          
        if any(media_collector.values()):  
            messages.append(Message(  
                role="user",  
                content="请注意以下生成的媒体内容",  
                **{k: v or None for k, v in media_collector.items()}  
            ))  
  
# ================= 具体模型实现示例 =================  
class MockModel(Model):  
    """模拟模型实现用于测试"""  
    def invoke(self, messages, **kwargs) -> ModelResponse:  
        return ModelResponse(  
            content="Mock response",  
            role="assistant",  
            tool_calls=[] if "no_tool" in kwargs else [{"id": "1", "name": "test_func", "arguments": {}}]  
        )  
      
    async def ainvoke(self, messages, **kwargs) -> ModelResponse:  
        return self.invoke(messages, **kwargs)  
      
    def invoke_stream(self, **kwargs) -> Iterator[ModelResponse]:  
        yield ModelResponse(content="Stream ", event=ModelResponseEvent.content.value)  
        yield ModelResponse(content="part1", event=ModelResponseEvent.content.value)  
        yield ModelResponse(content="part2", event=ModelResponseEvent.content.value)  
      
    async def ainvoke_stream(self, **kwargs) -> AsyncIterator[ModelResponse]:  
        for resp in self.invoke_stream(**kwargs):  
            yield resp  
      
    def _parse_provider_response(self, response, **kwargs) -> ModelResponse:  
        return response  
      
    def _parse_provider_response_delta(self, response) -> ModelResponse:  
        return response  
  
# ================= 测试函数 =================  
def test_model_response_flow():  
    """测试模型响应流程"""  
    print("="*50)  
    print("测试: 基础响应流程")  
    print("="*50)  
      
    # 1. 创建模拟函数  
    def mock_func(**kwargs) -> str:  
        return f"Executed mock_func with {kwargs}"  
      
    mock_function = Function(  
        name="test_func",  
        description="测试函数",  
        fn=mock_func,  
        parameters={"type": "object", "properties": {"input": {"type": "string"}}}  
    )  
      
    # 2. 初始化模型  
    model = MockModel(  
        id="mock-model",  
        name="MockModel",  
        cache_response=True,  
        cache_dir="./test_cache",  
    )  
      
    # 3. 订阅事件  
    def log_event(event: ModelResponse):  
        print(f"📬 事件接收: {event.event} - {event.content[:30]}{'...' if event.content else ''}")  
      
    model.event_bus.subscribe(ModelResponseEvent.tool_call_started.value, log_event)  
    model.event_bus.subscribe(ModelResponseEvent.tool_call_completed.value, log_event)  
      
    # 4. 执行同步调用  
    messages = [Message(role="user", content="Hello, execute test_func")]  
    response = model.response(messages, tools=[mock_function])  
      
    print(f"\n最终响应: {response.content}")  
    print(f"工具执行次数: {len(response.tool_executions or [])}")  
    print(f"缓存目录: {Path('./test_cache').resolve()}")  
      
    # 5. 测试缓存命中  
    print("\n" + "="*50)  
    print("测试: 缓存命中")  
    print("="*50)  
    cached_response = model.response(messages, tools=[mock_function])  
    print(f"缓存响应内容: {cached_response.content}")  
  
async def test_async_model():  
    """测试异步模型流程"""  
    print("\n" + "="*50)  
    print("测试: 异步响应流程")  
    print("="*50)  
      
    model = MockModel(id="async-mock", cache_response=False)  
    messages = [Message(role="user", content="Async test")]  
      
    response = await model.aresponse(messages)  
    print(f"异步响应: {response.content}")  
  
def test_media_handling():  
    """测试媒体处理逻辑"""  
    print("\n" + "="*50)  
    print("测试: 媒体处理")  
    print("="*50)  
      
    # 创建带媒体输出的函数  
    def image_generator():  
        yield Image(id="img1", url="http://example.com/1.jpg")  
        yield Image(id="img2", url="http://example.com/2.jpg")  
      
    media_function = Function(  
        name="generate_images",  
        fn=image_generator,  
        parameters={}  
    )  
      
    model = MockModel(id="media-model")  
    messages = [Message(role="user", content="生成图片")]  
      
    # 模拟工具调用  
    func_call = FunctionCall(  
        call_id="call_123",  
        function=media_function,  
        arguments={}  
    )  
      
    # 通过ToolExecutor执行  
    executor = ToolExecutor(model)  
    results = list(executor.run_function_call(func_call))  
      
    print("工具执行结果消息:")  
    for msg in results:  
        if isinstance(msg, Message):  
            print(f" - 角色: {msg.role}, 图片数: {len(msg.images or [])}")  
  
def test_safe_type_resolver():  
    """测试安全类型解析器"""  
    print("\n" + "="*50)  
    print("测试: 安全类型解析")  
    print("="*50)  
      
    test_cases = [  
        "str", "int", "List[str]", "Dict[str, int]",   
        "os.system('ls')",  # 恶意输入  
        "InvalidType"       # 无效类型  
    ]  
      
    for case in test_cases:  
        resolved = safe_type_resolver(case)  
        print(f"{case!r:20} -> {resolved!r}")  
  
async def main():  
    """主测试函数"""  
    # 同步测试  
    test_model_response_flow()  
    test_media_handling()  
    test_safe_type_resolver()  
      
    # 异步测试  
    await test_async_model()  
      
    # 清理测试缓存  
    import shutil  
    shutil.rmtree("./test_cache", ignore_errors=True)  
    print("\n测试缓存已清理")  
  
if __name__ == "__main__":  
    # 运行测试  
    asyncio.run(main())  