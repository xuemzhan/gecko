"""
Context Manager - 上下文管理器

职责：
1. 管理运行上下文和会话状态
2. 解析和注入依赖项
3. 处理会话元数据
4. 提供状态变量格式化
"""

from __future__ import annotations

import re
import string
from collections import ChainMap
from copy import deepcopy
from inspect import iscoroutine, iscoroutinefunction, signature
from typing import Any, Dict, Optional, Tuple
from uuid import uuid4

from gecko.domain.agent.value_objects import RunContext
from gecko.utils.logging import log_debug, log_warning
from gecko.utils.merge_dict import merge_dictionaries

# 临时导入，后续应移到gecko.domain
try:
    from agno.session import AgentSession
except ImportError:
    AgentSession = Any  # type: ignore


class ContextManager:
    """
    上下文管理器
    
    负责管理Agent运行时的所有上下文相关操作：
    - 会话初始化和状态管理
    - 依赖项解析和注入
    - 元数据管理
    - 模板变量格式化
    """
    
    def __init__(
        self,
        agent_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        default_user_id: Optional[str] = None,
        default_session_id: Optional[str] = None,
    ):
        """
        初始化上下文管理器
        
        Args:
            agent_id: Agent ID
            agent_name: Agent名称
            default_user_id: 默认用户ID
            default_session_id: 默认会话ID
        """
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.default_user_id = default_user_id
        self.default_session_id = default_session_id
    
    # ========================================
    # 会话初始化
    # ========================================
    
    def initialize_session(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Tuple[str, Optional[str]]:
        """初始化会话
        
        如果未提供session_id，则使用默认值或生成新的
        如果未提供user_id，则使用默认值
        
        Args:
            session_id: 会话ID
            user_id: 用户ID
            
        Returns:
            (session_id, user_id) 元组
        """
        # 处理session_id
        if session_id is None:
            if self.default_session_id:
                session_id = self.default_session_id
            else:
                session_id = str(uuid4())
                # 使会话ID粘性（sticky）
                self.default_session_id = session_id
        
        log_debug(f"Session ID: {session_id}", center=True)
        
        # 处理user_id
        if user_id is None or user_id == "":
            user_id = self.default_user_id
        
        return session_id, user_id
    
    def initialize_session_state(
        self,
        session_state: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """初始化会话状态
        
        添加当前运行的上下文信息到会话状态
        
        Args:
            session_state: 现有会话状态
            user_id: 用户ID
            session_id: 会话ID
            run_id: 运行ID
            
        Returns:
            初始化后的会话状态
        """
        if session_state is None:
            session_state = {}
        
        # 添加当前上下文信息
        if user_id:
            session_state["current_user_id"] = user_id
        if session_id is not None:
            session_state["current_session_id"] = session_id
        if run_id is not None:
            session_state["current_run_id"] = run_id
        
        return session_state
    
    # ========================================
    # 会话状态管理
    # ========================================
    
    def load_session_state(
        self,
        session: AgentSession,
        session_state: Dict[str, Any],
        overwrite_db_state: bool = False,
    ) -> Dict[str, Any]:
        """从数据库加载会话状态并合并
        
        合并优先级：run_params > db_state > agent_defaults
        
        Args:
            session: Agent会话对象
            session_state: 当前会话状态（agent_defaults + run_params）
            overwrite_db_state: 是否覆盖数据库状态
            
        Returns:
            合并后的会话状态
        """
        # 从数据库获取会话状态
        if session.session_data is not None and "session_state" in session.session_data:
            session_state_from_db = session.session_data.get("session_state")
            
            if (
                session_state_from_db is not None
                and isinstance(session_state_from_db, dict)
                and len(session_state_from_db) > 0
                and not overwrite_db_state
            ):
                # 保持优先级：run_params > db_state > agent_defaults
                merged_state = session_state_from_db.copy()
                merge_dictionaries(merged_state, session_state)
                session_state.clear()
                session_state.update(merged_state)
        
        # 更新会话中的状态
        if session.session_data is not None:
            session.session_data["session_state"] = session_state
        
        return session_state
    
    def update_session_state(
        self,
        session: AgentSession,
        updates: Dict[str, Any],
    ) -> Dict[str, Any]:
        """更新会话状态
        
        Args:
            session: Agent会话对象
            updates: 要更新的键值对
            
        Returns:
            更新后的会话状态
        """
        if session.session_data is None:
            session.session_data = {"session_state": {}}
        
        if "session_state" not in session.session_data:
            session.session_data["session_state"] = {}
        
        # 合并更新
        merge_dictionaries(session.session_data["session_state"], updates)
        
        return session.session_data["session_state"]
    
    def get_session_state(self, session: AgentSession) -> Dict[str, Any]:
        """获取会话状态
        
        Args:
            session: Agent会话对象
            
        Returns:
            会话状态字典
        """
        if session.session_data is None:
            return {}
        
        return session.session_data.get("session_state", {})
    
    def clear_temp_state(self, session_state: Dict[str, Any]) -> None:
        """清除临时状态变量
        
        移除current_*前缀的临时变量
        
        Args:
            session_state: 会话状态
        """
        session_state.pop("current_session_id", None)
        session_state.pop("current_user_id", None)
        session_state.pop("current_run_id", None)
    
    # ========================================
    # 元数据管理
    # ========================================
    
    def update_metadata(
        self,
        session: AgentSession,
        agent_metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """更新会话元数据
        
        合并Agent元数据和会话元数据
        
        Args:
            session: Agent会话对象
            agent_metadata: Agent的元数据
            
        Returns:
            更新后的元数据
        """
        # 从数据库读取元数据
        if session.metadata is not None:
            # 如果Agent有元数据，合并到会话元数据
            if agent_metadata is not None:
                merge_dictionaries(session.metadata, agent_metadata)
            return session.metadata
        
        return agent_metadata
    
    def get_agent_data(
        self,
        agent_name: Optional[str] = None,
        agent_id: Optional[str] = None,
        model: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """获取Agent数据
        
        Args:
            agent_name: Agent名称
            agent_id: Agent ID
            model: 模型对象
            
        Returns:
            Agent数据字典
        """
        agent_data: Dict[str, Any] = {}
        
        if agent_name is not None:
            agent_data["name"] = agent_name
        if agent_id is not None:
            agent_data["agent_id"] = agent_id
        if model is not None:
            agent_data["model"] = model.to_dict()
        
        return agent_data
    
    # ========================================
    # 依赖解析
    # ========================================
    
    def resolve_dependencies(
        self,
        run_context: RunContext,
        agent: Optional[Any] = None,
    ) -> None:
        """解析运行依赖项（同步）
        
        执行依赖项中的可调用对象，注入agent和run_context
        
        Args:
            run_context: 运行上下文
            agent: Agent实例
        """
        log_debug("Resolving dependencies")
        
        if not isinstance(run_context.dependencies, dict):
            log_warning("Run dependencies are not a dict")
            return
        
        for key, value in run_context.dependencies.items():
            if iscoroutine(value) or iscoroutinefunction(value):
                log_warning(
                    f"Dependency {key} is a coroutine. "
                    "Use async version or agent.arun()"
                )
                continue
            
            if callable(value):
                try:
                    sig = signature(value)
                    
                    # 构建参数
                    kwargs: Dict[str, Any] = {}
                    if "agent" in sig.parameters and agent is not None:
                        kwargs["agent"] = agent
                    if "run_context" in sig.parameters:
                        kwargs["run_context"] = run_context
                    
                    # 执行函数
                    result = value(**kwargs)
                    
                    # 更新结果
                    if result is not None:
                        run_context.dependencies[key] = result
                
                except Exception as e:
                    log_warning(f"Failed to resolve dependency '{key}': {e}")
            else:
                run_context.dependencies[key] = value
    
    async def aresolve_dependencies(
        self,
        run_context: RunContext,
        agent: Optional[Any] = None,
    ) -> None:
        """解析运行依赖项（异步）
        
        Args:
            run_context: 运行上下文
            agent: Agent实例
        """
        log_debug("Resolving dependencies (async)")
        
        if not isinstance(run_context.dependencies, dict):
            log_warning("Run dependencies are not a dict")
            return
        
        for key, value in run_context.dependencies.items():
            if not callable(value):
                run_context.dependencies[key] = value
                continue
            
            try:
                sig = signature(value)
                
                # 构建参数
                kwargs: Dict[str, Any] = {}
                if "agent" in sig.parameters and agent is not None:
                    kwargs["agent"] = agent
                if "run_context" in sig.parameters:
                    kwargs["run_context"] = run_context
                
                # 执行函数
                result = value(**kwargs)
                if iscoroutine(result) or iscoroutinefunction(result):
                    result = await result  # type: ignore
                
                run_context.dependencies[key] = result
            
            except Exception as e:
                log_warning(f"Failed to resolve dependency '{key}': {e}")
    
    # ========================================
    # 模板变量格式化
    # ========================================
    
    def format_message_with_state_variables(
        self,
        message: Any,
        session_state: Optional[Dict[str, Any]] = None,
        dependencies: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
    ) -> Any:
        """使用状态变量格式化消息
        
        支持模板变量替换：${var_name}
        使用ChainMap合并多个上下文
        
        Args:
            message: 消息内容
            session_state: 会话状态
            dependencies: 依赖项
            metadata: 元数据
            user_id: 用户ID
            
        Returns:
            格式化后的消息
        """
        if not isinstance(message, str):
            return message
        
        # 合并所有格式化变量
        format_variables = ChainMap(
            session_state or {},
            dependencies or {},
            metadata or {},
            {"user_id": user_id} if user_id is not None else {},
        )
        
        # 转换{var_name}为${var_name}以避免与JSON冲突
        converted_msg = deepcopy(message)
        for var_name in format_variables.keys():
            # 只转换独立的{var_name}模式，不转换嵌套的
            pattern = r"\{" + re.escape(var_name) + r"\}"
            replacement = "${" + var_name + "}"
            converted_msg = re.sub(pattern, replacement, converted_msg)
        
        # 使用Template安全替换变量
        template = string.Template(converted_msg)
        try:
            result = template.safe_substitute(format_variables)
            return result
        except Exception as e:
            log_warning(f"Template substitution failed: {e}")
            return message
    
    # ========================================
    # 运行上下文创建
    # ========================================
    
    def create_run_context(
        self,
        run_id: str,
        session_id: str,
        user_id: Optional[str] = None,
        session_state: Optional[Dict[str, Any]] = None,
        dependencies: Optional[Dict[str, Any]] = None,
        knowledge_filters: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RunContext:
        """创建运行上下文
        
        Args:
            run_id: 运行ID
            session_id: 会话ID
            user_id: 用户ID
            session_state: 会话状态
            dependencies: 依赖项
            knowledge_filters: 知识库过滤器
            metadata: 元数据
            
        Returns:
            运行上下文对象
        """
        return RunContext(
            run_id=run_id,
            session_id=session_id,
            user_id=user_id,
            session_state=session_state,
            dependencies=dependencies,
            knowledge_filters=knowledge_filters,
            metadata=metadata,
        )
    
    # ========================================
    # 会话状态工具（用于Agent工具调用）
    # ========================================
    
    @staticmethod
    def update_session_state_tool(
        session_state: Dict[str, Any],
        session_state_updates: Dict[str, Any],
    ) -> str:
        """更新共享会话状态（工具函数）
        
        这是一个可以作为Agent工具使用的函数
        
        Args:
            session_state: 当前会话状态
            session_state_updates: 要更新的键值对
            
        Returns:
            更新确认消息
        
        Example:
            session_state_updates = {"shopping_list": ["milk", "eggs", "bread"]}
        """
        for key, value in session_state_updates.items():
            session_state[key] = value
        
        return f"Updated session state: {session_state}"


class SessionStateHelper:
    """
    会话状态辅助类
    
    提供会话状态相关的工具方法
    """
    
    @staticmethod
    def merge_states(
        base_state: Dict[str, Any],
        update_state: Dict[str, Any],
        deep: bool = True,
    ) -> Dict[str, Any]:
        """合并会话状态
        
        Args:
            base_state: 基础状态
            update_state: 更新状态
            deep: 是否深度合并
            
        Returns:
            合并后的状态
        """
        if deep:
            result = deepcopy(base_state)
            merge_dictionaries(result, update_state)
            return result
        else:
            return {**base_state, **update_state}
    
    @staticmethod
    def filter_temp_keys(session_state: Dict[str, Any]) -> Dict[str, Any]:
        """过滤临时键
        
        移除current_*前缀的键
        
        Args:
            session_state: 会话状态
            
        Returns:
            过滤后的状态
        """
        return {
            k: v for k, v in session_state.items()
            if not k.startswith("current_")
        }
    
    @staticmethod
    def validate_state(session_state: Any) -> bool:
        """验证会话状态
        
        Args:
            session_state: 会话状态
            
        Returns:
            是否有效
        """
        if not isinstance(session_state, dict):
            return False
        
        # 检查是否可序列化
        try:
            import json
            json.dumps(session_state)
            return True
        except (TypeError, ValueError):
            return False
    
    @staticmethod
    def get_state_size(session_state: Dict[str, Any]) -> int:
        """获取会话状态大小（字节）
        
        Args:
            session_state: 会话状态
            
        Returns:
            大小（字节）
        """
        import json
        try:
            return len(json.dumps(session_state).encode('utf-8'))
        except Exception:
            return 0


class DependencyResolver:
    """
    依赖解析器
    
    专门处理依赖项的注入和解析
    """
    
    @staticmethod
    def extract_callable_params(func: callable) -> set:
        """提取可调用对象的参数名
        
        Args:
            func: 可调用对象
            
        Returns:
            参数名集合
        """
        try:
            sig = signature(func)
            return set(sig.parameters.keys())
        except Exception:
            return set()
    
    @staticmethod
    def can_inject_agent(func: callable) -> bool:
        """检查是否可以注入agent参数
        
        Args:
            func: 可调用对象
            
        Returns:
            是否可以注入
        """
        params = DependencyResolver.extract_callable_params(func)
        return "agent" in params
    
    @staticmethod
    def can_inject_context(func: callable) -> bool:
        """检查是否可以注入run_context参数
        
        Args:
            func: 可调用对象
            
        Returns:
            是否可以注入
        """
        params = DependencyResolver.extract_callable_params(func)
        return "run_context" in params
    
    @staticmethod
    def is_async_dependency(value: Any) -> bool:
        """检查是否为异步依赖
        
        Args:
            value: 依赖值
            
        Returns:
            是否为异步
        """
        return iscoroutine(value) or iscoroutinefunction(value)