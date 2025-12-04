# Gecko Engine 模块深度代码审查

**审查时间**: 2025-12-04  
**审查范围**: `gecko/core/engine/` (base.py, react.py, buffer.py)  
**审查目标**: 生产级工业应用就绪性

---

## 执行摘要

| 类别 | 评分 | 状态 |
|------|------|------|
| **架构设计** | 9/10 | ✅ 优秀，无状态设计完善 |
| **代码质量** | 8/10 | ⚠️ 良好，存在改进空间 |
| **错误处理** | 7/10 | ⚠️ 需强化，缺少部分边界情况 |
| **生产就绪** | 7.5/10 | ⚠️ 中等，需修复5个P1问题 + 优化 |
| **可扩展性** | 8/10 | ✅ 良好，Hook机制完善 |

---

## 一、架构设计评析

### 优势 ✅

1. **无状态设计 (Stateless)**
   - Engine 单例可安全并发处理多请求
   - ExecutionContext 完全隔离单次请求状态
   - **评价**: 优秀的微服务/云原生设计

2. **生命周期分解 (Lifecycle Decomposition)**
   - Think → Act → Observe 三阶段清晰分离
   - 子类可轻松扩展单个阶段（如 ReflexionEngine）
   - **评价**: 易于维护和扩展

3. **事件驱动流式处理**
   - 统一 AgentStreamEvent 协议，避免多层 yield 混淆
   - StreamBuffer 优雅处理 OpenAI 流式碎片和 JSON 清洗
   - **评价**: 符合现代 AI 应用的流式交互需求

### 潜在风险 ⚠️

1. **ExecutionContext 的浅拷贝问题**
   ```python
   # base.py 的问题代码
   self.messages = messages.copy()  # 浅拷贝
   ```
   - 如果列表中的 Message 对象被外部修改，会污染上下文
   - **建议**: 改为深拷贝或提供防御性编程文档

2. **缺少并发访问控制**
   - ExecutionContext 不是线程安全的（虽然设计上是异步隔离的）
   - 如果误在同一 context 上调用多个并发 await，会导致竞态条件
   - **建议**: 添加 _lock 或文档约束

---

## 二、P1 (严重) 问题清单

### P1-1: `step()` 中缺少错误处理

**位置**: `react.py` L150-200

**问题**: 
```python
async def step(...) -> Union[AgentOutput, T]:
    kwargs['response_model'] = response_model
    
    async def _run_once(msgs: List[Message]) -> Optional[AgentOutput]:
        final_res = None
        async for event in self.step_stream(msgs, **kwargs):
            if event.type == "result" and event.data:
                final_res = cast(AgentOutput, event.data.get("output"))
            elif event.type == "error":
                logger.error(f"Engine step error: {event.content}")
                raise AgentError(event.content)
        return final_res

    current_messages = list(input_messages)  # ❌ 浅拷贝，Message 可被污染
    final_output = await _run_once(current_messages)
    
    if not final_output:
        return AgentOutput(content="[System Error] No output generated.")  # ❌ 无 token 计数，可能导致计费错误
```

**风险**:
- 无异常处理的 step() 调用会导致应用崩溃
- 空输出返回值不包含 usage/tokens，无法正确计费
- cast() 可能失败但不被捕获

**修复方案**:
```python
async def step(...) -> Union[AgentOutput, T]:
    try:
        current_messages = [
            Message(**m.dict()) for m in input_messages
        ]  # 深拷贝
        
        final_output = await _run_once(current_messages)
        
        if not final_output:
            return AgentOutput(
                content="[System Error] No output generated.",
                usage={"input_tokens": 0, "output_tokens": 0}  # 补充计费信息
            )
        
        return final_output
    except Exception as e:
        logger.exception("step() failed", error=str(e))
        raise AgentError(f"Step execution failed: {e}") from e
```

**优先级**: P1 - 会导致应用异常

---

### P1-2: `step_stream()` 缺少超时保护

**位置**: `react.py` L225-260

**问题**:
```python
async def step_stream(self, input_messages: List[Message], **kwargs: Any) -> AsyncIterator[AgentStreamEvent]:
    context = await self._build_execution_context(input_messages)
    
    try:
        async for event in self._execute_lifecycle(context, **kwargs):
            yield event
    except Exception as e:
        logger.exception("Lifecycle execution crashed")
        await self.on_error(e, input_messages, **kwargs)
        yield AgentStreamEvent(type="error", content=str(e))
        raise
```

**风险**:
- `_execute_lifecycle` 的 while 循环可能因网络延迟或模型响应缓慢而无期限挂起
- 没有整体超时机制
- max_turns 只限制了轮数，但每轮的单个 LLM 调用可能无限等待

**修复方案**:
```python
async def step_stream(self, input_messages: List[Message], 
                     timeout: float = 300.0,  # 默认 5 分钟
                     **kwargs: Any) -> AsyncIterator[AgentStreamEvent]:
    try:
        async with asyncio.timeout(timeout):  # Python 3.11+
            async for event in self._execute_lifecycle(context, **kwargs):
                yield event
    except asyncio.TimeoutError:
        logger.error("step_stream timeout", timeout_s=timeout)
        yield AgentStreamEvent(type="error", 
                             content=f"Execution timeout after {timeout}s")
        raise
```

**优先级**: P1 - 导致资源泄漏

---

### P1-3: `_phase_think()` 中模型流异常未处理

**位置**: `react.py` L450-475

**问题**:
```python
async def _phase_think(self, context: ExecutionContext, **kwargs: Any) -> AsyncIterator[StreamChunk]:
    messages_payload = [m.to_openai_format() for m in context.messages]
    
    llm_params = self._build_llm_params(kwargs.get('response_model'), "auto")
    safe_kwargs = {k: v for k, v in kwargs.items() if k not in ['response_model']}
    llm_params.update(safe_kwargs)
    
    llm_params["stream"] = True
    
    stream_gen = self.model.astream(messages=messages_payload, **llm_params)
    async for chunk in stream_gen:  # ❌ 无异常处理
        yield chunk
```

**风险**:
- 模型 API 返回 4xx/5xx 错误时无重试机制
- StreamChunk 格式异常会导致整个流中断
- 没有错误事件 yield，上游无法获知失败原因

**修复方案**:
```python
async def _phase_think(self, context: ExecutionContext, **kwargs: Any) -> AsyncIterator[StreamChunk]:
    try:
        stream_gen = self.model.astream(messages=messages_payload, **llm_params)
        async for chunk in stream_gen:
            if not isinstance(chunk, StreamChunk):
                logger.warning(f"Invalid StreamChunk: {type(chunk)}")
                continue
            yield chunk
    except Exception as e:
        logger.error(f"Model streaming failed: {e}")
        yield AgentStreamEvent(
            type="error",
            content=f"Model API error: {e}"
        )
        raise
```

**优先级**: P1 - 生产环境频繁 API 错误

---

### P1-4: `_clean_arguments()` 中 JSON 修复不完整

**位置**: `buffer.py` L90-120

**问题**:
```python
def _clean_arguments(self, raw_json: str) -> str:
    if not raw_json:
        return "{}"
    
    try:
        json.loads(raw_json)
        return raw_json
    except json.JSONDecodeError:
        pass
    
    cleaned = raw_json.strip()
    
    # 去除 Markdown 代码块
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
    if match:
        cleaned = match.group(1)
    
    # 简单修复：去除首尾多余的误加引号
    if cleaned.startswith("'") and cleaned.endswith("'"):
        cleaned = cleaned[1:-1]
    elif cleaned.startswith('"') and cleaned.endswith('"'):
        cleaned = cleaned[1:-1]
    
    try:
        json.loads(cleaned)
        return cleaned
    except Exception:
        logger.warning(f"Failed to clean JSON arguments: {raw_json[:50]}...")
        return raw_json  # ❌ 返回脏数据
```

**风险**:
- 返回未验证的原始脏 JSON，后续工具执行会因 JSONDecodeError 崩溃
- 缺少对常见格式错误的修复（如去除尾部逗号、未转义的换行符）
- 日志中只记录前 50 字符，无法完整诊断

**修复方案**:
```python
def _clean_arguments(self, raw_json: str) -> str:
    """
    进阶 JSON 清洗：处理 LLM 的常见输出格式问题
    """
    if not raw_json:
        return "{}"
    
    try:
        json.loads(raw_json)
        return raw_json
    except json.JSONDecodeError:
        pass
    
    cleaned = raw_json.strip()
    
    # 1. 去除 Markdown 代码块
    if cleaned.startswith("```"):
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
        if match:
            cleaned = match.group(1).strip()
    
    # 2. 去除首尾错误加的引号
    if (cleaned.startswith("'") and cleaned.endswith("'")) or \
       (cleaned.startswith('"') and cleaned.endswith('"')):
        cleaned = cleaned[1:-1]
    
    # 3. 修复尾部逗号 ({"key": "value",})
    cleaned = re.sub(r',\s*}', '}', cleaned)  # {...,} -> {...}
    cleaned = re.sub(r',\s*\]', ']', cleaned)  # [...,] -> [...]
    
    # 4. 修复未转义的换行符和特殊字符
    # 将多行字符串中的 \n 替换为 \\n
    try:
        json.loads(cleaned)
        return cleaned
    except json.JSONDecodeError as e:
        logger.warning(f"JSON cleanup failed: {e}\nOriginal: {raw_json}")
        # 最后手段：尝试 eval（不推荐，仅在允许的环保下）
        # 如果无法修复，返回空对象而非脏数据
        return "{}"
```

**优先级**: P1 - 直接导致工具执行失败

---

### P1-5: `_build_execution_context()` 的系统提示注入缺陷

**位置**: `react.py` L505-525

**问题**:
```python
async def _build_execution_context(self, input_messages: List[Message]) -> ExecutionContext:
    history = await self._load_history()
    all_messages = history + input_messages
    
    has_system = any(m.role == "system" for m in all_messages)
    if not has_system:
        template_vars = {
            "tools": self.toolbox.to_openai_schema(),
            "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        system_content = self.prompt_template.format_safe(**template_vars)
        all_messages.insert(0, Message.system(system_content))  # ❌ 盲目插入
```

**风险**:
1. 如果用户在 input_messages 中已包含 system message，会导致:
   - 系统提示出现在用户消息之后（打乱消息顺序）
   - OpenAI API 不允许多个 system role 消息（API 错误）

2. `to_openai_schema()` 可能返回极大的 schema（大模型上万个 token），导致:
   - Context 窗口浪费
   - 成本增加

3. `prompt_template.format_safe()` 如果失败，会无声地返回原始模板

**修复方案**:
```python
async def _build_execution_context(self, input_messages: List[Message]) -> ExecutionContext:
    history = await self._load_history()
    all_messages = history + input_messages
    
    # 检查是否已有 system 消息（包括 input_messages 中的）
    system_msg = next((m for m in all_messages if m.role == "system"), None)
    
    if system_msg is None:
        template_vars = {
            "tools": self.toolbox.to_openai_schema(),
            "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        try:
            system_content = self.prompt_template.format_safe(**template_vars)
            # 确保 system 消息在最前面
            all_messages.insert(0, Message.system(system_content))
        except Exception as e:
            logger.warning(f"System prompt formatting failed: {e}")
            # Fallback: 使用最小系统提示
            all_messages.insert(0, Message.system("You are a helpful AI assistant."))
    elif any(m.role == "system" for m in input_messages):
        # 用户显式指定了 system，记录日志但不覆盖
        logger.info("User-specified system message will be used")
    
    return ExecutionContext(all_messages)
```

**优先级**: P1 - API 兼容性和成本问题

---

## 三、P2 (主要) 问题清单

### P2-1: StreamBuffer 中的并发访问问题

**位置**: `react.py` L280-295

**问题**:
```python
while context.turn < self.max_turns:
    context.turn += 1
    
    buffer = StreamBuffer()  # 新建缓冲区
    
    async for chunk in self._phase_think(context, **kwargs):
        text_delta = buffer.add_chunk(chunk)  # ❌ 不是线程安全的
        if text_delta:
            yield AgentStreamEvent(type="token", content=text_delta)
    
    assistant_msg = buffer.build_message()  # ❌ 可能产出不完整的消息
```

**风险**:
- 如果 StreamChunk 到达顺序错乱（虽然理论上不应该，但网络不可靠），会导致 tool_calls_map 状态混乱
- StreamBuffer 没有完整性校验（如 tool_calls 是否都完整接收）

**修复方案**:
```python
class StreamBuffer:
    def __init__(self):
        self.content_parts: List[str] = []
        self.tool_calls_map: Dict[int, Dict[str, Any]] = {}
        self._completed: bool = False  # 标记缓冲区是否已完成接收
        self._lock = asyncio.Lock()  # 若需多任务并发访问
    
    async def add_chunk(self, chunk: StreamChunk) -> Optional[str]:
        # 如果需要并发安全（虽然单个 stream 不需要）
        # async with self._lock:
        #     ... (现有逻辑)
        
        if not chunk or not hasattr(chunk, 'delta'):
            logger.warning(f"Received invalid chunk: {type(chunk)}")
            return None
        
        # ... 现有逻辑 ...
    
    def build_message(self) -> Message:
        """构建消息，验证完整性"""
        if not self.content_parts and not self.tool_calls_map:
            logger.warning("Building message from empty buffer")
        
        # ... 现有逻辑 ...
        return Message.assistant(...)
```

**优先级**: P2 - 潜在的数据完整性问题

---

### P2-2: 内存泄漏风险 - ExecutionContext 历史消息无界增长

**位置**: `react.py` L60-80

**问题**:
```python
class ExecutionContext:
    def __init__(self, messages: List[Message]):
        self.messages = messages.copy()  # ❌ 持续追加，无上限
        self.turn = 0
        self.metadata: Dict[str, Any] = {}
        self.consecutive_errors: int = 0
        self.last_tool_hash: Optional[int] = None
    
    def add_message(self, message: Message) -> None:
        self.messages.append(message)  # 每轮都会追加 1-3 条消息
```

**风险**:
- 如果 max_turns = 100，可能产生 300+ 消息，每条消息 1KB，总计 300KB+
- 对于大型 DAG 的分布式执行，可能累积 MB 级内存
- 长连接场景（如 REPL）会导致内存持续增长

**修复方案**:
```python
class ExecutionContext:
    def __init__(self, messages: List[Message], max_history: int = 50):
        self.messages = messages.copy()
        self.max_history = max_history
        self.turn = 0
        self.metadata: Dict[str, Any] = {}
        self.consecutive_errors: int = 0
        self.last_tool_hash: Optional[int] = None
    
    def add_message(self, message: Message) -> None:
        """追加消息，自动清理超出历史限制的旧消息"""
        self.messages.append(message)
        
        # 保留最近 max_history 条消息，以及所有 system 消息
        system_msgs = [m for m in self.messages if m.role == "system"]
        other_msgs = [m for m in self.messages if m.role != "system"]
        
        if len(other_msgs) > self.max_history:
            # 保留最新的 max_history 条，删除最老的
            self.messages = system_msgs + other_msgs[-self.max_history:]
            logger.info(f"Trimmed context to {self.max_history} messages")
```

**优先级**: P2 - 长期运行稳定性问题

---

### P2-3: `_detect_loop()` 的哈希碰撞风险

**位置**: `react.py` L545-570

**问题**:
```python
def _detect_loop(self, context: ExecutionContext, msg: Message) -> bool:
    if not msg.safe_tool_calls:
        return False
    
    try:
        calls_dump = json.dumps(
            [
                {
                    "name": tc.get("function", {}).get("name"),
                    "args": tc.get("function", {}).get("arguments"),
                }
                for tc in msg.safe_tool_calls
            ],
            sort_keys=True,
        )
        current_hash = hash(calls_dump)  # ❌ Python hash() 非加密安全，可能碰撞
        
        if context.last_tool_hash == current_hash:
            return True
        
        context.last_tool_hash = current_hash
        return False
    except Exception:
        return False
```

**风险**:
- `hash()` 是快速的，但非加密安全，存在碰撞风险（虽然概率低）
- 只检查与上一轮的相同性，无法检测 2-3 轮循环（如 A -> B -> A 的模式）
- 异常被吞掉，无法诊断问题

**修复方案**:
```python
import hashlib

def _detect_loop(self, context: ExecutionContext, msg: Message) -> bool:
    if not msg.safe_tool_calls:
        return False
    
    try:
        calls_dump = json.dumps(
            [{"name": tc.get("function", {}).get("name"),
              "args": tc.get("function", {}).get("arguments")}
             for tc in msg.safe_tool_calls],
            sort_keys=True
        )
        # 使用 SHA256 替代 hash()
        current_hash = hashlib.sha256(calls_dump.encode()).hexdigest()
        
        # 高级检测：检查最近 3 轮的工具调用
        if not hasattr(context, 'last_tool_hashes'):
            context.last_tool_hashes = []
        
        # 检查是否与最近任何一轮相同（2-3 轮循环）
        if current_hash in context.last_tool_hashes:
            logger.warning(f"Loop pattern detected: {calls_dump[:100]}")
            return True
        
        # 只保留最近 3 轮
        context.last_tool_hashes.append(current_hash)
        context.last_tool_hashes = context.last_tool_hashes[-3:]
        
        return False
    except Exception as e:
        logger.warning(f"Loop detection failed: {e}")
        return False
```

**优先级**: P2 - 死循环检测不完美

---

### P2-4: `_phase_observe()` 中错误提示注入不规范

**位置**: `react.py` L515-540

**问题**:
```python
async def _phase_observe(self, context: ExecutionContext, results: List[ToolExecutionResult]) -> bool:
    error_count = sum(1 for r in results if r.is_error)
    
    if error_count > 0:
        context.consecutive_errors += 1
    else:
        context.consecutive_errors = 0
    
    if context.consecutive_errors >= 3:
        logger.warning("Too many consecutive tool errors.")
        context.add_message(Message.user(
            "System Alert: The last 3 tool calls failed. "  # ❌ 用 user 角色而非 system
            "Please stop repeating the same action. "
            "Analyze the error message and change your parameters or approach."
        ))
        context.consecutive_errors = 0
        return True
    
    return True
```

**风险**:
- 系统错误提示作为 user message 注入，容易被 LLM 误识别为真实用户指令
- OpenAI API 对 user/system message 顺序有严格要求，此处违反
- 没有限制这种自动重试的次数，可能无限循环

**修复方案**:
```python
async def _phase_observe(self, context: ExecutionContext, 
                        results: List[ToolExecutionResult]) -> bool:
    error_count = sum(1 for r in results if r.is_error)
    
    if error_count > 0:
        context.consecutive_errors += 1
    else:
        context.consecutive_errors = 0
    
    # 最多允许自动重试 2 次
    max_auto_retries = 2
    
    if context.consecutive_errors >= 3 and context.turn < max_auto_retries:
        logger.warning(f"Tool errors detected: {error_count}/{len(results)}")
        # 使用 assistant 消息（由系统而非用户生成）或直接在 tool_result 中体现
        # 更好的做法是在 _phase_act 中返回友好的错误消息
        context.consecutive_errors = 0
        return True
    elif context.consecutive_errors >= 3:
        logger.error("Too many tool errors, stopping execution")
        return False
    
    return True
```

**优先级**: P2 - API 兼容性和提示词注入风险

---

## 四、P3 (改进) 建议

### P3-1: 缺少日志结构化

**当前**:
```python
logger.error(f"Engine step error: {event.content}")
logger.exception("Lifecycle execution crashed")
```

**建议**:
```python
logger.error("engine_step_failed", 
             error_type=type(e).__name__,
             error_msg=str(e),
             turn=context.turn,
             message_count=len(context.messages))
```

---

### P3-2: 缺少性能监控 Instrumentation

**建议添加**:
```python
class ReActEngine(CognitiveEngine):
    def __init__(self, ..., enable_metrics: bool = True):
        ...
        self.metrics = {
            "total_turns": 0,
            "tool_calls": 0,
            "errors": 0,
            "avg_turn_time": 0.0,
            "avg_token_per_request": 0,
        }
    
    async def _execute_lifecycle(self, context, **kwargs):
        turn_start = time.time()
        # ...
        turn_duration = time.time() - turn_start
        self.metrics["total_turns"] += 1
        self.metrics["avg_turn_time"] = (
            (self.metrics["avg_turn_time"] * (self.metrics["total_turns"] - 1) + turn_duration)
            / self.metrics["total_turns"]
        )
```

---

### P3-3: 缺少成本追踪

**建议**:
```python
class ExecutionContext:
    def __init__(self, ...):
        ...
        self.token_usage = {"input": 0, "output": 0, "total": 0}
        self.estimated_cost = 0.0
    
    def add_usage(self, input_tokens: int, output_tokens: int, 
                  cost_per_1k_input: float = 0.015, 
                  cost_per_1k_output: float = 0.03):
        self.token_usage["input"] += input_tokens
        self.token_usage["output"] += output_tokens
        self.estimated_cost += (input_tokens / 1000 * cost_per_1k_input) + \
                               (output_tokens / 1000 * cost_per_1k_output)
```

---

### P3-4: Hook 机制缺少验证

**当前**:
```python
async def before_step(self, input_messages: List[Message], **kwargs) -> None:
    if self.before_step_hook:
        try:
            if asyncio.iscoroutinefunction(self.before_step_hook):
                await self.before_step_hook(input_messages, **kwargs)
            else:
                self.before_step_hook(input_messages, **kwargs)
        except Exception as e:
            logger.warning("before_step_hook failed", error=str(e))
```

**问题**: Hook 抛异常时只记录 warning，不中断执行

**建议**:
```python
async def before_step(self, ..., fail_fast: bool = False) -> None:
    if self.before_step_hook:
        try:
            if asyncio.iscoroutinefunction(self.before_step_hook):
                await self.before_step_hook(input_messages, **kwargs)
            else:
                self.before_step_hook(input_messages, **kwargs)
        except Exception as e:
            logger.error("before_step_hook failed", error=str(e))
            if fail_fast:
                raise
```

---

## 五、安全性审查

### 提示词注入风险 ⚠️

**高风险区域**:
1. `_build_execution_context()` 中的 system prompt 格式化
   - 如果用户 input 中包含 "{{ tools }}"，可能被模板替换
   - **建议**: 使用更安全的模板引擎（如 Jinja2 with autoescape）

2. `_phase_observe()` 中的错误消息注入
   - 工具输出直接注入到消息历史
   - **建议**: 对工具输出进行清理和转义

---

### 资源耗尽风险 ⚠️

**高风险区域**:
1. 无界消息历史增长 ✅ (P2-2 已修复)
2. 工具输出无上限截断 ⚠️ (已有 max_observation_length，但无验证)
3. 并发工具执行无限制 ⚠️ (ToolBox.execute_many 应有 max_concurrency)

---

## 六、测试覆盖建议

```python
# 应补充的测试用例
class TestReActEngine:
    
    async def test_timeout_protection(self):
        """测试超时保护"""
        engine = ReActEngine(...)
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                engine.step(..., timeout=0.1),
                timeout=1.0
            )
    
    async def test_context_message_limit(self):
        """测试消息历史上限"""
        engine = ReActEngine(...)
        context = ExecutionContext([...], max_history=10)
        for _ in range(20):
            context.add_message(Message.assistant(...))
        assert len(context.messages) <= 10
    
    async def test_json_cleaning_edge_cases(self):
        """测试 JSON 清洗的边界情况"""
        buffer = StreamBuffer()
        
        # 测试尾部逗号
        assert buffer._clean_arguments('{"a": 1,}') == '{"a": 1}'
        
        # 测试 Markdown 包裹
        assert buffer._clean_arguments(
            '```json\n{"a": 1}\n```'
        ) == '{"a": 1}'
        
        # 测试脏数据降级
        result = buffer._clean_arguments('{"unclosed": ')
        assert result == '{}'  # 而非原始脏数据
    
    async def test_loop_detection_patterns(self):
        """测试死循环检测"""
        engine = ReActEngine(...)
        context = ExecutionContext([])
        
        # 测试简单循环（A -> A）
        msg1 = Message.assistant(tool_calls=[{"function": {"name": "A", "arguments": "{}"}}])
        assert engine._detect_loop(context, msg1) is False
        assert engine._detect_loop(context, msg1) is True  # 重复
        
        # 测试复杂循环（A -> B -> A）
        msg2 = Message.assistant(tool_calls=[{"function": {"name": "B", "arguments": "{}"}}])
        msg3 = Message.assistant(tool_calls=[{"function": {"name": "A", "arguments": "{}"}}])
        # ... 验证检测逻辑
```

---

## 七、工业级优化建议

### 1. 添加优雅降级 (Graceful Degradation)

```python
async def step(self, input_messages, **kwargs):
    """带降级的执行"""
    try:
        return await self.step_stream(input_messages, **kwargs)
    except ModelError:
        # Fallback 1: 使用缓存的上一个成功响应
        if self.memory.has_cache(cache_key):
            return await self.memory.get_cache(cache_key)
        
        # Fallback 2: 返回简单的错误响应而非抛异常
        logger.warning("Model failed, returning degraded response")
        return AgentOutput(
            content="I'm experiencing temporary difficulties. "
                   "Please try again in a moment.",
            is_degraded=True
        )
```

### 2. 添加速率限制和背压 (Backpressure)

```python
async def step_stream(self, input_messages, **kwargs):
    """带背压的流式执行"""
    # 限制并发请求数
    semaphore = asyncio.Semaphore(self.max_concurrent_requests)
    
    async with semaphore:
        async for event in self._execute_lifecycle(...):
            yield event
```

### 3. 添加链路追踪 (Distributed Tracing)

```python
from gecko.core.tracing import trace_id_var

async def step_stream(self, input_messages, **kwargs):
    trace_id = trace_id_var.get() or generate_trace_id()
    
    logger.info("step_stream_start", trace_id=trace_id, ...)
    try:
        async for event in self._execute_lifecycle(...):
            yield event
    finally:
        logger.info("step_stream_end", trace_id=trace_id, ...)
```

---

## 八、总体建议优先级

| 优先级 | 项目 | 预计工作量 | 完成期限 |
|--------|------|-----------|---------|
| P1 | 修复 5 个严重问题 | 2-3 天 | 本周 |
| P2 | 修复 4 个主要问题 | 2-3 天 | 本周 |
| P3 | 性能监控和优化 | 3-5 天 | 下周 |
| P3 | 工业级特性（降级、追踪） | 5-7 天 | 第三周 |

---

## 九、总体评分细分

| 维度 | 得分 | 说明 |
|------|------|------|
| **代码结构** | 9/10 | 架构清晰，分层明确 |
| **错误处理** | 6/10 | 缺少关键的异常处理（P1 问题） |
| **性能优化** | 7/10 | 无界内存增长，缺少监控 |
| **安全性** | 7/10 | 存在提示词注入风险 |
| **可维护性** | 8/10 | Hook 机制完善，但日志可改进 |
| **生产就绪** | 7.5/10 | **需修复 P1 问题后可用于生产** |

---

## 结论

**Gecko Engine 模块设计优秀，架构符合生产级要求**，但存在以下关键问题需要立即修复：

1. ✅ **无状态设计** - 优秀，支持高并发
2. ✅ **生命周期分解** - 优秀，易于扩展
3. ⚠️ **错误处理不完整** - P1 问题，需修复
4. ⚠️ **缺少超时保护** - P1 问题，需修复
5. ⚠️ **JSON 清洗不完善** - P1 问题，需修复

**建议**: 
- 立即修复 P1 和 P2 问题（预计 3-4 天）
- 后续进行性能监控和工业级优化
- 补充完整的单元测试和集成测试

**预期**: 修复后可以达到 **9/10 的生产就绪度**。
