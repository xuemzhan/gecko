# Gecko v0.4 开发设计规格说明书 (DDS)

**版本**: v0.4-Draft  
**状态**: 待评审  
**引用**: Gecko v0.4 系统分析与设计文档 (SAD)

---

## 1. 引言

### 1.1 目的
本设计文档旨在将 SAD 中定义的“并行执行”、“混合记忆”、“鲁棒IO”等高阶需求转化为具体的代码实现方案。重点解决 v0.3.1 中 `WorkflowGraph` 并行能力缺失及 `LiteLLM` 异常处理分散的问题。

### 1.2 范围
- **核心重构**: `gecko.compose.workflow` (图算法与调度器)
- **新增模块**: `gecko.core.memory.hybrid` (混合记忆), `gecko.core.resilience` (熔断与重试)
- **增强模块**: `gecko.plugins.models` (统一错误处理), `gecko.core.structure` (JSON修复)

---

## 2. 模块详细设计

### 2.1 Workflow 引擎重构 (Parallel Execution Engine)

#### 2.1.1 核心类：`WorkflowGraph`
**位置**: `gecko/compose/workflow/graph.py`

**变更点**: 实现分层拓扑排序，支持并行调度。

**算法逻辑 (Kahn's Algorithm 变体)**:
```python
def build_execution_layers(self, start_node: str) -> List[Set[str]]:
    """
    输入: start_node (图的入口)
    输出: List[Set[node_name]] (执行层级列表，例如: [{A}, {B, C}, {D}])
    """
    # 1. 计算入度 (In-Degree)
    # 仅计算从 start_node 可达的子图
    in_degree = {node: 0 for node in self.nodes}
    queue = deque([start_node])
    reachable_nodes = set()
    
    # BFS 遍历计算入度和可达性
    while queue:
        curr = queue.popleft()
        reachable_nodes.add(curr)
        for target, _ in self.edges.get(curr, []):
            in_degree[target] += 1
            if target not in reachable_nodes: # 避免重复入队
                queue.append(target)
    
    # 2. 分层处理
    layers = []
    current_layer = {start_node}
    
    while current_layer:
        layers.append(current_layer)
        next_layer = set()
        
        for node in current_layer:
            for neighbor, _ in self.edges.get(node, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    next_layer.add(neighbor)
        
        current_layer = next_layer
        
    return layers
```

#### 2.1.2 核心类：`Workflow` (Engine Facade)
**位置**: `gecko/compose/workflow/engine.py`

**变更点**: `_execute_loop` 需要重写以支持并行层级。

**伪代码逻辑**:
```python
async def _execute_loop(self, context, ...):
    # 1. 获取执行计划
    layers = self.graph.build_execution_layers(start_node)
    
    for layer in layers:
        # 2. 并行执行层级
        results = await self._execute_layer_parallel(layer, context)
        
        # 3. 状态合并 (Merge Context)
        self._merge_layer_results(context, results)
        
        # 4. 检查跳转 (Next 指令)
        # 注意：并行节点中如果出现 Next 跳转，需定义优先级策略(如 First-Win)
        if self._handle_control_flow(results, context):
            break 
            
        # 5. 持久化 Checkpoint (Layer 粒度)
        await self.persistence.save_checkpoint(...)

async def _execute_layer_parallel(self, layer: Set[str], context: WorkflowContext):
    async with anyio.create_task_group() as tg:
        for node_name in layer:
            # 克隆 Context 以隔离并行节点的修改 (Copy-On-Write 策略)
            node_ctx = context.model_copy(deep=True)
            tg.start_soon(self.executor.execute_node, node_name, ..., node_ctx)
```

### 2.2 Model IO 层增强 (Resilience & Standardization)

#### 2.2.1 异常统一映射
**位置**: `gecko/plugins/models/exceptions.py` (新增)

建立统一的异常体系，屏蔽 LiteLLM/OpenAI/Anthropic 的差异：

```python
class ProviderError(GeckoError): ...
class ContextWindowExceededError(ProviderError): ... # 可自动截断/总结
class RateLimitError(ProviderError): ...             # 可自动退避重试
class AuthenticationError(ProviderError): ...        # 不可重试
class ServiceUnavailableError(ProviderError): ...    # 可熔断
```

#### 2.2.2 熔断器 (Circuit Breaker)
**位置**: `gecko/core/resilience/circuit_breaker.py`

**设计**:
- **状态**: `CLOSED` (正常), `OPEN` (熔断), `HALF_OPEN` (试探)。
- **配置**: `failure_threshold` (5次), `recovery_timeout` (30秒)。
- **实现**: 装饰器模式，应用于 `LiteLLMDriver`。

```python
class CircuitBreaker:
    def __call__(self, func):
        async def wrapper(*args, **kwargs):
            if self.state == 'OPEN':
                if time.now() > self.next_try_time:
                    self.state = 'HALF_OPEN'
                else:
                    raise CircuitOpenError("Model service is currently suspended.")
            
            try:
                result = await func(*args, **kwargs)
                if self.state == 'HALF_OPEN':
                    self.reset()
                return result
            except (RateLimitError, ServiceUnavailableError):
                self.record_failure()
                raise
        return wrapper
```

### 2.3 混合记忆系统 (Hybrid Memory)

#### 2.3.1 架构设计
**位置**: `gecko/core/memory/hybrid.py`

整合 `TokenMemory` (短期) 和 `VectorInterface` (长期)。

```python
class HybridMemory(TokenMemory):
    def __init__(self, short_term_limit: int, vector_store: VectorInterface, ...):
        self.buffer = TokenMemory(max_tokens=short_term_limit)
        self.archive = vector_store
        
    async def get_history(self, query: str = None) -> List[Message]:
        # 1. 获取短期记忆 (Recent N turns)
        short_term_msgs = await self.buffer.get_history(...)
        
        # 2. 如果有 Query，检索长期记忆
        long_term_context = []
        if query:
            docs = await self.archive.search(query, top_k=3)
            long_term_context = [Message.system(f"Relevant Context: {d['text']}") for d in docs]
            
        # 3. 合并策略：长期记忆作为 System Prompt 的一部分或插入到头部
        return long_term_context + short_term_msgs

    async def add_message(self, message: Message):
        # 1. 写入短期 Buffer
        await self.buffer.add_message(message)
        
        # 2. 检查是否触发归档 (e.g., Buffer 满或会话结束)
        if self.buffer.should_archive():
            # 摘要/Embedding 后存入 Vector Store
            await self._archive_oldest_messages()
```

### 2.4 结构化输出修复 (Self-Healing JSON)

#### 2.4.1 修复策略
**位置**: `gecko/core/structure/repair.py`

当 `json.loads` 失败时，启动一个微型 Agent 流程。

```python
REPAIR_PROMPT = """
The following text was intended to be JSON but failed parsing:
{broken_json}

Error: {error_msg}

Please fix the JSON formatting and output ONLY the valid JSON.
"""

async def repair_json(broken_text: str, error: str, model: ModelProtocol) -> Dict:
    prompt = REPAIR_PROMPT.format(broken_json=broken_text, error_msg=error)
    response = await model.acompletion([{"role": "user", "content": prompt}])
    # 尝试再次解析
    return extract_json_from_text(response.content)
```

---

## 3. 数据模型设计

### 3.1 数据库 Schema 更新 (SQLModel)
**位置**: `gecko/plugins/storage/backends/sqlite.py`

```python
class SessionModel(SQLModel, table=True):
    __tablename__ = "gecko_sessions"
    session_id: str = Field(primary_key=True)
    state_json: str
    
    # v0.4 新增
    version: int = Field(default=1)  # 乐观锁版本号
    vector_collection_id: Optional[str] = Field(index=True) # 关联向量库集合
    status: str = Field(default="idle") # idle, running, error
```

### 3.2 WorkflowContext 优化
**位置**: `gecko/compose/workflow/models.py`

增加对大对象的引用支持，避免 Context 膨胀。

```python
class WorkflowContext(BaseModel):
    # ... 原有字段
    
    # 引用大对象（如 PDF 内容、图片 Base64），不直接参与每次 Prompt 构建
    artifacts: Dict[str, Any] = Field(default_factory=dict)
    
    def get_artifact(self, key: str) -> Any:
        return self.artifacts.get(key)
```

---

## 4. 接口规范 (API Spec)

### 4.1 插件协议 (Plugin Protocol)
为了支持动态加载，插件必须遵循以下协议：

**位置**: `gecko/core/protocols/plugin.py`

```python
class PluginManifest(TypedDict):
    name: str
    version: str
    description: str
    entry_point: str # 模块路径

@runtime_checkable
class GeckoPlugin(Protocol):
    def register(self, container: Container) -> None:
        """
        在 DI 容器中注册服务、工具或驱动
        """
        ...
```

### 4.2 配置加载
支持 `pyproject.toml` 的 entry_points：

```toml
[project.entry-points."gecko.plugins"]
my_custom_tool = "my_package.plugin:MyPlugin"
```

---

## 5. 实现计划与优先级

### Phase 1: Core Foundation (Week 1)
1.  **WorkflowGraph**: 实现 `build_execution_layers` 及环检测优化。
2.  **Model Exceptions**: 定义 `gecko/plugins/models/exceptions.py` 并重构 `LiteLLMDriver` 的错误捕获。

### Phase 2: Parallel Execution (Week 2)
1.  **Executor Upgrade**: 改造 `NodeExecutor` 支持 `anyio.TaskGroup`。
2.  **Context Merging**: 实现 Context 的分叉（Fork）与合并（Join）逻辑。
3.  **Tests**: 编写并行节点的单元测试，验证数据竞态。

### Phase 3: Resilience & Memory (Week 3)
1.  **Circuit Breaker**: 实现熔断器装饰器并应用。
2.  **Hybrid Memory**: 实现 `HybridMemory` 类，对接 `ChromaStorage`。
3.  **JSON Repair**: 集成修复逻辑到 `StructureEngine`。

### Phase 4: Plugin & Polish (Week 4)
1.  **Plugin Loader**: 实现基于 `importlib.metadata` 的加载器。
2.  **Documentation**: 更新 API 文档，增加并行工作流示例。

---

## 6. 安全与风控

### 6.1 SSRF 防护
在 `DuckDuckGoSearchTool` 或任何网络请求工具中，增加 URL 检查：
- 禁止访问私有 IP 段 (10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16, 127.0.0.0/8)。
- 限制重定向次数。

### 6.2 状态回滚
当 `Workflow` 执行失败时，提供 `rollback` 选项：
- 利用 `SessionModel` 的 `version` 字段，若检测到执行失败，可选择丢弃当前内存 Context，重新加载上一次成功的 Checkpoint。

---

## 7. 测试策略

1.  **并行性测试**:
    - 构造一个 "Diamond" 形状的 DAG (Start -> A, B -> End)。
    - 在 A 和 B 中加入 `sleep`。
    - 断言总耗时约为 `max(time_a, time_b)` 而非 `sum`。

2.  **熔断测试**:
    - Mock 一个总是抛出 `RateLimitError` 的 Model Driver。
    - 连续调用 5 次，断言第 6 次调用立即抛出 `CircuitOpenError` 且耗时极短。

3.  **序列化兼容性**:
    - 构造包含 `threading.Lock` 等不可序列化对象的 Context。
    - 调用 `PersistenceManager.save_checkpoint`。
    - 断言不抛错，且该字段被替换为占位符。