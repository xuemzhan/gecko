# examples/protocol_demo.py
"""
Gecko Protocol 演示文件

展示如何实现和使用 Gecko 框架中定义的各种协议。

包含的演示：
1. ModelProtocol - 基础模型实现
2. StreamableModelProtocol - 流式模型实现
3. StorageProtocol - 存储后端实现
4. ToolProtocol - 工具实现
5. EmbedderProtocol - 嵌入模型实现
6. RunnableProtocol - 可运行对象实现
7. VectorStoreProtocol - 向量存储实现
8. 能力检测和验证

运行方式:
    python examples/protocol_demo.py
"""

import asyncio
from typing import Any, AsyncIterator, Dict, List, Optional

from gecko.core.protocols import (
    # 协议
    ModelProtocol,
    StreamableModelProtocol,
    StorageProtocol,
    ToolProtocol,
    EmbedderProtocol,
    RunnableProtocol,
    VectorStoreProtocol,
    # 响应模型
    CompletionResponse,
    CompletionChoice,
    CompletionUsage,
    StreamChunk,
    # 工具函数
    check_protocol,
    supports_streaming,
    supports_function_calling,
    supports_vision,
    get_model_name,
    validate_model,
    validate_storage,
    validate_tool,
)


# ==================== 1. ModelProtocol 示例 ====================

class SimpleModel:
    """
    简单的模型实现（仅支持补全）
    
    演示：
    - 实现 ModelProtocol 的最小要求
    - 返回标准的 CompletionResponse
    """
    
    def __init__(self, model_name: str = "simple-model-v1"):
        self.model_name = model_name
        self._call_count = 0
    
    async def acompletion(
        self, 
        messages: List[Dict[str, Any]], 
        **kwargs
    ) -> CompletionResponse:
        """异步补全接口"""
        self._call_count += 1
        
        # 模拟 API 调用延迟
        await asyncio.sleep(0.1)
        
        # 提取最后一条用户消息
        user_message = next(
            (msg["content"] for msg in reversed(messages) if msg["role"] == "user"),
            "No user message"
        )
        
        # 构建响应
        response_content = f"Echo: {user_message} (call #{self._call_count})"
        
        return CompletionResponse(
            id=f"cmpl-{self._call_count}",
            model=self.model_name,
            created=1234567890,
            choices=[
                CompletionChoice(
                    index=0,
                    message={
                        "role": "assistant",
                        "content": response_content
                    },
                    finish_reason="stop"
                )
            ],
            usage=CompletionUsage(
                prompt_tokens=len(user_message),
                completion_tokens=len(response_content),
                total_tokens=len(user_message) + len(response_content)
            )
        )


# ==================== 2. StreamableModelProtocol 示例 ====================

class StreamingModel:
    """
    支持流式输出的模型
    
    演示：
    - 实现 StreamableModelProtocol
    - 流式生成响应内容
    """
    
    def __init__(self, model_name: str = "streaming-model-v1"):
        self.model_name = model_name
        self._supports_function_calling = True
        self._supports_vision = False
    
    async def acompletion(
        self, 
        messages: List[Dict[str, Any]], 
        **kwargs
    ) -> CompletionResponse:
        """非流式补全"""
        await asyncio.sleep(0.1)
        
        user_message = next(
            (msg["content"] for msg in reversed(messages) if msg["role"] == "user"),
            ""
        )
        
        response_content = f"Complete response to: {user_message}"
        
        return CompletionResponse(
            id="cmpl-streaming",
            model=self.model_name,
            choices=[
                CompletionChoice(
                    message={"role": "assistant", "content": response_content},
                    finish_reason="stop"
                )
            ]
        )
    
    async def astream(
        self, 
        messages: List[Dict[str, Any]], 
        **kwargs
    ) -> AsyncIterator[StreamChunk]:
        """流式补全"""
        user_message = next(
            (msg["content"] for msg in reversed(messages) if msg["role"] == "user"),
            ""
        )
        
        response_text = f"Streaming response to: {user_message}"
        words = response_text.split()
        
        for i, word in enumerate(words):
            # 模拟流式延迟
            await asyncio.sleep(0.05)
            
            # 生成 chunk
            chunk = StreamChunk(
                id="chunk-streaming",
                model=self.model_name,
                choices=[
                    {
                        "index": 0,
                        "delta": {"content": word + " "},
                        "finish_reason": None if i < len(words) - 1 else "stop"
                    }
                ]
            )
            
            yield chunk


# ==================== 3. StorageProtocol 示例 ====================

class MemoryStorage:
    """
    内存存储后端（用于演示和测试）
    
    演示：
    - 实现 StorageProtocol
    - 支持 TTL（过期时间）
    """
    
    def __init__(self):
        self._data: Dict[str, Dict[str, Any]] = {}
    
    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        """获取数据"""
        if key not in self._data:
            return None
        
        item = self._data[key]
        
        # 检查 TTL
        if "ttl" in item and "stored_at" in item:
            import time
            age = time.time() - item["stored_at"]
            if age > item["ttl"]:
                # 已过期，删除并返回 None
                del self._data[key]
                return None
        
        return item.get("value")
    
    async def set(
        self, 
        key: str, 
        value: Dict[str, Any], 
        ttl: Optional[int] = None
    ) -> None:
        """存储数据"""
        import time
        
        self._data[key] = {
            "value": value,
            "stored_at": time.time(),
        }
        
        if ttl is not None:
            self._data[key]["ttl"] = ttl
    
    async def delete(self, key: str) -> bool:
        """删除数据"""
        if key in self._data:
            del self._data[key]
            return True
        return False
    
    async def exists(self, key: str) -> bool:
        """检查键是否存在（包含过期检查）"""
        result = await self.get(key)
        return result is not None
    
    async def clear(self) -> None:
        """清空所有数据"""
        self._data.clear()


# ==================== 4. ToolProtocol 示例 ====================

class CalculatorTool:
    """
    计算器工具
    
    演示：
    - 实现 ToolProtocol
    - 定义参数 Schema
    - 安全执行用户输入
    """
    
    name = "calculator"
    description = "Execute mathematical expressions safely"
    parameters = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "The mathematical expression to evaluate (e.g., '2+2', '10*5')"
            }
        },
        "required": ["expression"]
    }
    
    async def execute(self, arguments: Dict[str, Any]) -> str:
        """执行计算"""
        expression = arguments.get("expression", "")
        
        # 安全检查：仅允许数字和基本运算符
        allowed_chars = set("0123456789+-*/()., ")
        if not all(c in allowed_chars for c in expression):
            return f"Error: Invalid characters in expression '{expression}'"
        
        try:
            # 使用 eval（在受控环境中）
            result = eval(expression, {"__builtins__": {}}, {})
            return f"Result: {result}"
        except Exception as e:
            return f"Error: {str(e)}"


class WebSearchTool:
    """
    网络搜索工具（模拟）
    
    演示：
    - 更复杂的参数 Schema
    - 异步操作
    """
    
    name = "web_search"
    description = "Search the web for information"
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query"
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return",
                "default": 5
            },
            "language": {
                "type": "string",
                "description": "Search language (e.g., 'en', 'zh')",
                "default": "en"
            }
        },
        "required": ["query"]
    }
    
    async def execute(self, arguments: Dict[str, Any]) -> str:
        """执行搜索"""
        query = arguments["query"]
        max_results = arguments.get("max_results", 5)
        language = arguments.get("language", "en")
        
        # 模拟搜索延迟
        await asyncio.sleep(0.2)
        
        # 模拟搜索结果
        results = [
            f"Result {i+1}: Information about '{query}' (lang: {language})"
            for i in range(max_results)
        ]
        
        return "\n".join(results)


# ==================== 5. EmbedderProtocol 示例 ====================

class SimpleEmbedder:
    """
    简单的嵌入模型（使用随机向量演示）
    
    演示：
    - 实现 EmbedderProtocol
    - 批量处理文本
    """
    
    def __init__(self, dimension: int = 384):
        self._dimension = dimension
    
    async def embed(self, texts: List[str]) -> List[List[float]]:
        """批量嵌入文本"""
        import random
        
        # 模拟 API 延迟
        await asyncio.sleep(0.1)
        
        # 生成随机向量（实际应调用真实的嵌入模型）
        embeddings = []
        for text in texts:
            # 使用文本长度作为种子，保持一致性
            random.seed(len(text))
            embedding = [random.random() for _ in range(self._dimension)]
            embeddings.append(embedding)
        
        return embeddings
    
    async def embed_single(self, text: str) -> List[float]:
        """嵌入单个文本"""
        results = await self.embed([text])
        return results[0]
    
    def get_dimension(self) -> int:
        """获取向量维度"""
        return self._dimension


# ==================== 6. RunnableProtocol 示例 ====================

class SimpleAgent:
    """
    简单的可运行 Agent
    
    演示：
    - 实现 RunnableProtocol
    - 统一的运行接口
    """
    
    def __init__(self, name: str = "SimpleAgent"):
        self.name = name
    
    async def run(self, input: Any) -> str:
        """运行 Agent"""
        # 处理不同类型的输入
        if isinstance(input, str):
            query = input
        elif isinstance(input, dict):
            query = input.get("query", str(input))
        else:
            query = str(input)
        
        # 模拟处理
        await asyncio.sleep(0.1)
        
        return f"[{self.name}] Processed: {query}"


# ==================== 7. VectorStoreProtocol 示例 ====================

class SimpleVectorStore:
    """
    简单的向量存储（内存实现）
    
    演示：
    - 实现 VectorStoreProtocol
    - 余弦相似度搜索
    """
    
    def __init__(self):
        self._vectors: Dict[str, List[float]] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
    
    async def add(
        self,
        ids: List[str],
        vectors: List[List[float]],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """添加向量"""
        for i, vector_id in enumerate(ids):
            self._vectors[vector_id] = vectors[i]
            if metadata and i < len(metadata):
                self._metadata[vector_id] = metadata[i]
    
    async def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """检索相似向量"""
        import math
        
        def cosine_similarity(v1: List[float], v2: List[float]) -> float:
            """计算余弦相似度"""
            dot_product = sum(a * b for a, b in zip(v1, v2))
            magnitude1 = math.sqrt(sum(a * a for a in v1))
            magnitude2 = math.sqrt(sum(b * b for b in v2))
            
            if magnitude1 == 0 or magnitude2 == 0:
                return 0.0
            
            return dot_product / (magnitude1 * magnitude2)
        
        # 计算所有向量的相似度
        similarities = []
        for vector_id, vector in self._vectors.items():
            # 应用过滤器（如果有）
            if filters:
                metadata = self._metadata.get(vector_id, {})
                if not all(metadata.get(k) == v for k, v in filters.items()):
                    continue
            
            score = cosine_similarity(query_vector, vector)
            similarities.append({
                "id": vector_id,
                "score": score,
                "metadata": self._metadata.get(vector_id, {})
            })
        
        # 排序并返回 top_k
        similarities.sort(key=lambda x: x["score"], reverse=True)
        return similarities[:top_k]
    
    async def delete(self, ids: List[str]) -> None:
        """删除向量"""
        for vector_id in ids:
            self._vectors.pop(vector_id, None)
            self._metadata.pop(vector_id, None)
    
    async def update(
        self,
        ids: List[str],
        metadata: List[Dict[str, Any]],
    ) -> None:
        """更新元数据"""
        for i, vector_id in enumerate(ids):
            if vector_id in self._vectors and i < len(metadata):
                self._metadata[vector_id] = metadata[i]


# ==================== 演示函数 ====================

async def demo_model_protocol():
    """演示 ModelProtocol"""
    print("\n" + "=" * 60)
    print("1. ModelProtocol 演示")
    print("=" * 60)
    
    # 创建模型
    model = SimpleModel()
    
    # 验证协议
    print(f"✓ 实现了 ModelProtocol: {check_protocol(model, ModelProtocol)}")
    print(f"✓ 支持流式: {supports_streaming(model)}")
    print(f"✓ 模型名称: {get_model_name(model)}")
    
    # 调用模型
    messages = [
        {"role": "user", "content": "Hello, world!"}
    ]
    
    response = await model.acompletion(messages)
    print(f"\n请求: {messages[0]['content']}")
    print(f"响应: {response.choices[0].message['content']}")
    print(f"Token 使用: {response.usage.total_tokens}")


async def demo_streaming_model():
    """演示 StreamableModelProtocol"""
    print("\n" + "=" * 60)
    print("2. StreamableModelProtocol 演示")
    print("=" * 60)
    
    model = StreamingModel()
    
    print(f"✓ 实现了 StreamableModelProtocol: {check_protocol(model, StreamableModelProtocol)}")
    print(f"✓ 支持流式: {supports_streaming(model)}")
    print(f"✓ 支持函数调用: {supports_function_calling(model)}")
    
    # 流式调用
    messages = [{"role": "user", "content": "Tell me a story"}]
    
    print("\n流式输出: ", end="")
    async for chunk in model.astream(messages):
        if chunk.content:
            print(chunk.content, end="", flush=True)
    print()


async def demo_storage_protocol():
    """演示 StorageProtocol"""
    print("\n" + "=" * 60)
    print("3. StorageProtocol 演示")
    print("=" * 60)
    
    storage = MemoryStorage()
    
    print(f"✓ 实现了 StorageProtocol: {check_protocol(storage, StorageProtocol)}")
    validate_storage(storage)
    
    # 存储数据
    await storage.set("user:123", {"name": "Alice", "age": 25})
    print("\n✓ 存储数据: user:123")
    
    # 获取数据
    data = await storage.get("user:123")
    print(f"✓ 获取数据: {data}")
    
    # 带 TTL 的存储
    await storage.set("temp:session", {"token": "abc123"}, ttl=2)
    print("\n✓ 存储临时数据 (TTL: 2秒)")
    
    print("✓ 立即获取:", await storage.get("temp:session"))
    
    print("✓ 等待 3 秒...")
    await asyncio.sleep(3)
    
    print("✓ 再次获取:", await storage.get("temp:session"))


async def demo_tool_protocol():
    """演示 ToolProtocol"""
    print("\n" + "=" * 60)
    print("4. ToolProtocol 演示")
    print("=" * 60)
    
    # 计算器工具
    calc = CalculatorTool()
    
    print(f"✓ 工具名称: {calc.name}")
    print(f"✓ 工具描述: {calc.description}")
    print(f"✓ 实现了 ToolProtocol: {check_protocol(calc, ToolProtocol)}")
    
    validate_tool(calc)
    
    # 执行工具
    result1 = await calc.execute({"expression": "2 + 2"})
    print(f"\n计算 '2 + 2': {result1}")
    
    result2 = await calc.execute({"expression": "10 * 5 + 3"})
    print(f"计算 '10 * 5 + 3': {result2}")
    
    # 搜索工具
    print("\n" + "-" * 60)
    search = WebSearchTool()
    
    print(f"✓ 工具名称: {search.name}")
    validate_tool(search)
    
    result = await search.execute({
        "query": "Python asyncio",
        "max_results": 3,
        "language": "en"
    })
    print(f"\n搜索结果:\n{result}")


async def demo_embedder_protocol():
    """演示 EmbedderProtocol"""
    print("\n" + "=" * 60)
    print("5. EmbedderProtocol 演示")
    print("=" * 60)
    
    embedder = SimpleEmbedder(dimension=8)  # 使用小维度便于展示
    
    print(f"✓ 实现了 EmbedderProtocol: {check_protocol(embedder, EmbedderProtocol)}")
    print(f"✓ 向量维度: {embedder.get_dimension()}")
    
    # 嵌入单个文本
    text = "Hello, world!"
    embedding = await embedder.embed_single(text)
    print(f"\n文本: '{text}'")
    print(f"向量 (前5维): {[f'{x:.4f}' for x in embedding[:5]]}")
    
    # 批量嵌入
    texts = ["Python", "JavaScript", "Rust"]
    embeddings = await embedder.embed(texts)
    print(f"\n批量嵌入 {len(texts)} 个文本:")
    for text, emb in zip(texts, embeddings):
        print(f"  '{text}': {[f'{x:.4f}' for x in emb[:5]]}")


async def demo_runnable_protocol():
    """演示 RunnableProtocol"""
    print("\n" + "=" * 60)
    print("6. RunnableProtocol 演示")
    print("=" * 60)
    
    agent = SimpleAgent(name="DemoAgent")
    
    print(f"✓ 实现了 RunnableProtocol: {check_protocol(agent, RunnableProtocol)}")
    
    # 不同类型的输入
    result1 = await agent.run("What is AI?")
    print(f"\n字符串输入: {result1}")
    
    result2 = await agent.run({"query": "Explain Python", "context": "beginner"})
    print(f"字典输入: {result2}")


async def demo_vector_store_protocol():
    """演示 VectorStoreProtocol"""
    print("\n" + "=" * 60)
    print("7. VectorStoreProtocol 演示")
    print("=" * 60)
    
    store = SimpleVectorStore()
    embedder = SimpleEmbedder(dimension=8)
    
    print(f"✓ 实现了 VectorStoreProtocol: {check_protocol(store, VectorStoreProtocol)}")
    
    # 准备文档
    documents = [
        "Python is a programming language",
        "JavaScript is used for web development",
        "Machine learning is a subset of AI",
        "Deep learning uses neural networks"
    ]
    
    # 生成嵌入
    print(f"\n添加 {len(documents)} 个文档到向量库...")
    embeddings = await embedder.embed(documents)
    
    await store.add(
        ids=[f"doc_{i}" for i in range(len(documents))],
        vectors=embeddings,
        metadata=[{"text": doc, "index": i} for i, doc in enumerate(documents)]
    )
    
    # 搜索
    query = "What is Python?"
    query_embedding = await embedder.embed_single(query)
    
    print(f"\n查询: '{query}'")
    results = await store.search(query_embedding, top_k=2)
    
    print("\n最相似的文档:")
    for i, result in enumerate(results, 1):
        print(f"  {i}. {result['metadata']['text']} (相似度: {result['score']:.4f})")


async def demo_protocol_validation():
    """演示协议验证"""
    print("\n" + "=" * 60)
    print("8. 协议验证演示")
    print("=" * 60)
    
    # 有效的模型
    valid_model = SimpleModel()
    try:
        validate_model(valid_model)
        print("✓ 有效模型验证通过")
    except Exception as e:
        print(f"✗ 验证失败: {e}")
    
    # 无效的模型
    class InvalidModel:
        pass
    
    invalid_model = InvalidModel()
    try:
        validate_model(invalid_model)
        print("✗ 无效模型应该验证失败")
    except TypeError as e:
        print(f"✓ 无效模型正确拒绝: {str(e)[:50]}...")
    
    # 无效的工具（缺少属性）
    class InvalidTool:
        async def execute(self, arguments):
            return "result"
    
    invalid_tool = InvalidTool()
    try:
        validate_tool(invalid_tool)
        print("✗ 无效工具应该验证失败")
    except ValueError as e:
        print(f"✓ 无效工具正确拒绝: {str(e)[:50]}...")


# ==================== 主函数 ====================

async def main():
    """运行所有演示"""
    print("\n" + "=" * 60)
    print("Gecko Protocol 演示")
    print("=" * 60)
    print("\n本演示将展示如何实现和使用 Gecko 框架中的各种协议。")
    
    # 运行所有演示
    await demo_model_protocol()
    await demo_streaming_model()
    await demo_storage_protocol()
    await demo_tool_protocol()
    await demo_embedder_protocol()
    await demo_runnable_protocol()
    await demo_vector_store_protocol()
    await demo_protocol_validation()
    
    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)
    print("\n提示:")
    print("- 所有协议都使用 Protocol 定义，支持鸭子类型")
    print("- 使用 check_protocol() 进行运行时类型检查")
    print("- 使用 validate_*() 函数进行完整验证")
    print("- 实现协议时只需实现必需的方法和属性")
    print("\n详细文档请参考: gecko/core/protocols.py")


if __name__ == "__main__":
    # 运行演示
    asyncio.run(main())