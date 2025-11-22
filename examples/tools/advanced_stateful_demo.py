# examples/tools/advanced_stateful_demo.py
"""
高级工具使用示例：有状态工具与复杂参数

本示例展示：
1. Stateful Tools: 如何在工具中维护状态（模拟数据库）。
2. Complex Schema: 如何使用嵌套的 Pydantic 模型作为工具参数。
3. Dependency Injection: 如何在工具初始化时注入外部依赖。
"""
import asyncio
import os
import json
from typing import Dict, List, Optional, Type

from pydantic import BaseModel, Field

from gecko.core.agent import Agent
from gecko.core.builder import AgentBuilder
# [重要] 适配新版模型架构
from gecko.plugins.models import ZhipuChat
from gecko.plugins.tools.base import BaseTool, ToolResult

# ==========================================
# 1. 模拟外部依赖 (Mock Database)
# ==========================================

class OrderDatabase:
    """一个简单的内存数据库，用于存储订单"""
    def __init__(self):
        self._orders: Dict[str, Dict] = {}
        self._lock = asyncio.Lock()

    async def add_order(self, order_id: str, items: List[Dict], total: float):
        async with self._lock:
            self._orders[order_id] = {
                "items": items,
                "total": total,
                "status": "pending"
            }
            print(f"\n[Database] 💾 Order {order_id} saved. Items: {len(items)}")

    async def get_order(self, order_id: str) -> Optional[Dict]:
        return self._orders.get(order_id)

# ==========================================
# 2. 定义复杂参数结构 (Complex Schema)
# ==========================================

class OrderItem(BaseModel):
    product_name: str = Field(..., description="商品名称")
    quantity: int = Field(..., description="数量", ge=1)
    price: float = Field(..., description="单价")

class PlaceOrderArgs(BaseModel):
    user_id: str = Field(..., description="用户ID")
    items: List[OrderItem] = Field(..., description="订单项列表")
    notes: Optional[str] = Field(None, description="备注信息")

class QueryOrderArgs(BaseModel):
    order_id: str = Field(..., description="订单ID")

# ==========================================
# 3. 定义有状态工具 (Stateful Tools)
# ==========================================

class PlaceOrderTool(BaseTool):
    name: str = "place_order"
    description: str = "下订单工具。支持一次性购买多个商品。"
    args_schema: Type[BaseModel] = PlaceOrderArgs

    def __init__(self, db: OrderDatabase):
        # 必须显式调用 super().__init__ 以初始化 Pydantic 模型
        super().__init__() # type: ignore
        # 将数据库依赖注入为私有属性（不参与 Schema 生成）
        object.__setattr__(self, "_db", db)

    async def _run(self, args: PlaceOrderArgs) -> ToolResult: # type: ignore
        # 计算总价
        total = sum(item.quantity * item.price for item in args.items)
        
        # 生成订单ID (模拟)
        import uuid
        order_id = f"ORD-{uuid.uuid4().hex[:6].upper()}"
        
        # 写入数据库
        items_dict = [item.model_dump() for item in args.items]
        await self._db.add_order(order_id, items_dict, total) # type: ignore
        
        return ToolResult(
            content=json.dumps({
                "status": "success",
                "order_id": order_id,
                "total_price": total,
                "message": "订单创建成功"
            }, ensure_ascii=False)
        )

class QueryOrderTool(BaseTool):
    name: str = "query_order"
    description: str = "查询订单状态和详情。"
    args_schema: Type[BaseModel] = QueryOrderArgs

    def __init__(self, db: OrderDatabase):
        super().__init__() # type: ignore
        object.__setattr__(self, "_db", db)

    async def _run(self, args: QueryOrderArgs) -> ToolResult: # type: ignore
        order = await self._db.get_order(args.order_id) # type: ignore
        
        if not order:
            return ToolResult(content="未找到该订单", is_error=True)
            
        return ToolResult(content=json.dumps(order, ensure_ascii=False))

# ==========================================
# 4. 主流程
# ==========================================

async def main():
    print("🚀 Advanced Tool Demo: Stateful & Complex Schema\n")

    # 0. 准备 API Key
    api_key = os.environ.get("ZHIPU_API_KEY")
    if not api_key:
        print("请设置 ZHIPU_API_KEY 环境变量")
        return

    # 1. 初始化共享资源 (Dependency)
    db = OrderDatabase()

    # 2. 实例化工具并注入依赖
    # 注意：这里我们手动实例化工具，而不是通过字符串名称加载，
    # 因为我们需要传递 `db` 对象。
    tools = [
        PlaceOrderTool(db=db),
        QueryOrderTool(db=db)
    ]

    # 3. 构建 Agent
    model = ZhipuChat(api_key=api_key, model="glm-4-air", temperature=0.1)
    
    agent = (
        AgentBuilder()
        .with_model(model)
        .with_tools(tools) # 直接传递实例化好的工具列表
        .with_system_prompt("你是一个订单助手。请帮助用户下单或查询。")
        .build()
    )

    # 4. 场景演示
    
    # 场景 A: 复杂下单 (LLM 需要生成嵌套 JSON)
    prompt1 = "我要买两台 MacBook Pro (单价15000) 和 一个鼠标 (单价500)，用户ID是 user_888"
    print(f"👤 User: {prompt1}")
    
    response1 = await agent.run(prompt1)
    print(f"🤖 Agent: {response1.content}\n") # type: ignore
    
    # 场景 B: 基于上下文查询状态
    # LLM 需要从上一步的回复中提取 order_id
    prompt2 = "请帮我查一下刚刚那个订单的详情"
    print(f"👤 User: {prompt2}")
    
    response2 = await agent.run(prompt2)
    print(f"🤖 Agent: {response2.content}\n") # type: ignore

if __name__ == "__main__":
    asyncio.run(main())