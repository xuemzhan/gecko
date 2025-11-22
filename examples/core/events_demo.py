import asyncio
import time
from typing import Optional

# 导入核心组件
from gecko.core.events import EventBus, BaseEvent
from gecko.core.logging import get_logger

logger = get_logger(__name__)

# ==========================================
# 1. 定义自定义事件
# ==========================================

class UserLoginEvent(BaseEvent):
    """用户登录事件"""
    type: str = "user.login"
    
class OrderCreatedEvent(BaseEvent):
    """订单创建事件"""
    type: str = "order.created"

# ==========================================
# 2. 定义处理器 (Handlers)
# ==========================================

async def async_logger(event: BaseEvent):
    """异步日志处理器"""
    # 模拟耗时 I/O
    await asyncio.sleep(0.1)
    print(f"📝 [Async Logger] {event.type}: {event.data}")

def sync_metrics(event: BaseEvent):
    """同步指标统计处理器"""
    print(f"📊 [Sync Metrics] Counting event: {event.type}")

async def slow_processor(event: BaseEvent):
    """慢速处理器（用于演示后台任务等待）"""
    print(f"⏳ [Slow Proc] Start processing {event.type}...")
    await asyncio.sleep(1.0) # 模拟长任务
    print(f"✅ [Slow Proc] Finished {event.type}")

# ==========================================
# 3. 定义中间件 (Middleware)
# ==========================================

async def audit_middleware(event: BaseEvent) -> Optional[BaseEvent]:
    """审计中间件：给所有事件添加审计时间戳"""
    event.data["audit_ts"] = time.time()
    return event

async def spam_filter_middleware(event: BaseEvent) -> Optional[BaseEvent]:
    """垃圾过滤中间件：拦截包含 'spam' 的事件"""
    if event.data.get("is_spam"):
        print(f"🚫 [Middleware] Blocked spam event: {event.type}")
        return None  # 返回 None 拦截事件
    return event

# ==========================================
# 4. 主演示流程
# ==========================================

async def main():
    print("🚀 Starting EventBus Demo...\n")

    # 使用上下文管理器自动处理 shutdown
    async with EventBus() as bus:
        
        # --- 注册组件 ---
        print("1️⃣  Registering Handlers & Middleware")
        
        # 订阅特定事件
        bus.subscribe("user.login", async_logger)
        bus.subscribe("user.login", sync_metrics)
        
        # 订阅所有事件 (通配符)
        bus.subscribe("*", lambda e: print(f"👀 [Global Watcher] Saw {e.type}"))
        
        # 注册中间件
        bus.add_middleware(audit_middleware)
        bus.add_middleware(spam_filter_middleware)
        print("   Done.\n")

        # --- 场景 1: 正常发布 (等待模式) ---
        print("2️⃣  Publishing User Login (Wait=True)")
        login_event = UserLoginEvent(data={"user_id": 101, "ip": "127.0.0.1"})
        
        await bus.publish(login_event, wait=True)
        # 此时 async_logger 已经执行完毕
        print("   Event processing completed.\n")

        # --- 场景 2: 中间件拦截 ---
        print("3️⃣  Publishing Spam Event")
        spam_event = OrderCreatedEvent(data={"order_id": 999, "is_spam": True})
        
        await bus.publish(spam_event, wait=True)
        print("   Spam event published (should be blocked).\n")

        # --- 场景 3: 后台任务 (不等待) ---
        print("4️⃣  Publishing Slow Event (Wait=False)")
        
        # 临时订阅一个慢速任务
        bus.subscribe("order.created", slow_processor)
        
        order_event = OrderCreatedEvent(data={"order_id": 202, "amount": 50.0})
        
        # 这里不会阻塞 1秒，而是立即返回
        start_time = time.time()
        await bus.publish(order_event, wait=False)
        print(f"   Publish returned in {time.time() - start_time:.4f}s (Non-blocking)")
        print("   Main logic continues doing other work...\n")

    # --- 自动 Shutdown ---
    # 退出 async with 块时，会自动调用 shutdown(wait=True)
    # 这将等待上面的 slow_processor 执行完毕
    print("5️⃣  EventBus Shutdown")
    print("   Context manager exited. All background tasks should be finished now.")

if __name__ == "__main__":
    asyncio.run(main())