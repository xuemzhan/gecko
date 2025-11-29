# gecko/plugins/storage/pool.py
"""
连接池抽象层

提供统一的连接池管理接口，支持：
- 异步连接获取/释放
- 连接健康检查
- 自动重连
- 资源限制
"""
from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Deque, Generic, Optional, TypeVar

from gecko.core.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")  # 连接类型


@dataclass
class PoolConfig:
    """连接池配置"""
    min_size: int = 1
    max_size: int = 10
    max_idle_time: float = 300.0  # 空闲连接超时（秒）
    acquire_timeout: float = 30.0  # 获取连接超时
    health_check_interval: float = 60.0  # 健康检查间隔
    retry_on_failure: bool = True
    max_retries: int = 3


@dataclass
class PooledConnection(Generic[T]):
    """池化连接包装器"""
    connection: T
    created_at: float
    last_used_at: float
    use_count: int = 0
    is_healthy: bool = True


class ConnectionPool(ABC, Generic[T]):
    """
    异步连接池抽象基类
    
    子类需要实现：
    - _create_connection(): 创建新连接
    - _close_connection(): 关闭连接
    - _check_health(): 检查连接健康状态
    """
    
    def __init__(self, config: Optional[PoolConfig] = None):
        self.config = config or PoolConfig()
        
        self._pool: Deque[PooledConnection[T]] = deque()
        self._in_use: set[PooledConnection[T]] = set()
        self._lock = asyncio.Lock()
        self._available = asyncio.Semaphore(self.config.max_size)
        
        self._is_closed = False
        self._health_check_task: Optional[asyncio.Task] = None
        
        # 统计
        self._stats = {
            "connections_created": 0,
            "connections_closed": 0,
            "acquires": 0,
            "releases": 0,
            "health_checks": 0,
            "health_failures": 0,
        }
    
    @abstractmethod
    async def _create_connection(self) -> T:
        """创建新连接（子类实现）"""
        pass
    
    @abstractmethod
    async def _close_connection(self, conn: T) -> None:
        """关闭连接（子类实现）"""
        pass
    
    @abstractmethod
    async def _check_health(self, conn: T) -> bool:
        """检查连接健康状态（子类实现）"""
        pass
    
    async def initialize(self) -> None:
        """初始化连接池"""
        logger.info(
            "Initializing connection pool",
            min_size=self.config.min_size,
            max_size=self.config.max_size
        )
        
        # 创建最小数量的连接
        for _ in range(self.config.min_size):
            try:
                conn = await self._create_pooled_connection()
                self._pool.append(conn)
            except Exception as e:
                logger.error("Failed to create initial connection", error=str(e))
        
        # 启动健康检查任务
        if self.config.health_check_interval > 0:
            self._health_check_task = asyncio.create_task(self._health_check_loop())
    
    async def close(self) -> None:
        """关闭连接池"""
        self._is_closed = True
        
        # 取消健康检查
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        # 关闭所有连接
        async with self._lock:
            while self._pool:
                pooled = self._pool.popleft()
                await self._safe_close(pooled.connection)
            
            for pooled in self._in_use:
                await self._safe_close(pooled.connection)
            self._in_use.clear()
        
        logger.info("Connection pool closed", stats=self._stats)
    
    async def acquire(self) -> T:
        """
        获取连接
        
        返回:
            可用的连接
            
        异常:
            TimeoutError: 超时未能获取连接
            RuntimeError: 连接池已关闭
        """
        if self._is_closed:
            raise RuntimeError("Connection pool is closed")
        
        # 等待信号量（限制并发连接数）
        try:
            await asyncio.wait_for(
                self._available.acquire(),
                timeout=self.config.acquire_timeout
            )
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"Failed to acquire connection within {self.config.acquire_timeout}s"
            )
        
        try:
            pooled = await self._get_or_create_connection()
            
            async with self._lock:
                self._in_use.add(pooled)
            
            pooled.use_count += 1
            pooled.last_used_at = asyncio.get_event_loop().time()
            
            self._stats["acquires"] += 1
            
            return pooled.connection
            
        except Exception:
            self._available.release()
            raise
    
    async def release(self, conn: T) -> None:
        """释放连接"""
        pooled = None
        
        async with self._lock:
            for p in self._in_use:
                if p.connection is conn:
                    pooled = p
                    break
            
            if pooled:
                self._in_use.remove(pooled)
        
        if pooled:
            # 检查是否应该回收
            if self._should_recycle(pooled):
                await self._safe_close(pooled.connection)
            else:
                async with self._lock:
                    self._pool.append(pooled)
        
        self._available.release()
        self._stats["releases"] += 1
    
    @asynccontextmanager
    async def connection(self) -> AsyncIterator[T]:
        """
        上下文管理器方式获取连接
        
        示例:
            async with pool.connection() as conn:
                await conn.execute(...)
        """
        conn = await self.acquire()
        try:
            yield conn
        finally:
            await self.release(conn)
    
    async def _get_or_create_connection(self) -> PooledConnection[T]:
        """获取或创建连接"""
        async with self._lock:
            # 尝试从池中获取
            while self._pool:
                pooled = self._pool.popleft()
                
                # 检查是否过期
                if self._is_expired(pooled):
                    await self._safe_close(pooled.connection)
                    continue
                
                # 简单健康检查
                if pooled.is_healthy:
                    return pooled
                else:
                    await self._safe_close(pooled.connection)
        
        # 创建新连接
        return await self._create_pooled_connection()
    
    async def _create_pooled_connection(self) -> PooledConnection[T]:
        """创建池化连接"""
        conn = await self._create_connection()
        now = asyncio.get_event_loop().time()
        
        self._stats["connections_created"] += 1
        
        return PooledConnection(
            connection=conn,
            created_at=now,
            last_used_at=now,
        )
    
    async def _safe_close(self, conn: T) -> None:
        """安全关闭连接"""
        try:
            await self._close_connection(conn)
            self._stats["connections_closed"] += 1
        except Exception as e:
            logger.warning("Error closing connection", error=str(e))
    
    def _is_expired(self, pooled: PooledConnection[T]) -> bool:
        """检查连接是否过期"""
        now = asyncio.get_event_loop().time()
        idle_time = now - pooled.last_used_at
        return idle_time > self.config.max_idle_time
    
    def _should_recycle(self, pooled: PooledConnection[T]) -> bool:
        """判断是否应该回收连接"""
        # 不健康的连接回收
        if not pooled.is_healthy:
            return True
        
        # 池已满，回收
        if len(self._pool) >= self.config.max_size:
            return True
        
        return False
    
    async def _health_check_loop(self) -> None:
        """后台健康检查任务"""
        while not self._is_closed:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                await self._run_health_check()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Health check loop error", error=str(e))
    
    async def _run_health_check(self) -> None:
        """执行健康检查"""
        async with self._lock:
            connections_to_check = list(self._pool)
        
        for pooled in connections_to_check:
            self._stats["health_checks"] += 1
            
            try:
                is_healthy = await self._check_health(pooled.connection)
                pooled.is_healthy = is_healthy
                
                if not is_healthy:
                    self._stats["health_failures"] += 1
                    logger.warning("Connection health check failed")
                    
            except Exception as e:
                pooled.is_healthy = False
                self._stats["health_failures"] += 1
                logger.warning("Connection health check error", error=str(e))
    
    def get_stats(self) -> dict:
        """获取连接池统计"""
        return {
            **self._stats,
            "pool_size": len(self._pool),
            "in_use": len(self._in_use),
            "available": self.config.max_size - len(self._in_use),
        }


# ==================== 导出 ====================

__all__ = [
    "ConnectionPool",
    "PoolConfig",
    "PooledConnection",
]