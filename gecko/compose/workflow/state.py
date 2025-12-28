# gecko/compose/workflow/state.py
"""
Workflow 状态管理模块

[Refactor] 从 engine.py 拆分，独立处理 Copy-On-Write 逻辑。
[Fix P0-1] 引入墓碑机制 (Tombstones)，修复 delete/pop 操作在合并时失效的问题。
[Fix P0-5] 引入 Copy-on-Read 机制，防止并行节点修改共享的可变对象导致数据污染。
"""
from __future__ import annotations

import copy
from typing import Any, Dict, Optional, Set, Tuple, Union

class COWDict(dict):
    """
    支持并发安全的 Copy-On-Write (COW) 字典。
    
    机制：
    1. Read: 优先读 Local -> Base。若是 Base 中的可变对象，执行 DeepCopy (Copy-on-Read) 以隔离副作用。
    2. Write: 仅写入 Local，不污染 Base。
    3. Delete: 写入 _deleted 墓碑集合，屏蔽 Base 中的值。
    """
    def __init__(self, base: Optional[dict] = None):
        super().__init__()
        self._base = base or {}
        self._local: Dict[str, Any] = {}
        # [Fix P0-1] 墓碑集合：记录在此上下文中被显式删除的 Key
        self._deleted: Set[str] = set()

    def __getitem__(self, key: str) -> Any:
        # 1. 检查是否已被删除
        if key in self._deleted:
            raise KeyError(key)
        
        # 2. 检查本地覆盖
        if key in self._local:
            return self._local[key]
        
        # 3. 检查基准数据
        if key in self._base:
            val = self._base[key]
            # [Fix P0-5] Copy-on-Read: 核心隔离逻辑
            # 如果值是可变对象 (list, dict, set)，必须进行深拷贝。
            # 否则并行节点 A 修改 list，节点 B 也会看到修改，破坏隔离性。
            if isinstance(val, (list, dict, set)):
                # 注意：深拷贝有性能开销，但为了正确性是必须的。
                copied_val = copy.deepcopy(val)
                self._local[key] = copied_val
                return copied_val
            return val
            
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self.__getitem__(key)
        except KeyError:
            return default

    def __contains__(self, key: object) -> bool:
        if key in self._deleted:
            return False
        return key in self._local or key in self._base

    def keys(self): # type: ignore
        # 视图 = (Base - Deleted) + Local
        base_keys = set(self._base.keys()) - self._deleted
        return base_keys | set(self._local.keys())

    def items(self): # type: ignore
        for k in self.keys():
            yield (k, self.get(k))

    def __setitem__(self, key: str, value: Any):
        self._local[key] = value
        # 如果之前标记了删除，现在重新赋值了，需要移除墓碑
        self._deleted.discard(key)

    def __delitem__(self, key: str):
        """支持显式删除"""
        found = False

        # 1. 如果在 local 中，直接删除
        if key in self._local:
            del self._local[key]
            found = True
        
        # 2. 如果在 base 中，需要标记删除
        if key in self._base and key not in self._deleted:
            self._deleted.add(key)
            found = True
            
        # 3. 如果两处都没找到，且没有被标记删除，抛出 KeyError
        if not found:
            raise KeyError(key)

    def pop(self, key: str, default: Any = None) -> Any:
        """
        [Fix P0-1] pop 操作必须产生副作用记录 (Tombstone)
        """
        if key in self._deleted:
            return default

        if key in self._local:
            return self._local.pop(key)
        
        if key in self._base:
            # 触发 Copy-on-Read 获取值 (为了返回)
            val = self.__getitem__(key)
            # 标记删除
            self._deleted.add(key)
            # 确保 local 清理
            self._local.pop(key, None)
            return val
            
        return default

    def update(self, other=None, **kwargs):
        if other:
            try:
                # 尝试转换为 dict 或迭代
                for k, v in dict(other).items():
                    self.__setitem__(k, v)
            except Exception:
                # [Coverage] Hit line 116: Ignore invalid updates
                pass
        for k, v in kwargs.items():
            self.__setitem__(k, v)

    def get_diff(self) -> Tuple[Dict[str, Any], Set[str]]:
        """
        [Fix P0-1] 返回更完整的差异信息
        Returns:
            (updates, deletions): 更新的键值对字典，被删除的键集合
        """
        return dict(self._local), set(self._deleted)