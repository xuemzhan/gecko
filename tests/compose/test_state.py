# tests/compose/test_state.py
"""
Tests for the specialized COWDict in gecko.compose.workflow.state

Coverage Targets:
- Copy-on-Read (Deepcopy on access for mutables)
- Tombstone mechanism (Handling deletions)
- View consistency (keys, contains)
"""
import pytest
from gecko.compose.workflow.state import COWDict

class TestCOWDict:
    
    def test_basic_read_write(self):
        base = {"a": 1, "b": 2}
        cow = COWDict(base)
        
        # Read from base
        assert cow["a"] == 1
        
        # Write local
        cow["c"] = 3
        assert cow["c"] == 3
        assert "c" not in base
        
        # Overwrite base
        cow["b"] = 20
        assert cow["b"] == 20
        assert base["b"] == 2
        
    def test_copy_on_read_isolation(self):
        """[Fix P0-5] 验证读取可变对象时触发深拷贝"""
        base_list = [1, 2, 3]
        base_dict = {"inner": "val"}
        base = {"l": base_list, "d": base_dict}
        
        cow = COWDict(base)
        
        # 1. Accessing list triggers copy
        l = cow["l"]
        assert l == [1, 2, 3]
        assert l is not base_list # Must be a new object
        
        # Mutate local copy
        l.append(4)
        assert cow["l"] == [1, 2, 3, 4]
        assert base["l"] == [1, 2, 3] # Base remains untouched
        
        # 2. Accessing dict triggers copy
        d = cow["d"]
        d["new"] = 1
        assert base["d"] == {"inner": "val"}
        
    def test_tombstone_deletion(self):
        """[Fix P0-1] 验证墓碑机制"""
        base = {"a": 1, "b": 2}
        cow = COWDict(base)
        
        # Delete key existing in base
        del cow["a"]
        
        # Assertions
        assert "a" not in cow
        with pytest.raises(KeyError):
            _ = cow["a"]
            
        # Verify internal state
        assert "a" in cow._deleted
        
        # Pop
        val = cow.pop("b")
        assert val == 2
        assert "b" not in cow
        assert "b" in cow._deleted
        
    def test_resurrection(self):
        """验证删除后重新赋值（复活）"""
        base = {"a": 1}
        cow = COWDict(base)
        
        del cow["a"]
        assert "a" not in cow
        
        cow["a"] = 100
        assert cow["a"] == 100
        assert "a" not in cow._deleted
        
    def test_keys_view_consistency(self):
        """验证 keys() 视图过滤掉已删除的键"""
        base = {"a": 1, "b": 2}
        cow = COWDict(base)
        
        del cow["a"]
        cow["c"] = 3
        
        keys = set(cow.keys())
        assert "a" not in keys
        assert "b" in keys
        assert "c" in keys
        
    def test_get_diff(self):
        """验证 diff 生成正确"""
        base = {"a": 1, "b": 2}
        cow = COWDict(base)
        
        cow["b"] = 20 # Update
        cow["c"] = 30 # New
        del cow["a"]  # Delete
        
        updates, deletions = cow.get_diff()
        
        assert updates == {"b": 20, "c": 30}
        assert deletions == {"a"}