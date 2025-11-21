# gecko/core/protocols.py  
"""  
模型协议定义（扩展版）  
  
- 基础模型需实现 acompletion  
- 如支持流式输出，还应实现 astream  
"""  
  
from __future__ import annotations  
  
from typing import Any, AsyncIterator, Dict, List, Protocol, runtime_checkable  
  
  
@runtime_checkable  
class ModelProtocol(Protocol):  
    async def acompletion(self, messages: List[Dict[str, Any]], **kwargs) -> Any:  
        ...  
  
    async def astream(self, messages: List[Dict[str, Any]], **kwargs) -> AsyncIterator[Any]:  
        """  
        可选：支持流式输出的模型应实现此方法  
        """  
        ...  
