# gecko/cli/commands/config.py
from __future__ import annotations
import click
from gecko.cli.utils import print_table, print_error

@click.command()
def config():
    """显示当前的 Gecko 全局配置信息。"""
    try:
        # 延迟导入
        from gecko.config import get_settings
        settings = get_settings()
        
        rows = []
        # 将 pydantic settings 转为字典
        data = settings.model_dump()
        
        for field, value in data.items():
            # [安全] 敏感信息脱敏处理
            key_lower = field.lower()
            if any(s in key_lower for s in ["key", "secret", "password", "token"]):
                if value:
                    # 显示前3后3位，中间打码
                    val_str = str(value)
                    if len(val_str) > 8:
                        value = val_str[:3] + "****" + val_str[-3:]
                    else:
                        value = "********"
                else:
                    value = "<Not Set>"
            
            rows.append([field, str(value)])
            
        print_table("Current Configuration", ["Key", "Value"], rows)
        
    except Exception as e:
        print_error(f"无法加载配置: {e}")