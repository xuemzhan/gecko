# gecko/cli/commands/tools.py
from __future__ import annotations
import click
from gecko.cli.utils import print_table, print_error, print_info

@click.command()
@click.option("--verbose", "-v", is_flag=True, help="显示工具参数 Schema")
def tools(verbose: bool):
    """列出 Gecko 中已注册的所有工具。"""
    try:
        from gecko.plugins.tools.registry import ToolRegistry
        
        # 尝试触发标准工具的注册（确保内置工具可见）
        try:
            import gecko.plugins.tools.standard  # noqa
        except ImportError:
            pass
            
        tool_names = ToolRegistry.list_tools()
        
        if not tool_names:
            print_info("当前未注册任何工具。")
            return

        rows = []
        for name in tool_names:
            try:
                # 获取工具类
                tool_cls = ToolRegistry._registry.get(name)
                if not tool_cls:
                    continue
                
                # 获取描述（优先从类属性获取，避免实例化开销）
                # 注意：BaseTool 的 description 是 Pydantic 字段，通常定义在类上或由 Default 填充
                # 这里假设工具类定义了 description 属性或 annotation
                desc = getattr(tool_cls, "description", "")
                # 如果是 FieldInfo, 尝试获取 default
                if not isinstance(desc, str):
                     # 尝试实例化一个空的（如果允许无参）或者从 model_fields 获取
                     if hasattr(tool_cls, "model_fields") and "description" in tool_cls.model_fields:
                         desc = tool_cls.model_fields["description"].description or tool_cls.model_fields["description"].default
                         if not isinstance(desc, str):
                             desc = "No description available"
                
                args_desc = ""
                if verbose and hasattr(tool_cls, "args_schema"):
                    # 获取参数 schema
                    schema = tool_cls.args_schema.model_json_schema()
                    props = schema.get("properties", {})
                    args_desc = ", ".join(props.keys())

                row = [name, str(desc)[:60] + ("..." if len(str(desc)) > 60 else "")]
                if verbose:
                    row.append(args_desc)
                rows.append(row)
                
            except Exception:
                rows.append([name, "Error loading details"])

        cols = ["Name", "Description"]
        if verbose:
            cols.append("Arguments")
            
        print_table("Available Tools", cols, rows)

    except Exception as e:
        print_error(f"获取工具列表失败: {e}")