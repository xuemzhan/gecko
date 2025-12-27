# gecko/cli/commands/run.py
from __future__ import annotations

import click
import json
import os
import sys
import importlib.util
from typing import Optional, Any

from gecko.cli.utils import async_cmd, print_error, print_info, print_panel, print_markdown

@click.command()
@click.argument("workflow_file", type=click.Path(exists=True, dir_okay=False))
@click.option("--input", "-i", help="输入数据 (JSON 字符串或 JSON 文件路径)")
@async_cmd
async def run(workflow_file: str, input: Optional[str]):
    """
    执行工作流定义文件。
    
    WORKFLOW_FILE 必须是一个 Python 脚本 (.py)，
    其中必须包含一个名为 `workflow` 的导出变量 (gecko.Workflow 实例)。
    """
    
    # 1. 解析输入数据
    input_data: Any = {}
    if input:
        if os.path.exists(input):
            try:
                with open(input, 'r', encoding='utf-8') as f:
                    input_data = json.load(f)
            except Exception as e:
                print_error(f"无法读取输入文件: {e}")
                return
        else:
            try:
                input_data = json.loads(input)
            except json.JSONDecodeError:
                # 如果不是 JSON，视为普通字符串输入
                input_data = input

    print_info(f"正在加载工作流: {workflow_file}")
    
    # 2. 动态加载 Python 文件
    try:
        # 将工作流文件所在目录加入 sys.path，以便它能导入同级模块
        file_path = os.path.abspath(workflow_file)
        file_dir = os.path.dirname(file_path)
        sys.path.insert(0, file_dir)

        spec = importlib.util.spec_from_file_location("custom_workflow_module", file_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # 检查约定：必须定义 `workflow` 变量
            if not hasattr(module, "workflow"):
                print_error(f"文件 {workflow_file} 未定义 'workflow' 变量。")
                return
            
            workflow_instance = getattr(module, "workflow")
            
            # 简单的类型鸭子类型检查
            if not hasattr(workflow_instance, "execute"):
                print_error("'workflow' 变量似乎不是有效的 Gecko Workflow 对象 (缺少 execute 方法)。")
                return
            
            # 3. 执行工作流
            print_panel("Starting Execution", style="green")
            
            try:
                # 假设 execute 是异步的
                result = await workflow_instance.execute(input_data)
                
                print_panel("Execution Result", style="green")
                
                # 尝试美化输出
                if isinstance(result, (dict, list)):
                    print_markdown(f"```json\n{json.dumps(result, indent=2, ensure_ascii=False)}\n```")
                else:
                    print_markdown(str(result))
                    
            except Exception as exec_err:
                print_error(f"工作流执行期间发生错误: {exec_err}")
                # 在调试模式下打印堆栈
                if os.getenv("GECKO_DEBUG"):
                    import traceback
                    traceback.print_exc()

        else:
            print_error("无法加载 Python 模块规范。")
            
    except Exception as e:
        print_error(f"加载工作流失败: {e}")
    finally:
        # 清理 path
        if sys.path[0] == file_dir: # type: ignore
            sys.path.pop(0)