import os
import sys
import pytest
import asyncio
import json
from unittest.mock import MagicMock, AsyncMock, patch
from click.testing import CliRunner

# 导入被测模块
import gecko.cli.utils
from gecko.cli import main as cli_entry_point
from gecko.cli.main import cli

# ============================================================================
# 1. 测试 gecko.cli.utils (工具库)
# ============================================================================

class TestUtils:
    def test_async_cmd_decorator(self):
        """测试异步命令包装器"""
        @gecko.cli.utils.async_cmd
        async def async_func(x):
            await asyncio.sleep(0.01)
            return x + 1

        assert async_func(1) == 2

    def test_print_functions_with_rich(self):
        """测试安装了 Rich 时的输出调用"""
        # 强制 RICH_AVAILABLE = True
        with patch("gecko.cli.utils.RICH_AVAILABLE", True):
            with patch("gecko.cli.utils.console") as mock_console:
                # Markdown
                gecko.cli.utils.print_markdown("# Title")
                assert mock_console.print.called
                
                # Panel
                gecko.cli.utils.print_panel("Text", title="Title")
                assert mock_console.print.called
                
                # Table
                gecko.cli.utils.print_table("T", ["Col"], [["Val"]])
                assert mock_console.print.called
                
                # Error/Info
                gecko.cli.utils.print_error("Err")
                mock_console.print.assert_called_with("[bold red]Error:[/bold red] Err")
                
                gecko.cli.utils.print_info("Inf")
                mock_console.print.assert_called_with("[bold blue]Info:[/bold blue] Inf")

                # Spinner
                with gecko.cli.utils.SpinnerContext("Loading"):
                    pass
                assert mock_console.status.called

    def test_print_functions_without_rich(self):
        """测试未安装 Rich 时的降级输出 (click.echo)"""
        # 强制 RICH_AVAILABLE = False
        with patch("gecko.cli.utils.RICH_AVAILABLE", False):
            with patch("click.echo") as mock_echo, patch("click.secho") as mock_secho:
                # Markdown
                gecko.cli.utils.print_markdown("# Title")
                mock_echo.assert_called_with("# Title")
                
                # Panel
                gecko.cli.utils.print_panel("Text", title="Title")
                assert mock_echo.call_count >= 2 # Title line, Content, Separator
                
                # Table
                gecko.cli.utils.print_table("T", ["Col"], [["Val"]])
                # 验证是否打印了表头和内容
                args_list = [c[0][0] for c in mock_echo.call_args_list]
                assert any("Col" in str(arg) for arg in args_list)
                assert any("Val" in str(arg) for arg in args_list)
                
                # Error
                gecko.cli.utils.print_error("Err")
                mock_secho.assert_called_with("Error: Err", fg="red", err=True)
                
                # Info
                gecko.cli.utils.print_info("Inf")
                mock_secho.assert_called_with("Info: Inf", fg="blue")

                # Spinner
                with patch("sys.stdout.flush"):
                    with gecko.cli.utils.SpinnerContext("Loading"):
                        pass
                assert any("Loading..." in str(c) for c in mock_echo.call_args_list)

# ============================================================================
# 2. 测试 gecko.cli.__init__ (入口点异常处理)
# ============================================================================

class TestEntryPoint:
    def test_main_success(self):
        """测试正常启动"""
        # [Fix] Patch 'gecko.cli.cli' 而不是 'gecko.cli.main.cli'
        # 因为 gecko/cli/__init__.py 已经导入了 cli 对象
        with patch("gecko.cli.cli") as mock_cli:
            cli_entry_point()
            mock_cli.assert_called_once()

    def test_main_critical_error_production(self):
        """测试生产环境下的致命错误 (无 Traceback)"""
        # [Fix] Patch 正确的引用路径
        with patch("gecko.cli.cli", side_effect=Exception("Boom")):
            with patch.dict(os.environ, {"GECKO_DEBUG": "0"}):
                with pytest.raises(SystemExit) as excinfo:
                    cli_entry_point()
                assert excinfo.value.code == 1

    def test_main_critical_error_debug(self):
        """测试调试环境下的致命错误 (抛出异常)"""
        with patch("gecko.cli.cli", side_effect=ValueError("Boom")):
            with patch.dict(os.environ, {"GECKO_DEBUG": "1"}):
                with pytest.raises(ValueError, match="Boom"):
                    cli_entry_point()

# ============================================================================
# 3. 测试 gecko.cli.commands.config
# ============================================================================

class TestConfigCommand:
    def test_config_display(self):
        runner = CliRunner()
        
        # 模拟 Settings 对象
        mock_settings = MagicMock()
        mock_settings.model_dump.return_value = {
            "default_model": "gpt-4",
            "openai_api_key": "sk-123456",
            "secret_token": "secret"
        }
        
        with patch("gecko.config.get_settings", return_value=mock_settings):
            result = runner.invoke(cli, ["config"])
            
            assert result.exit_code == 0
            assert "gpt-4" in result.output
            # 验证脱敏
            assert "sk-123456" not in result.output 
            assert "********" in result.output # Key masked
            assert "Current Configuration" in result.output

    def test_config_error(self):
        runner = CliRunner()
        with patch("gecko.config.get_settings", side_effect=Exception("Config Load Fail")):
            result = runner.invoke(cli, ["config"])
            assert "无法加载配置: Config Load Fail" in result.output

# ============================================================================
# 4. 测试 gecko.cli.commands.tools
# ============================================================================

class TestToolsCommand:
    def test_tools_list_empty(self):
        runner = CliRunner()
        with patch("gecko.plugins.tools.registry.ToolRegistry.list_tools", return_value=[]):
            result = runner.invoke(cli, ["tools"])
            assert "当前未注册任何工具" in result.output

    def test_tools_list_success(self):
        runner = CliRunner()
        
        # 模拟工具类
        class MockTool:
            description = "A mock tool"
            
        with patch("gecko.plugins.tools.registry.ToolRegistry.list_tools", return_value=["mock_tool"]):
            with patch("gecko.plugins.tools.registry.ToolRegistry._registry", {"mock_tool": MockTool}):
                result = runner.invoke(cli, ["tools"])
                assert "mock_tool" in result.output
                assert "A mock tool" in result.output

    def test_tools_list_verbose_and_error(self):
        runner = CliRunner()
        
        # 模拟一个正常工具和一个加载错误的工具
        class GoodTool:
            description = "Good"
            class Args:
                @staticmethod
                def model_json_schema():
                    return {"properties": {"arg1": {}}}
            args_schema = Args
            
        with patch("gecko.plugins.tools.registry.ToolRegistry.list_tools", return_value=["good", "bad"]):
            # 模拟 Registry.get 行为
            def get_side_effect(name):
                if name == "good": return GoodTool
                if name == "bad": raise Exception("Load Error")
                return None

            with patch("gecko.plugins.tools.registry.ToolRegistry._registry") as mock_reg:
                mock_reg.get.side_effect = get_side_effect
                
                result = runner.invoke(cli, ["tools", "-v"])
                
                assert "good" in result.output
                assert "arg1" in result.output # verbose check
                assert "bad" in result.output
                assert "Error loading details" in result.output

    def test_tools_exception(self):
        runner = CliRunner()
        with patch("gecko.plugins.tools.registry.ToolRegistry.list_tools", side_effect=Exception("Reg Error")):
            result = runner.invoke(cli, ["tools"])
            assert "获取工具列表失败: Reg Error" in result.output

# ============================================================================
# 5. 测试 gecko.cli.commands.chat
# ============================================================================

class TestChatCommand:
    def test_chat_import_error(self):
        runner = CliRunner()
        # 模拟导入失败
        with patch.dict(sys.modules, {"gecko": None}):
            result = runner.invoke(cli, ["chat"])
            assert "无法导入 Gecko 组件" in result.output

    def test_chat_missing_api_key(self):
        runner = CliRunner()
        # 确保环境变量为空
        with patch.dict(os.environ, clear=True):
            result = runner.invoke(cli, ["chat", "--api-key", ""])
            assert "未提供 API Key" in result.output

    def test_chat_init_error(self):
        runner = CliRunner()
        with patch("gecko.plugins.models.OpenAIChat", side_effect=Exception("Init Fail")):
            result = runner.invoke(cli, ["chat", "--api-key", "dummy"])
            assert "Agent 初始化失败: Init Fail" in result.output

    def test_chat_ollama(self):
        """测试 Ollama 路径 (无需 API Key)"""
        runner = CliRunner()
        
        mock_agent = AsyncMock()
        mock_agent.run.return_value = "Hello from Ollama"
        
        mock_builder = MagicMock()
        mock_builder.with_model.return_value = mock_builder
        mock_builder.build.return_value = mock_agent

        # 修正：Patch 正确的类路径
        with patch("gecko.plugins.models.OllamaChat") as MockOllama:
            # 修正：AgentBuilder 通常直接从 gecko 导入
            with patch("gecko.AgentBuilder", return_value=mock_builder):
                # 输入 exit 退出循环
                result = runner.invoke(cli, ["chat", "-m", "ollama/llama3"], input="hi\nexit\n")
                
                assert MockOllama.called
                assert "Gecko Chat Session" in result.output
                assert "Hello from Ollama" in result.output

    def test_chat_interaction_loop(self):
        """测试完整的对话交互"""
        runner = CliRunner()
        
        # 模拟 Agent 输出
        from gecko.core.output import AgentOutput
        output_obj = AgentOutput(
            content="Response Content", 
            tool_calls=[{"function": {"name": "test_tool"}}]
        )
        
        mock_agent = AsyncMock()
        mock_agent.run.return_value = output_obj
        
        mock_builder = MagicMock()
        mock_builder.with_model.return_value = mock_builder
        mock_builder.with_system_prompt.return_value = mock_builder
        mock_builder.build.return_value = mock_agent

        with patch("gecko.plugins.models.OpenAIChat"), \
             patch("gecko.AgentBuilder", return_value=mock_builder):
            
            # 第一次输入 hello，第二次输入 quit
            result = runner.invoke(
                cli, 
                ["chat", "--api-key", "sk-test", "--system", "sys"], 
                input="hello\nquit\n"
            )
            
            assert "System: sys" in result.output
            assert "Response Content" in result.output
            assert "test_tool" in result.output # Check tool call log
            assert "Goodbye!" in result.output

    def test_chat_runtime_error(self):
        runner = CliRunner()
        
        # 1. 配置 Agent Mock
        # run 方法本身需要是一个 AsyncMock，以便被 await
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(side_effect=Exception("Run Fail"))
    
        # 2. 配置 Builder Mock
        mock_builder = MagicMock()
        # [关键修复] 配置链式调用返回自身 (Fluent Interface)
        mock_builder.with_model.return_value = mock_builder
        mock_builder.with_system_prompt.return_value = mock_builder
        # 配置最终 build 返回 mock_agent
        mock_builder.build.return_value = mock_agent
    
        with patch("gecko.plugins.models.OpenAIChat"), \
             patch("gecko.AgentBuilder", return_value=mock_builder):
             
             # 模拟用户输入：先打招呼触发错误，然后退出
             result = runner.invoke(cli, ["chat", "--api-key", "sk-test"], input="hi\nexit\n")
             
             # 验证输出
             # 注意：click 的 output 可能会包含换行符或格式，使用 in 判断即可
             assert "运行时错误: Run Fail" in result.output

# ============================================================================
# 6. 测试 gecko.cli.commands.run
# ============================================================================

class TestRunCommand:
    @pytest.fixture
    def mock_workflow_file(self, tmp_path):
        """创建一个临时的 workflow.py 文件"""
        f = tmp_path / "my_workflow.py"
        content = """
class MockWorkflow:
    async def execute(self, input):
        return {"status": "ok", "input": input}

workflow = MockWorkflow()
"""
        f.write_text(content, encoding="utf-8")
        return str(f)

    def test_run_file_not_found(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "non_existent.py"])
        assert result.exit_code != 0
        assert "Invalid value for 'WORKFLOW_FILE'" in result.output

    def test_run_input_file_fail(self):
        runner = CliRunner()
        # 创建一个空 workflow 文件以便通过参数检查
        with runner.isolated_filesystem():
            with open("dummy.py", "w") as f: f.write("workflow=None")
            # 创建一个无效的 JSON 文件来触发读取/解析错误
            with open("broken.json", "w") as f: f.write("{invalid")
            
            result = runner.invoke(cli, ["run", "dummy.py", "-i", "broken.json"])
            
            # [Fix] 验证正确的错误信息
            # 当文件读取成功但 JSON 解析失败时，代码会捕获异常并打印错误
            assert "无法读取输入文件" in result.output

    def test_run_success(self, mock_workflow_file):
        runner = CliRunner()
        
        # 使用 input JSON 字符串
        result = runner.invoke(cli, ["run", mock_workflow_file, "-i", '{"key": "val"}'])
        
        assert "正在加载工作流" in result.output
        assert "Starting Execution" in result.output
        assert '"status": "ok"' in result.output
        assert '"key": "val"' in result.output

    def test_run_invalid_workflow_file(self, tmp_path):
        """测试没有定义 workflow 变量的文件"""
        f = tmp_path / "bad.py"
        f.write_text("x = 1", encoding="utf-8")
        
        runner = CliRunner()
        result = runner.invoke(cli, ["run", str(f)])
        assert "未定义 'workflow' 变量" in result.output

    def test_run_invalid_workflow_object(self, tmp_path):
        """测试 workflow 对象缺少 execute 方法"""
        f = tmp_path / "bad_obj.py"
        f.write_text("class W: pass\nworkflow = W()", encoding="utf-8")
        
        runner = CliRunner()
        result = runner.invoke(cli, ["run", str(f)])
        assert "缺少 execute 方法" in result.output

    def test_run_execution_exception(self, tmp_path):
        """测试工作流执行报错"""
        f = tmp_path / "crash.py"
        content = """
class CrashW:
    async def execute(self, i): raise ValueError("Crash")
workflow = CrashW()
"""
        f.write_text(content, encoding="utf-8")
        
        runner = CliRunner()
        # 模拟 DEBUG 环境变量
        with patch.dict(os.environ, {"GECKO_DEBUG": "1"}):
            result = runner.invoke(cli, ["run", str(f)])
            assert "工作流执行期间发生错误: Crash" in result.output
            assert "Traceback" in result.output # Debug mode should show traceback

if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))