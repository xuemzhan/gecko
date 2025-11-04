# agno/ui/display.py

"""
UI 展示与命令行打印模块

该模块使用 `rich` 库为 Agent 和 Team 的运行结果提供美观的命令行界面展示。
它整合了之前分散在多个文件中的打印逻辑，提供了一个统一的、功能丰富的
UI 渲染引擎。

主要功能:
- 支持 Agent 和 Team 的同步/异步、流式/非流式结果打印。
- 以结构化的 Panel 形式展示消息、思维过程、工具调用、成员响应等。
- 优雅地处理流式输出，动态更新界面。
"""

import json
from typing import (
    TYPE_CHECKING, Any, AsyncIterable, Dict, Iterable, List, Optional, Set, 
    Union, cast, get_args
)

from pydantic import BaseModel
from rich.console import Console, Group
from rich.json import JSON
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.status import Status
from rich.text import Text

# 动态导入以避免循环依赖
try:
    from agno.models.response import ToolExecution
    from agno.reasoning.step import ReasoningStep
    from agno.run.agent import RunEvent, RunOutput, RunOutputEvent, RunPausedEvent
    from agno.run.team import TeamRunOutput, TeamRunOutputEvent
    from agno.utils.message import get_text_from_message
    from agno.utils.timer import Timer
except ImportError:
    # 定义桩类以供独立测试
    class RunOutput: pass
    class ToolExecution: pass
    class ReasoningStep: pass
    class RunOutputEvent: pass
    class RunPausedEvent: pass
    class TeamRunOutput: pass
    class TeamRunOutputEvent: pass
    class Timer: def elapsed(self): return 0.0
    def get_text_from_message(m): return str(m)

if TYPE_CHECKING:
    from agno.agent import Agent
    from agno.team import Team

# 假设日志模块已按规划重构
# from agno.utils.log import logger
# 在独立运行时，使用标准日志库
import logging
logger = logging.getLogger(__name__)


# --- 核心渲染函数 ---

CONSOLE = Console()

def display_run_output(
    run_output: Union[
        RunOutput, Iterable[RunOutputEvent],
        TeamRunOutput, Iterable[TeamRunOutputEvent],
    ],
    entity: Union["Agent", "Team"], # Agent 或 Team 实例
    show_reasoning: bool = True,
    show_full_reasoning: bool = False,
    markdown: bool = True,
    show_time: bool = True,
    **kwargs: Any # 保留以兼容旧的 print 函数签名
) -> None:
    """
    根据运行结果的类型（流式或非流式）分派到相应的渲染函数。
    """
    if isinstance(run_output, (RunOutput, TeamRunOutput)):
        _display_static_output(run_output, entity, show_reasoning, show_full_reasoning, markdown, show_time)
    else:
        _display_streaming_output(run_output, entity, show_reasoning, show_full_reasoning, markdown, show_time)

async def adisplay_run_output(
    run_output: Union[
        RunOutput, AsyncIterable[RunOutputEvent],
        TeamRunOutput, AsyncIterable[TeamRunOutputEvent],
    ],
    entity: Union["Agent", "Team"],
    show_reasoning: bool = True,
    show_full_reasoning: bool = False,
    markdown: bool = True,
    show_time: bool = True,
    **kwargs: Any
) -> None:
    """
    异步版本的 `display_run_output`。
    """
    if isinstance(run_output, (RunOutput, TeamRunOutput)):
        _display_static_output(run_output, entity, show_reasoning, show_full_reasoning, markdown, show_time)
    else:
        await _adisplay_streaming_output(run_output, entity, show_reasoning, show_full_reasoning, markdown, show_time)


# --- 静态输出渲染 ---

def _display_static_output(
    run_output: Union[RunOutput, TeamRunOutput],
    entity: Union["Agent", "Team"],
    show_reasoning: bool,
    show_full_reasoning: bool,
    markdown: bool,
    show_time: bool
) -> None:
    """为非流式的、已完成的 RunOutput 渲染最终界面。"""
    
    panels = []
    timer = Timer()
    if run_output.metrics and run_output.metrics.duration:
        timer.elapsed_time = run_output.metrics.duration
    
    # 1. 渲染输入消息
    if run_output.input and run_output.input.input_content:
        panels.append(_build_message_panel(run_output.input.input_content))

    # 2. 渲染推理过程
    if show_reasoning:
        panels.extend(_build_reasoning_panels(run_output, show_full_reasoning))

    # 3. 渲染团队成员交互 (仅对 TeamRunOutput)
    if isinstance(run_output, TeamRunOutput) and run_output.member_responses:
        panels.extend(_build_member_response_panels(run_output, entity, markdown))
        
    # 4. 渲染工具调用
    if run_output.tools:
        panels.append(_build_tool_calls_panel(run_output.tools, entity))

    # 5. 渲染最终响应
    panels.append(_build_response_panel(run_output, timer, markdown, show_time))
    
    # 6. 渲染引用
    if run_output.citations and run_output.citations.urls:
        panels.append(_build_citations_panel(run_output.citations))
        
    CONSOLE.print(Group(*panels))


# --- 流式输出渲染 ---

def _display_streaming_output(
    run_output_stream: Iterable[Union[RunOutputEvent, TeamRunOutputEvent]],
    entity: Union["Agent", "Team"],
    show_reasoning: bool,
    show_full_reasoning: bool,
    markdown: bool,
    show_time: bool
) -> None:
    """为流式输出动态渲染界面。"""
    with Live(console=CONSOLE, auto_refresh=False) as live:
        status = Status("思考中...", spinner="dots")
        timer = Timer()
        timer.start()

        stream_state = {
            "content": "",
            "reasoning_content": "",
            "reasoning_steps": [],
            "tools": [],
            "member_responses": {},
            "citations": None,
            "input_content": None
        }

        for event in run_output_stream:
            _update_stream_state(stream_state, event, entity)
            
            panels = _build_panels_from_stream_state(
                stream_state, entity, timer, status, show_reasoning, 
                show_full_reasoning, markdown, show_time
            )
            live.update(Group(*panels), refresh=True)
            
        timer.stop()
        # 最终渲染，移除 status
        final_panels = _build_panels_from_stream_state(
            stream_state, entity, timer, None, show_reasoning, 
            show_full_reasoning, markdown, show_time
        )
        live.update(Group(*final_panels), refresh=True)


async def _adisplay_streaming_output(
    run_output_stream: AsyncIterable[Union[RunOutputEvent, TeamRunOutputEvent]],
    entity: Union["Agent", "Team"],
    show_reasoning: bool,
    show_full_reasoning: bool,
    markdown: bool,
    show_time: bool
) -> None:
    """异步版本的流式输出渲染。"""
    with Live(console=CONSOLE, auto_refresh=False) as live:
        status = Status("思考中...", spinner="dots")
        timer = Timer()
        timer.start()

        stream_state = {
            "content": "",
            "reasoning_content": "",
            "reasoning_steps": [],
            "tools": [],
            "member_responses": {},
            "citations": None,
            "input_content": None
        }
        
        async for event in run_output_stream:
            _update_stream_state(stream_state, event, entity)
            
            panels = _build_panels_from_stream_state(
                stream_state, entity, timer, status, show_reasoning, 
                show_full_reasoning, markdown, show_time
            )
            live.update(Group(*panels), refresh=True)
            
        timer.stop()
        final_panels = _build_panels_from_stream_state(
            stream_state, entity, timer, None, show_reasoning, 
            show_full_reasoning, markdown, show_time
        )
        live.update(Group(*final_panels), refresh=True)


# --- 界面构建辅助函数 (Panels) ---

def _build_message_panel(input_content: Any) -> Panel:
    """构建输入消息面板。"""
    message_text = get_text_from_message(input_content)
    return Panel(Text(message_text, style="green"), title="输入消息", border_style="cyan")

def _build_reasoning_panels(
    run_output: Union[RunOutput, TeamRunOutput, Dict], 
    show_full: bool
) -> List[Panel]:
    """构建推理过程面板（思维内容和步骤）。"""
    panels = []
    steps = run_output.get("reasoning_steps") if isinstance(run_output, dict) else run_output.reasoning_steps
    content = run_output.get("reasoning_content") if isinstance(run_output, dict) else run_output.reasoning_content

    if content:
        panels.append(Panel(Markdown(content), title="思考过程", border_style="yellow"))
    
    if steps:
        for i, step in enumerate(steps):
            step_content = Text()
            if step.title: step_content.append(f"{step.title}\n", style="bold")
            if show_full and step.reasoning: step_content.append(f"推理: {step.reasoning}\n", style="dim")
            if step.action: step_content.append(f"动作: {step.action}\n")
            if step.result: step_content.append(f"结果: {step.result}\n")
            panels.append(Panel(step_content, title=f"推理步骤 {i+1}", border_style="yellow"))
            
    return panels

def _build_tool_calls_panel(tools: List[ToolExecution], entity: Union["Agent", "Team"]) -> Panel:
    """构建工具调用面板。"""
    title = f"{type(entity).__name__} 工具调用"
    content = "\n".join([f"• `{_format_tool_call(tool)}`" for tool in tools])
    return Panel(Markdown(content), title=title, border_style="magenta")

def _build_member_response_panels(
    run_output: TeamRunOutput, team: "Team", markdown: bool
) -> List[Panel]:
    """构建团队成员响应面板。"""
    panels = []
    for resp in run_output.member_responses:
        member_name = "未知成员"
        if hasattr(team, "_get_member_name"): # 兼容性检查
            member_id = resp.agent_id if isinstance(resp, RunOutput) else resp.team_id
            member_name = team._get_member_name(member_id)

        content = _format_content_for_display(resp.content, markdown)
        panels.append(Panel(content, title=f"成员响应: {member_name}", border_style="blue"))
    return panels

def _build_response_panel(
    run_output: Union[RunOutput, TeamRunOutput, Dict], 
    timer: Timer, 
    markdown: bool,
    show_time: bool
) -> Panel:
    """构建最终响应面板。"""
    content_data = run_output.get("content") if isinstance(run_output, dict) else run_output.content
    content = _format_content_for_display(content_data, markdown)
    title = f"最终响应"
    if show_time: title += f" (耗时 {timer.elapsed:.2f}s)"
    return Panel(content, title=title, border_style="green", expand=True)

def _build_citations_panel(citations: Any) -> Panel:
    """构建引用来源面板。"""
    content = "\n".join([f"{i+1}. [{c.title or c.url}]({c.url})" for i, c in enumerate(citations.urls)])
    return Panel(Markdown(content), title="引用来源", border_style="cyan")


# --- 流式状态管理与渲染辅助 ---

def _update_stream_state(state: Dict, event: Any, entity: "Team") -> None:
    """根据流入的事件更新流式状态字典。"""
    if not hasattr(event, "event"): return

    if state["input_content"] is None and hasattr(event, "run_input") and event.run_input:
        state["input_content"] = event.run_input.input_content

    if event.event in (RunEvent.RUN_CONTENT, TeamRunOutputEvent.run_content.value):
        if isinstance(event.content, str):
            state["content"] += event.content
        else:
            state["content"] = event.content # 覆盖，用于结构化输出
        
        if event.reasoning_content:
            state["reasoning_content"] += event.reasoning_content
        if event.citations:
            state["citations"] = event.citations
    
    if hasattr(event, "reasoning_steps") and event.reasoning_steps:
        state["reasoning_steps"] = event.reasoning_steps

    if hasattr(event, "tool") and event.tool and event.tool not in state["tools"]:
         state["tools"].append(event.tool)

    if hasattr(event, "member_responses") and event.member_responses:
        for resp in event.member_responses:
            member_id = resp.agent_id if isinstance(resp, RunOutput) else resp.team_id
            state["member_responses"][member_id] = resp


def _build_panels_from_stream_state(
    state: Dict, entity: Union["Agent", "Team"], timer: Timer, status: Optional[Status], 
    show_reasoning: bool, show_full_reasoning: bool, markdown: bool, show_time: bool
) -> List[Union[Panel, Status]]:
    """根据当前的流式状态构建UI面板列表。"""
    panels: List[Union[Panel, Status]] = []
    
    if status: panels.append(status)
    if state["input_content"]: panels.append(_build_message_panel(state["input_content"]))
    if show_reasoning: panels.extend(_build_reasoning_panels(state, show_full_reasoning))

    if isinstance(entity, get_args(TYPE_CHECKING and Union["Team"])) and state["member_responses"]:
        from agno.run.team import TeamRunOutput as DummyTeamRunOutput
        mock_team_run = DummyTeamRunOutput(member_responses=list(state["member_responses"].values()))
        panels.extend(_build_member_response_panels(mock_team_run, entity, markdown))

    if state["tools"]: panels.append(_build_tool_calls_panel(state["tools"], entity))
    if state["content"]: panels.append(_build_response_panel(state, timer, markdown, show_time))
    if state["citations"]: panels.append(_build_citations_panel(state["citations"]))
    
    return panels


# --- 其他格式化辅助 ---

def _format_content_for_display(content: Any, markdown: bool) -> Union[str, Markdown, JSON]:
    """将不同类型的内容格式化为 Rich 可渲染对象。"""
    if isinstance(content, str):
        return Markdown(content) if markdown else content
    if isinstance(content, BaseModel):
        try:
            return JSON(content.model_dump_json(indent=2, exclude_none=True))
        except Exception as e:
            logger.warning(f"无法将 Pydantic 模型序列化为 JSON: {e}")
            return str(content)
    try:
        return JSON(json.dumps(content, indent=2, default=str))
    except Exception:
        return str(content)

def _format_tool_call(tool: ToolExecution) -> str:
    """将 ToolExecution 对象格式化为可读字符串。"""
    args_str = ", ".join(f"{k}={v!r}" for k, v in (tool.tool_args or {}).items())
    return f"{tool.tool_name}({args_str})"


if __name__ == "__main__":
    # --- 测试代码 ---
    from agno.agent import Agent
    from agno.team import Team
    
    print("--- 正在运行 agno/ui/display.py 的测试代码 ---")
    
    # 1. 准备模拟数据
    mock_agent = Agent(name="TestAgent")
    mock_team = Team(name="TestTeam", members=[mock_agent])
    
    mock_input = RunInput(input_content="你好，世界！")
    mock_tool = ToolExecution(tool_name="search", tool_args={"query": "Agno"})
    
    mock_run_output = RunOutput(
        run_id="run-1",
        input=mock_input,
        content="这是 Agent 的最终回答。",
        reasoning_content="我应该回答这个问题。",
        tools=[mock_tool],
        metrics={"duration": 1.23}
    )
    
    mock_member_response = RunOutput(run_id="run-2", agent_id=mock_agent.id, content="这是成员的回答。")
    mock_team_run_output = TeamRunOutput(
        run_id="team-run-1",
        input=mock_input,
        content="这是 Team 的最终回答。",
        member_responses=[mock_member_response],
        metrics={"duration": 2.34}
    )
    
    # 2. 测试静态 Agent 输出
    print("\n[1] 测试静态 Agent 输出:")
    display_run_output(mock_run_output, mock_agent)

    # 3. 测试静态 Team 输出
    print("\n[2] 测试静态 Team 输出:")
    display_run_output(mock_team_run_output, mock_team)
    
    # 4. 测试流式 Agent 输出 (模拟)
    print("\n[3] 测试流式 Agent 输出 (模拟):")
    def agent_stream_generator():
        yield RunOutputEvent(event=RunEvent.RUN_STARTED, run_input=mock_input)
        yield RunOutputEvent(event=RunEvent.RUN_CONTENT, reasoning_content="开始思考...")
        yield RunOutputEvent(event=RunEvent.TOOL_CALL_STARTED, tool=mock_tool)
        yield RunOutputEvent(event=RunEvent.RUN_CONTENT, content="这是", reasoning_content="思考更多...")
        yield RunOutputEvent(event=RunEvent.RUN_CONTENT, content="最终答案。")

    display_run_output(agent_stream_generator(), mock_agent)

    print("\n--- 测试结束 ---")