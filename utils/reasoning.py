# agno/utils/reasoning.py

"""
Agent 推理过程工具模块

该模块提供了用于处理和记录 Agent 推理步骤（Reasoning Steps）的辅助函数。
这些功能对于实现思维链（Chain of Thought）以及提供模型决策过程的透明度至关重要。

主要功能:
- 从 LLM 的原始输出中提取被特定标签（如 `<think>...</think>`）包裹的思维内容。
- 将结构化的 `ReasoningStep` 对象和原始推理文本附加到 `RunOutput` 中。
- 将推理过程的性能度量（如耗时）添加到 `RunOutput` 的元数据中。
"""

from typing import TYPE_CHECKING, List, Optional, Tuple, Union

from agno.models.message import Message
from agno.models.metrics import Metrics
from agno.reasoning.step import ReasoningStep

# 假设日志模块已按规划重构
# from agno.utils.log import log_error
# 在独立运行时，使用标准日志库
import logging
logger = logging.getLogger(__name__)

def log_error(message: str):
    logger.error(message)

# 避免循环导入
if TYPE_CHECKING:
    from agno.run.agent import RunOutput
    from agno.run.team import TeamRunOutput


def extract_thinking_content(content: str, start_tag: str = "<think>", end_tag: str = "</think>") -> Tuple[Optional[str], str]:
    """
    从 LLM 的响应文本中提取被特定 XML 风格标签包裹的“思维”内容。

    例如，对于输入 "Thinking...<think>I should use the search tool.</think>Here is the answer."，
    它会返回 ("I should use the search tool.", "Thinking...Here is the answer.")。
    
    该函数可以处理多个思维块，并将它们合并。

    Args:
        content (str): LLM 的原始输出字符串。
        start_tag (str): 思维内容的开始标签。
        end_tag (str): 思维内容的结束标签。

    Returns:
        Tuple[Optional[str], str]: 一个元组，第一个元素是提取出的所有思维内容（如果没有则为 None），
                                   第二个元素是移除了思维内容及其标签后的剩余文本。
    """
    if not content or end_tag not in content:
        return None, content

    reasoning_parts = []
    output_parts = []
    last_end = 0

    while (start_idx := content.find(start_tag, last_end)) != -1:
        end_idx = content.find(end_tag, start_idx)
        if end_idx == -1:
            # 如果有开始标签但没有结束标签，则停止处理
            break

        # 添加开始标签之前的内容
        output_parts.append(content[last_end:start_idx])
        
        # 提取思维内容
        reasoning_start = start_idx + len(start_tag)
        reasoning_parts.append(content[reasoning_start:end_idx].strip())
        
        last_end = end_idx + len(end_tag)

    # 添加最后一个结束标签之后的内容
    output_parts.append(content[last_end:])

    reasoning_content = "\n".join(reasoning_parts).strip() if reasoning_parts else None
    output_content = "".join(output_parts).strip()

    return reasoning_content, output_content


def update_run_output_with_reasoning(
    run_response: Union["RunOutput", "TeamRunOutput"],
    reasoning_steps: List[ReasoningStep],
    reasoning_agent_messages: List[Message],
) -> None:
    """
    用推理过程的结果更新 `RunOutput` 或 `TeamRunOutput` 对象。

    此函数会：
    1. 将结构化的 `ReasoningStep` 对象列表追加到 `run_response.reasoning_steps`。
    2. 将包含推理过程的 `Message` 对象列表追加到 `run_response.reasoning_messages`。
    3. 从 `ReasoningStep` 对象生成人类可读的 `reasoning_content` 字符串并追加。

    Args:
        run_response: 要更新的 `RunOutput` 或 `TeamRunOutput` 实例。
        reasoning_steps: 在推理过程中生成的 `ReasoningStep` 对象列表。
        reasoning_agent_messages: 在推理过程中生成的 `Message` 对象列表。
    """
    # 1. 更新 reasoning_steps
    if run_response.reasoning_steps is None:
        run_response.reasoning_steps = []
    run_response.reasoning_steps.extend(reasoning_steps)

    # 2. 更新 reasoning_messages
    if run_response.reasoning_messages is None:
        run_response.reasoning_messages = []
    run_response.reasoning_messages.extend(reasoning_agent_messages)

    # 3. 生成并追加 reasoning_content 字符串
    new_reasoning_content = ""
    for step in reasoning_steps:
        if step.title:
            new_reasoning_content += f"## {step.title}\n"
        if step.reasoning:
            new_reasoning_content += f"{step.reasoning}\n"
        if step.action:
            new_reasoning_content += f"Action: {step.action}\n"
        if step.result:
            new_reasoning_content += f"Result: {step.result}\n"
        new_reasoning_content += "\n"

    if not run_response.reasoning_content:
        run_response.reasoning_content = new_reasoning_content.strip()
    else:
        run_response.reasoning_content += f"\n{new_reasoning_content.strip()}"


def add_reasoning_metrics_to_metadata(
    run_response: Union["RunOutput", "TeamRunOutput"], 
    reasoning_time_taken: float
) -> None:
    """
    将推理过程的性能度量（如此次推理的总耗时）添加到 `RunOutput` 的元数据中。

    它会创建一个包含 `Metrics` 对象的 `Message`，并将其添加到 `reasoning_messages` 列表。

    Args:
        run_response: 要更新的 `RunOutput` 或 `TeamRunOutput` 实例。
        reasoning_time_taken (float): 推理过程所花费的总时间（秒）。
    """
    try:
        if run_response.reasoning_messages is None:
            run_response.reasoning_messages = []

        metrics_message = Message(
            role="assistant",
            content=run_response.reasoning_content,  # 将当前所有推理内容作为消息内容
            metrics=Metrics(duration=reasoning_time_taken),
        )

        run_response.reasoning_messages.append(metrics_message)

    except Exception as e:
        log_error(f"向元数据添加推理度量时失败: {e}")


if __name__ == "__main__":
    # --- 测试代码 ---
    from agno.run.agent import RunOutput

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    print("--- 正在运行 agno/utils/reasoning.py 的测试代码 ---")

    # 1. 测试 extract_thinking_content
    print("\n[1] 测试 extract_thinking_content:")
    
    test_cases = [
        "No tags here.",
        "<think>Step 1: analyze the query.</think>Here is your answer.",
        "Some initial text. <think>First thought.</think> Some middle text. <think>Second thought.</think> Final text.",
        "Mismatched tag <think>... no end tag"
    ]
    
    for i, content in enumerate(test_cases):
        reasoning, output = extract_thinking_content(content)
        print(f"\n  测试用例 {i+1}:")
        print(f"    输入: '{content}'")
        print(f"    提取的思维: '{reasoning}'")
        print(f"    剩余的输出: '{output}'")

    # 2. 测试 update_run_output_with_reasoning
    print("\n[2] 测试 update_run_output_with_reasoning:")
    
    run_output = RunOutput(run_id="run-1")
    
    steps = [
        ReasoningStep(title="思考", reasoning="我需要搜索信息。"),
        ReasoningStep(action="search('Agno AGI')", result="Agno是一个AI框架。")
    ]
    messages = [
        Message(role="assistant", content="<thinking>我正在搜索...</thinking>")
    ]
    
    update_run_output_with_reasoning(run_output, steps, messages)
    
    print("  更新后的 RunOutput:")
    print(f"    - Reasoning Steps 数量: {len(run_output.reasoning_steps or [])}")
    print(f"    - Reasoning Messages 数量: {len(run_output.reasoning_messages or [])}")
    print( "    - Reasoning Content:")
    print("---")
    print(run_output.reasoning_content)
    print("---")
    
    assert len(run_output.reasoning_steps or []) == 2
    assert len(run_output.reasoning_messages or []) == 1
    assert "Agno是一个AI框架" in (run_output.reasoning_content or "")

    # 3. 测试 add_reasoning_metrics_to_metadata
    print("\n[3] 测试 add_reasoning_metrics_to_metadata:")
    
    run_output_metrics = RunOutput(run_id="run-2", reasoning_content="总思考过程")
    add_reasoning_metrics_to_metadata(run_output_metrics, reasoning_time_taken=1.234)
    
    print("  添加度量后的 RunOutput:")
    metric_msg = (run_output_metrics.reasoning_messages or [])[0]
    print(f"    - Reasoning Messages 数量: {len(run_output_metrics.reasoning_messages or [])}")
    print(f"    - 度量消息内容: '{metric_msg.content}'")
    print(f"    - 度量消息耗时: {metric_msg.metrics.duration if metric_msg.metrics else 'N/A'}")
    
    assert metric_msg.metrics is not None
    assert metric_msg.metrics.duration == 1.234
    
    print("\n--- 测试结束 ---")