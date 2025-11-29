# examples/output_demo.py
"""
Gecko output 模块综合示例

本 Demo 覆盖以下能力：
1. AgentOutput 基本用法（文本输出）
2. 携带 TokenUsage 的输出及成本估算
3. 带工具调用的输出
4. 格式化输出 format()
5. 统计信息 get_stats()
6. 转换为 OpenAI 消息格式 to_message_dict()
7. 使用工厂函数 create_text_output / create_tool_output
8. 合并多个输出 merge_outputs()
9. 文本预览 get_text_preview()
10. 布尔值转换 __bool__()

扩展能力：
11. JsonOutput 结构化 JSON 输出 + create_json_output()
12. StreamingOutput 流式输出（StreamingChunk）及 finalize()
"""

from gecko.core.output import (
    AgentOutput,
    TokenUsage,
    create_text_output,
    create_tool_output,
    create_json_output,
    merge_outputs,
    JsonOutput,
    StreamingOutput,
    StreamingChunk,
)


def main():
    print("=== Gecko AgentOutput / JsonOutput / StreamingOutput 示例 ===\n")

    # 1. 简单文本输出 ---------------------------------------------------------
    print("1. 简单文本输出")
    output1 = AgentOutput(content="Hello, how can I help you today?")
    print(f"   {output1}")
    print(f"   有内容: {output1.has_content()}")
    print(f"   是否为空: {output1.is_empty()}\n")

    # 2. 带 Token 使用的输出 ---------------------------------------------------
    print("2. 带 Token 使用统计")
    usage = TokenUsage(
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
    )
    output2 = AgentOutput(
        content="Based on my analysis...",
        usage=usage,
    )
    print(f"   {output2}")
    print(f"   Usage: {output2.usage}")

    # 估算成本（示例价格）
    cost = usage.get_cost_estimate(
        prompt_price_per_1k=0.03,
        completion_price_per_1k=0.06,
    )
    print(f"   估算成本: ${cost:.4f}\n")

    # 3. 带工具调用的输出 -------------------------------------------------------
    print("3. 带工具调用的输出")
    output3 = AgentOutput(
        content="I'll search for that information.",
        tool_calls=[
            {
                "id": "call_1",
                "function": {
                    "name": "search",
                    "arguments": '{"query": "AI trends 2024"}',
                },
            },
            {
                "id": "call_2",
                "function": {
                    "name": "calculator",
                    "arguments": '{"expression": "2+2"}',
                },
            },
        ],
    )
    print(f"   {output3}")
    print(f"   工具调用数: {output3.tool_call_count()}")
    print(f"   工具名称: {output3.get_tool_names()}\n")

    # 4. 格式化输出 -----------------------------------------------------------
    print("4. 格式化输出")
    print(output3.format())

    # 5. 输出统计 -------------------------------------------------------------
    print("5. 输出统计")
    stats = output3.get_stats()
    print(f"   统计信息: {stats}\n")

    # 6. 转换为消息格式 -------------------------------------------------------
    print("6. 转换为 OpenAI 消息格式")
    msg_dict = output3.to_message_dict()
    print(f"   {msg_dict}\n")

    # 7. 使用工具函数快速创建 ---------------------------------------------------
    print("7. 使用工具函数（create_text_output）")
    quick_output = create_text_output(
        "Quick response",
        usage=TokenUsage(prompt_tokens=10, completion_tokens=5),
        source="demo",
    )
    print(f"   {quick_output}")
    print(f"   元数据: {quick_output.metadata}\n")

    print("7+. 使用工具函数（create_tool_output）")
    quick_tool_output = create_tool_output(
        tool_calls=[
            {
                "id": "call_demo",
                "function": {
                    "name": "demo_tool",
                    "arguments": '{"foo": "bar"}',
                },
            }
        ],
        content="I will call demo_tool for you.",
        scene="demo",
    )
    print(f"   {quick_tool_output}")
    print(f"   工具调用数: {quick_tool_output.tool_call_count()}")
    print(f"   元数据: {quick_tool_output.metadata}\n")

    # 8. 合并多个输出 ---------------------------------------------------------
    print("8. 合并多个输出")
    out1 = AgentOutput(
        content="Part 1",
        usage=TokenUsage(prompt_tokens=10, completion_tokens=5),
    )
    out2 = AgentOutput(
        content="Part 2",
        usage=TokenUsage(prompt_tokens=20, completion_tokens=10),
    )

    merged = merge_outputs([out1, out2])
    print(f"   合并内容: {merged.content}")
    print(f"   合并 usage: {merged.usage}\n")

    # 9. 输出预览 -------------------------------------------------------------
    print("9. 文本预览")
    long_output = AgentOutput(content="A" * 200)
    preview = long_output.get_text_preview(50)
    print(f"   预览: {preview}\n")

    # 10. 布尔值转换 ----------------------------------------------------------
    print("10. 布尔值转换")
    empty_output = AgentOutput()
    print(f"   空输出: bool(empty_output) = {bool(empty_output)}")
    print(f"   有效输出: bool(output1) = {bool(output1)}\n")

    # 11. JsonOutput 示例 -----------------------------------------------------
    print("11. JsonOutput 结构化输出示例")
    json_data = {
        "status": "ok",
        "items": [
            {"id": 1, "name": "foo"},
            {"id": 2, "name": "bar"},
        ],
    }
    json_output: JsonOutput = create_json_output(
        data=json_data,
        usage=TokenUsage(prompt_tokens=30, completion_tokens=20),
        schema_version="v1",
        source="json-demo",
    )
    print(f"   JsonOutput.data: {json_output.data}")
    print(f"   JsonOutput.metadata: {json_output.metadata}")
    print(f"   JsonOutput.usage: {json_output.usage}")
    print(f"   JsonOutput.summary(): {json_output.summary()}")

    # 将 JsonOutput 转换为 AgentOutput（方便统一处理/记录日志）
    json_as_agent: AgentOutput = json_output.to_agent_output(pretty=True)
    print("\n   JsonOutput -> AgentOutput (pretty JSON):")
    print(json_as_agent.format(include_metadata=True))

    # 12. StreamingOutput 示例 -----------------------------------------------
    print("12. StreamingOutput 流式输出示例")

    # 创建 StreamingOutput，用于收集多个流式片段
    streaming = StreamingOutput(metadata={"model": "glm-4-flash"})

    # 模拟 Provider 分片输出内容
    chunk1 = StreamingChunk(index=0, content_delta="Hello")
    chunk2 = StreamingChunk(index=1, content_delta=", ")
    chunk3 = StreamingChunk(index=2, content_delta="streaming world!")

    streaming.append_chunk(chunk1)
    streaming.append_chunk(chunk2)
    streaming.append_chunk(chunk3)

    # 边消费边打印（模拟流式输出过程）
    print("   流式增量输出：", end="")
    for delta in streaming.iter_contents():
        print(delta, end="")
    print("\n")

    # 流结束后汇总为 AgentOutput
    final_output = streaming.finalize()
    print("   StreamingOutput.finalize() 得到的 AgentOutput：")
    print(f"   内容: {final_output.content}")
    print(f"   元数据: {final_output.metadata}")
    print(f"   is_empty: {final_output.is_empty()}")
    print(f"   summary: {final_output.summary()}\n")

    print("=== Demo 结束 ===")


if __name__ == "__main__":
    main()
