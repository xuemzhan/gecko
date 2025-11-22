# examples/output_demo.py
from gecko.core.output import (
    AgentOutput,
    TokenUsage,
    create_text_output,
    create_tool_output,
    merge_outputs
)


def main():
    print("=== Gecko AgentOutput 示例 ===\n")
    
    # 1. 简单文本输出
    print("1. 简单文本输出")
    output1 = AgentOutput(content="Hello, how can I help you today?")
    print(f"   {output1}")
    print(f"   有内容: {output1.has_content()}")
    print(f"   是否为空: {output1.is_empty()}\n")
    
    # 2. 带 Token 使用的输出
    print("2. 带 Token 使用统计")
    usage = TokenUsage(
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150
    )
    output2 = AgentOutput(
        content="Based on my analysis...",
        usage=usage
    )
    print(f"   {output2}")
    print(f"   Usage: {output2.usage}")
    
    # 估算成本（GPT-4 价格示例）
    cost = usage.get_cost_estimate(
        prompt_price_per_1k=0.03,
        completion_price_per_1k=0.06
    )
    print(f"   估算成本: ${cost:.4f}\n")
    
    # 3. 带工具调用的输出
    print("3. 带工具调用的输出")
    output3 = AgentOutput(
        content="I'll search for that information.",
        tool_calls=[
            {
                "id": "call_1",
                "function": {
                    "name": "search",
                    "arguments": '{"query": "AI trends 2024"}'
                }
            },
            {
                "id": "call_2",
                "function": {
                    "name": "calculator",
                    "arguments": '{"expression": "2+2"}'
                }
            }
        ]
    )
    print(f"   {output3}")
    print(f"   工具调用数: {output3.tool_call_count()}")
    print(f"   工具名称: {output3.get_tool_names()}\n")
    
    # 4. 格式化输出
    print("4. 格式化输出")
    print(output3.format())
    
    # 5. 输出统计
    print("5. 输出统计")
    stats = output3.get_stats()
    print(f"   统计信息: {stats}\n")
    
    # 6. 转换为消息格式
    print("6. 转换为 OpenAI 消息格式")
    msg_dict = output3.to_message_dict()
    print(f"   {msg_dict}\n")
    
    # 7. 使用工具函数快速创建
    print("7. 使用工具函数")
    quick_output = create_text_output(
        "Quick response",
        usage=TokenUsage(prompt_tokens=10, completion_tokens=5),
        source="demo"
    )
    print(f"   {quick_output}")
    print(f"   元数据: {quick_output.metadata}\n")
    
    # 8. 合并多个输出
    print("8. 合并多个输出")
    out1 = AgentOutput(content="Part 1", usage=TokenUsage(prompt_tokens=10, completion_tokens=5))
    out2 = AgentOutput(content="Part 2", usage=TokenUsage(prompt_tokens=20, completion_tokens=10))
    
    merged = merge_outputs([out1, out2])
    print(f"   合并内容: {merged.content}")
    print(f"   合并 usage: {merged.usage}\n")
    
    # 9. 输出预览
    print("9. 文本预览")
    long_output = AgentOutput(content="A" * 200)
    preview = long_output.get_text_preview(50)
    print(f"   预览: {preview}\n")
    
    # 10. 布尔值转换
    print("10. 布尔值转换")
    empty_output = AgentOutput()
    print(f"   空输出: bool(empty_output) = {bool(empty_output)}")
    print(f"   有效输出: bool(output1) = {bool(output1)}")


if __name__ == "__main__":
    main()