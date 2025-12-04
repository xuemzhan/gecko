# P3 指标与成本追踪增强

## 概述
在 `CognitiveEngine` 中添加了轻量级的 token 成本追踪和模型定价支持，用于精细化的执行统计和成本监控。

## 核心改动

### 1. 模型定价配置 (`MODEL_PRICING`)
支持主流模型的定价查表，包括 OpenAI（GPT-3.5, GPT-4）和 Anthropic（Claude 3 系列）。
单位：USD per token。

### 2. `ExecutionStats` 增强
- 分离 `input_tokens` 和 `output_tokens`（原 `total_tokens` 改为计算属性）。
- 新增 `estimated_cost` 字段，用于累计估算成本。
- 新增方法：
  - `add_step()` 改为接收 `input_tokens` 和 `output_tokens` 分别参数。
  - `add_cost(cost: float)` 累加成本。
  - `get_total_tokens()` 返回总 token 数。
  - `get_error_rate()` 返回错误率。

### 3. `CognitiveEngine` 新方法
- `record_cost(input_tokens, output_tokens, model_name)` — 基于模型定价计算并记录成本。
- `get_stats_summary()` — 获取完整的执行统计摘要。

## 用法示例

```python
# 在 ReActEngine 的 step() 中
output = await self.model.acompletion(...)
engine.record_cost(
    input_tokens=output.usage.prompt_tokens,
    output_tokens=output.usage.completion_tokens,
    model_name="gpt-3.5-turbo"
)

# 获取统计
summary = engine.get_stats_summary()
print(f"成本: ${summary['estimated_cost']:.4f}")
print(f"错误率: {summary['error_rate']:.2%}")
```

## 测试覆盖
- `test_engine_metrics.py` 包含 5 个单元测试，覆盖：
  - 基础统计收集。
  - 错误率计算。
  - 成本追踪。
  - 统计转换为字典。
  - 模型定价配置验证。

## 性能影响
- 轻量级实现（无额外网络调用）。
- 计算复杂度为 O(1)。
- 可通过 `enable_stats=False` 在构造时禁用。

## 后续改进方向
- 支持自定义模型定价。
- 集成到监控/告警系统。
- 添加成本预警阈值。
