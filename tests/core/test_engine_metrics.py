import pytest
from gecko.core.engine.base import ExecutionStats, MODEL_PRICING


def test_execution_stats_basic():
    """Test basic stats collection."""
    stats = ExecutionStats()
    assert stats.total_steps == 0
    assert stats.get_total_tokens() == 0
    assert stats.estimated_cost == 0.0
    
    # Add a step with tokens
    stats.add_step(duration=0.5, input_tokens=100, output_tokens=50)
    assert stats.total_steps == 1
    assert stats.total_time == 0.5
    assert stats.input_tokens == 100
    assert stats.output_tokens == 50
    assert stats.get_total_tokens() == 150


def test_execution_stats_error_rate():
    """Test error rate calculation."""
    stats = ExecutionStats()
    stats.add_step(duration=0.1, input_tokens=10, output_tokens=5, had_error=False)
    stats.add_step(duration=0.2, input_tokens=20, output_tokens=10, had_error=True)
    stats.add_step(duration=0.15, input_tokens=15, output_tokens=8, had_error=False)
    
    assert stats.total_steps == 3
    assert stats.errors == 1
    assert abs(stats.get_error_rate() - 1.0/3) < 0.01


def test_execution_stats_cost_tracking():
    """Test cost calculation based on tokens."""
    stats = ExecutionStats()
    # gpt-3.5-turbo pricing: 0.5 / 1M input, 1.5 / 1M output
    pricing = MODEL_PRICING["gpt-3.5-turbo"]
    
    input_tokens = 1_000_000
    output_tokens = 500_000
    
    expected_cost = (input_tokens * pricing["input"]) + (output_tokens * pricing["output"])
    stats.add_cost(expected_cost)
    
    assert abs(stats.estimated_cost - (0.5 + 0.75)) < 0.01  # $0.5 + $0.75 = $1.25


def test_execution_stats_to_dict():
    """Test stats conversion to dict."""
    stats = ExecutionStats()
    stats.add_step(duration=0.5, input_tokens=100, output_tokens=50, had_error=False)
    stats.add_tool_call()
    stats.add_cost(0.005)
    
    stats_dict = stats.to_dict()
    
    assert stats_dict["total_steps"] == 1
    assert stats_dict["total_time"] == 0.5
    assert stats_dict["avg_step_time"] == 0.5
    assert stats_dict["input_tokens"] == 100
    assert stats_dict["output_tokens"] == 50
    assert stats_dict["total_tokens"] == 150
    assert stats_dict["tool_calls"] == 1
    assert stats_dict["errors"] == 0
    assert stats_dict["error_rate"] == 0.0
    assert abs(stats_dict["estimated_cost"] - 0.005) < 0.0001


def test_model_pricing_available():
    """Test that model pricing is available for common models."""
    required_models = ["gpt-3.5-turbo", "gpt-4", "claude-3-haiku"]
    
    for model in required_models:
        assert model in MODEL_PRICING
        assert "input" in MODEL_PRICING[model]
        assert "output" in MODEL_PRICING[model]
        assert MODEL_PRICING[model]["input"] > 0
        assert MODEL_PRICING[model]["output"] > 0
