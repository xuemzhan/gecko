import pytest
import asyncio
import threading
import time
from unittest.mock import MagicMock, AsyncMock, patch

from gecko.core.engine.base import (
    CognitiveEngine,
    ExecutionStats,
    MODEL_PRICING,
    get_model_pricing,
    load_model_pricing,
    create_engine,
)
from gecko.core.output import AgentOutput
from gecko.core.message import Message
from gecko.core.exceptions import AgentError


class ConcreteEngine(CognitiveEngine):
    async def step(self, input_messages, **kwargs):
        return AgentOutput(content="test response")


class TestExecutionStats:
    def test_initialization(self):
        stats = ExecutionStats()
        assert stats.total_steps == 0
        assert stats.total_time == 0.0
        assert stats.input_tokens == 0
        assert stats.output_tokens == 0
        assert stats.tool_calls == 0
        assert stats.errors == 0
        assert stats.estimated_cost == 0.0

    def test_add_step(self):
        stats = ExecutionStats()
        stats.add_step(duration=1.5, input_tokens=100, output_tokens=50)
        
        assert stats.total_steps == 1
        assert stats.total_time == 1.5
        assert stats.input_tokens == 100
        assert stats.output_tokens == 50

    def test_add_step_with_error(self):
        stats = ExecutionStats()
        stats.add_step(duration=0.5, had_error=True)
        
        assert stats.total_steps == 1
        assert stats.errors == 1

    def test_add_tokens(self):
        stats = ExecutionStats()
        stats.add_tokens(input_tokens=50, output_tokens=25)
        stats.add_tokens(input_tokens=30, output_tokens=15)
        
        assert stats.input_tokens == 80
        assert stats.output_tokens == 40
        assert stats.get_total_tokens() == 120

    def test_add_tokens_empty(self):
        stats = ExecutionStats()
        stats.add_tokens()
        
        assert stats.input_tokens == 0
        assert stats.output_tokens == 0

    def test_add_tool_call(self):
        stats = ExecutionStats()
        stats.add_tool_call()
        stats.add_tool_call(3)
        
        assert stats.tool_calls == 4

    def test_add_tool_call_negative(self):
        stats = ExecutionStats()
        stats.add_tool_call(-1)
        
        assert stats.tool_calls == 0

    def test_add_error(self):
        stats = ExecutionStats()
        stats.add_error()
        stats.add_error(2)
        
        assert stats.errors == 3

    def test_add_cost(self):
        stats = ExecutionStats()
        stats.add_cost(0.5)
        stats.add_cost(0.3)
        
        assert abs(stats.estimated_cost - 0.8) < 0.0001

    def test_get_avg_step_time(self):
        stats = ExecutionStats()
        stats.add_step(duration=1.0)
        stats.add_step(duration=2.0)
        stats.add_step(duration=3.0)
        
        assert stats.get_avg_step_time() == 2.0

    def test_get_avg_step_time_no_steps(self):
        stats = ExecutionStats()
        assert stats.get_avg_step_time() == 0.0

    def test_get_error_rate(self):
        stats = ExecutionStats()
        stats.add_step(duration=0.1, had_error=False)
        stats.add_step(duration=0.2, had_error=True)
        stats.add_step(duration=0.15, had_error=False)
        
        assert abs(stats.get_error_rate() - (1.0 / 3)) < 0.01

    def test_get_error_rate_no_steps(self):
        stats = ExecutionStats()
        assert stats.get_error_rate() == 0.0

    def test_to_dict(self):
        stats = ExecutionStats()
        stats.add_step(duration=1.5, input_tokens=100, output_tokens=50)
        stats.add_tool_call(2)
        stats.add_cost(0.025)
        
        result = stats.to_dict()
        
        assert result["total_steps"] == 1
        assert result["total_time"] == 1.5
        assert result["avg_step_time"] == 1.5
        assert result["input_tokens"] == 100
        assert result["output_tokens"] == 50
        assert result["total_tokens"] == 150
        assert result["tool_calls"] == 2
        assert result["errors"] == 0
        assert result["error_rate"] == 0.0
        assert abs(result["estimated_cost"] - 0.025) < 0.0001

    def test_thread_safety(self):
        stats = ExecutionStats()
        
        def worker():
            for _ in range(100):
                stats.add_step(duration=0.1, input_tokens=10, output_tokens=5)
        
        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert stats.total_steps == 1000
        assert stats.input_tokens == 10000
        assert stats.output_tokens == 5000


class TestModelPricing:
    def test_default_pricing_loaded(self):
        pricing = get_model_pricing()
        
        assert "gpt-3.5-turbo" in pricing
        assert "gpt-4" in pricing
        assert "claude-3-haiku" in pricing
        
        assert pricing["gpt-3.5-turbo"]["input"] == 0.5
        assert pricing["gpt-3.5-turbo"]["output"] == 1.5

    def test_get_model_pricing_returns_copy(self):
        pricing1 = get_model_pricing()
        pricing2 = get_model_pricing()
        
        assert pricing1 is not pricing2
        assert pricing1 == pricing2

    def test_get_model_pricing_force_reload(self):
        pricing1 = get_model_pricing()
        pricing2 = get_model_pricing(force_reload=True)
        
        assert pricing1 == pricing2

    def test_model_pricing_dict_operations(self):
        assert "gpt-4" in MODEL_PRICING
        assert MODEL_PRICING.get("gpt-4") is not None
        assert len(MODEL_PRICING) > 0
        assert list(MODEL_PRICING.keys())
        assert list(MODEL_PRICING.values())
        assert list(MODEL_PRICING.items())

    def test_model_pricing_dict_set_item(self):
        MODEL_PRICING["test-model"] = {"input": 1.0, "output": 2.0}
        assert MODEL_PRICING["test-model"]["input"] == 1.0

    def test_model_pricing_dict_thread_safety(self):
        def worker():
            for i in range(10):
                MODEL_PRICING[f"model-{threading.current_thread().name}-{i}"] = {
                    "input": 1.0,
                    "output": 2.0
                }
        
        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()


class TestCognitiveEngine:
    @pytest.fixture
    def mock_model(self):
        model = MagicMock()
        model._supports_function_calling = True
        return model

    @pytest.fixture
    def mock_toolbox(self):
        toolbox = MagicMock()
        toolbox.to_openai_schema.return_value = []
        return toolbox

    @pytest.fixture
    def mock_memory(self):
        memory = MagicMock()
        memory.storage = None
        memory.session_id = "test_session"
        return memory

    @pytest.fixture
    def engine(self, mock_model, mock_toolbox, mock_memory):
        return ConcreteEngine(mock_model, mock_toolbox, mock_memory)

    def test_abstract_class_cannot_instantiate(self, mock_model, mock_toolbox, mock_memory):
        with pytest.raises(TypeError):
            CognitiveEngine(mock_model, mock_toolbox, mock_memory)

    def test_model_validation(self, mock_toolbox, mock_memory):
        invalid_model = object()
        
        with pytest.raises(TypeError):
            ConcreteEngine(invalid_model, mock_toolbox, mock_memory)

    def test_initialization(self, engine, mock_model):
        assert engine.model == mock_model
        assert engine.max_iterations == 10
        assert engine.enable_stats is True
        assert engine.stats is not None
        assert isinstance(engine.stats, ExecutionStats)

    def test_initialization_without_stats(self, mock_model, mock_toolbox, mock_memory):
        engine = ConcreteEngine(
            mock_model, mock_toolbox, mock_memory, enable_stats=False
        )
        assert engine.stats is None

    def test_validate_input_empty(self, engine):
        with pytest.raises(ValueError, match="cannot be empty"):
            engine.validate_input([])

    def test_validate_input_invalid_type(self, engine):
        with pytest.raises(TypeError, match="must be Message instances"):
            engine.validate_input([{"role": "user", "content": "test"}])

    def test_validate_input_success(self, engine):
        messages = [Message.user("test")]
        engine.validate_input(messages)

    def test_record_step(self, engine):
        engine.record_step(duration=1.5, input_tokens=100, output_tokens=50)
        
        assert engine.stats.total_steps == 1
        assert engine.stats.total_time == 1.5
        assert engine.stats.input_tokens == 100

    def test_record_step_without_stats(self, mock_model, mock_toolbox, mock_memory):
        engine = ConcreteEngine(
            mock_model, mock_toolbox, mock_memory, enable_stats=False
        )
        engine.record_step(duration=1.0)

    def test_record_tokens(self, engine):
        engine.record_tokens(input_tokens=100, output_tokens=50)
        assert engine.stats.input_tokens == 100
        assert engine.stats.output_tokens == 50

    def test_record_tool_call(self, engine):
        engine.record_tool_call(3)
        assert engine.stats.tool_calls == 3

    def test_record_error(self, engine):
        engine.record_error(2)
        assert engine.stats.errors == 2

    def test_record_cost(self, engine):
        engine.record_cost(
            input_tokens=1_000_000,
            output_tokens=500_000,
            model_name="gpt-3.5-turbo"
        )
        
        expected_cost = (1_000_000 * 0.5 / 1_000_000) + (500_000 * 1.5 / 1_000_000)
        assert abs(engine.stats.estimated_cost - expected_cost) < 0.0001

    def test_record_cost_unknown_model(self, engine):
        engine.record_cost(
            input_tokens=1000,
            output_tokens=500,
            model_name="unknown-model"
        )
        
        assert engine.stats.estimated_cost > 0

    def test_record_cost_model_prefix_match(self, engine):
        engine.record_cost(
            input_tokens=1000,
            output_tokens=500,
            model_name="gpt-4-turbo-preview"
        )
        
        assert engine.stats.estimated_cost > 0

    def test_get_stats_summary(self, engine):
        engine.record_step(duration=1.0, input_tokens=100, output_tokens=50)
        summary = engine.get_stats_summary()
        
        assert summary["total_steps"] == 1
        assert summary["input_tokens"] == 100

    def test_get_stats_summary_without_stats(self, mock_model, mock_toolbox, mock_memory):
        engine = ConcreteEngine(
            mock_model, mock_toolbox, mock_memory, enable_stats=False
        )
        assert engine.get_stats_summary() == {}

    def test_get_stats(self, engine):
        stats = engine.get_stats()
        assert stats is not None
        assert isinstance(stats, dict)

    def test_reset_stats(self, engine):
        engine.record_step(duration=1.0)
        engine.reset_stats()
        
        assert engine.stats.total_steps == 0

    def test_get_config(self, engine):
        engine.set_config("test_key", "test_value")
        assert engine.get_config("test_key") == "test_value"
        assert engine.get_config("nonexistent", "default") == "default"

    def test_set_config(self, engine):
        engine.set_config("max_retries", 5)
        assert engine.get_config("max_retries") == 5

    def test_supports_streaming(self, engine):
        result = engine.supports_streaming()
        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_initialize(self, engine):
        await engine.initialize()

    @pytest.mark.asyncio
    async def test_cleanup(self, engine):
        await engine.cleanup()

    @pytest.mark.asyncio
    async def test_cleanup_timeout(self, mock_model, mock_toolbox, mock_memory):
        class SlowEngine(ConcreteEngine):
            async def _do_cleanup(self):
                await asyncio.sleep(60)
        
        engine = SlowEngine(mock_model, mock_toolbox, mock_memory)
        
        start = time.time()
        await engine.cleanup()
        elapsed = time.time() - start
        
        assert elapsed < 35

    @pytest.mark.asyncio
    async def test_context_manager(self, mock_model, mock_toolbox, mock_memory):
        initialized = False
        cleaned_up = False
        
        class TestEngine(ConcreteEngine):
            async def initialize(self):
                nonlocal initialized
                initialized = True
            
            async def _do_cleanup(self):
                nonlocal cleaned_up
                cleaned_up = True
        
        async with TestEngine(mock_model, mock_toolbox, mock_memory) as engine:
            assert initialized is True
            assert isinstance(engine, TestEngine)
        
        assert cleaned_up is True

    @pytest.mark.asyncio
    async def test_health_check(self, engine):
        health = await engine.health_check()
        
        assert health["engine"] == "ConcreteEngine"
        assert health["status"] == "healthy"
        assert "model" in health
        assert "supports_streaming" in health

    @pytest.mark.asyncio
    async def test_before_step(self, engine):
        messages = [Message.user("test")]
        await engine.before_step(messages)

    @pytest.mark.asyncio
    async def test_before_step_with_hook(self, engine):
        hook_called = False
        
        async def hook(messages, **kwargs):
            nonlocal hook_called
            hook_called = True
        
        engine.before_step_hook = hook
        await engine.before_step([Message.user("test")])
        
        assert hook_called is True

    @pytest.mark.asyncio
    async def test_before_step_hook_failure(self, engine):
        def failing_hook(messages, **kwargs):
            raise RuntimeError("Hook failed")
        
        engine.before_step_hook = failing_hook
        engine.hooks_fail_fast = False
        
        await engine.before_step([Message.user("test")])

    @pytest.mark.asyncio
    async def test_before_step_hook_failure_fast(self, engine):
        def failing_hook(messages, **kwargs):
            raise RuntimeError("Hook failed")
        
        engine.before_step_hook = failing_hook
        engine.hooks_fail_fast = True
        
        with pytest.raises(RuntimeError):
            await engine.before_step([Message.user("test")])

    @pytest.mark.asyncio
    async def test_after_step(self, engine):
        messages = [Message.user("test")]
        output = AgentOutput(content="response")
        await engine.after_step(messages, output)

    @pytest.mark.asyncio
    async def test_after_step_with_hook(self, engine):
        hook_called = False
        
        async def hook(messages, output, **kwargs):
            nonlocal hook_called
            hook_called = True
        
        engine.after_step_hook = hook
        output = AgentOutput(content="response")
        await engine.after_step([Message.user("test")], output)
        
        assert hook_called is True

    @pytest.mark.asyncio
    async def test_on_error(self, engine):
        error = RuntimeError("test error")
        await engine.on_error(error, [Message.user("test")])

    @pytest.mark.asyncio
    async def test_on_error_with_hook(self, engine):
        hook_called = False
        
        async def hook(error, messages, **kwargs):
            nonlocal hook_called
            hook_called = True
        
        engine.on_error_hook = hook
        await engine.on_error(RuntimeError("test"), [Message.user("test")])
        
        assert hook_called is True

    @pytest.mark.asyncio
    async def test_maybe_await_sync_function(self, engine):
        def sync_func():
            return "result"
        
        result = await engine._maybe_await(sync_func)
        assert result == "result"

    @pytest.mark.asyncio
    async def test_maybe_await_async_function(self, engine):
        async def async_func():
            return "async_result"
        
        result = await engine._maybe_await(async_func)
        assert result == "async_result"

    @pytest.mark.asyncio
    async def test_publish_event(self, engine, mock_model, mock_toolbox, mock_memory):
        event_bus = MagicMock()
        event_bus.publish = AsyncMock()
        
        engine_with_bus = ConcreteEngine(
            mock_model, mock_toolbox, mock_memory, event_bus=event_bus
        )
        
        await engine_with_bus._publish_event("test_event", {"key": "value"})
        
        event_bus.publish.assert_called_once()

    @pytest.mark.asyncio
    async def test_publish_event_no_bus(self, engine):
        await engine._publish_event("test_event", {"key": "value"})

    @pytest.mark.asyncio
    async def test_safe_execute(self, engine):
        def success_func():
            return "success"
        
        result = await engine._safe_execute(success_func)
        assert result == "success"

    @pytest.mark.asyncio
    async def test_safe_execute_with_error(self, engine):
        def failing_func():
            raise RuntimeError("test error")
        
        with pytest.raises(RuntimeError):
            await engine._safe_execute(failing_func)

    @pytest.mark.asyncio
    async def test_safe_execute_records_error(self, engine):
        def failing_func():
            raise RuntimeError("test error")
        
        try:
            await engine._safe_execute(failing_func)
        except RuntimeError:
            pass
        
        assert engine.stats.errors == 1

    def test_repr(self, engine):
        repr_str = repr(engine)
        assert "ConcreteEngine" in repr_str
        assert "max_iterations=10" in repr_str


class TestCreateEngine:
    @pytest.fixture
    def mock_model(self):
        model = MagicMock()
        model._supports_function_calling = True
        return model

    @pytest.fixture
    def mock_toolbox(self):
        toolbox = MagicMock()
        toolbox.to_openai_schema.return_value = []
        return toolbox

    @pytest.fixture
    def mock_memory(self):
        memory = MagicMock()
        memory.storage = None
        return memory

    def test_create_engine_success(self, mock_model, mock_toolbox, mock_memory):
        engine = create_engine(
            ConcreteEngine, mock_model, mock_toolbox, mock_memory
        )
        
        assert isinstance(engine, ConcreteEngine)
        assert engine.model == mock_model

    def test_create_engine_with_kwargs(self, mock_model, mock_toolbox, mock_memory):
        engine = create_engine(
            ConcreteEngine,
            mock_model,
            mock_toolbox,
            mock_memory,
            max_iterations=20,
            enable_stats=False
        )
        
        assert engine.max_iterations == 20
        assert engine.stats is None

    def test_create_engine_invalid_class(self, mock_model, mock_toolbox, mock_memory):
        class NotAnEngine:
            pass
        
        with pytest.raises(TypeError, match="must be a subclass"):
            create_engine(NotAnEngine, mock_model, mock_toolbox, mock_memory)