# tests/core/test_structure.py
from unittest.mock import patch
import importlib

import pytest
from pydantic import BaseModel, Field, RootModel

from gecko.core.structure import (
    StructureEngine,
    StructureParseError,
    extract_json_from_text,
    parse_structured_output,
    ExtractionStrategy,
    register_extraction_strategy,
)

# 内部模块做白盒测试（schema / json_extractor）
from gecko.core.structure import schema as schema_utils  # noqa: F401
from gecko.core.structure import json_extractor


# ==========================
# 测试用模型定义
# ==========================

class User(BaseModel):
    name: str
    age: int


class ComplexInner(BaseModel):
    key: str
    value: int


class ComplexModel(BaseModel):
    title: str = Field(description="标题")
    items: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
    inner: ComplexInner | None = None


# ==========================
# Schema 工具测试
# ==========================

class TestSchemaTools:
    """Schema 相关工具函数测试"""

    def test_to_openai_tool_basic(self):
        """基础模型的 OpenAI Tool Schema 生成"""
        tool = StructureEngine.to_openai_tool(User)
        assert tool["type"] == "function"

        fn = tool["function"]
        assert fn["name"]  # 有名字
        assert "parameters" in fn

        params = fn["parameters"]
        props = params["properties"]
        assert "name" in props and "age" in props

    def test_to_openai_tool_flatten_defs(self):
        """带嵌套模型时 $defs 展开逻辑测试"""
        tool = StructureEngine.to_openai_tool(ComplexModel)
        params = tool["function"]["parameters"]

        # schema 中应当已经没有 $defs
        assert "$defs" not in params
        # 检查 inner 属性存在（证明嵌套模型定义被展开）
        assert "inner" in params["properties"]

    def test_get_schema_diff_and_type_mismatch(self):
        """get_schema_diff 与类型不匹配检测测试"""
        data = {"name": "Alice", "age": "not-int", "extra": 1}

        diff = StructureEngine.get_schema_diff(data, User)

        # age 在数据中存在，但类型不匹配
        assert "age" not in diff["missing_required"]
        # extra 为多余字段
        assert "extra" in diff["extra_fields"]

        mismatches = diff["type_mismatches"]
        assert any(m["field"] == "age" for m in mismatches)


# ==========================
# Engine + json_extractor 测试
# ==========================

class TestJsonExtractorCore:
    """核心解析路径测试：Engine + json_extractor"""

    @pytest.mark.asyncio
    async def test_engine_parse_direct_json(self):
        """直接 JSON 解析"""
        user = await StructureEngine.parse('{"name": "Alice", "age": 25}', User)
        assert user.name == "Alice"
        assert user.age == 25

    @pytest.mark.asyncio
    async def test_engine_parse_markdown_json(self):
        """Markdown 代码块中的 JSON 解析"""
        markdown = """
        prefix
        ```json
        {"name": "Bob", "age": 30}
        ```
        postfix
        """
        user = await StructureEngine.parse(markdown, User)
        assert user.name == "Bob"
        assert user.age == 30

    @pytest.mark.asyncio
    async def test_engine_parse_from_tool_call(self):
        """从 tool_calls 中解析"""
        tool_calls = [
            {"function": {"arguments": '{"name": "Charlie", "age": 35}'}}
        ]
        user = await StructureEngine.parse(
            "",
            User,
            raw_tool_calls=tool_calls,
        )
        assert user.name == "Charlie"
        assert user.age == 35

    @pytest.mark.asyncio
    async def test_engine_parse_braced_json(self):
        """括号匹配提取 {...}"""
        text = 'some text {"name": "David", "age": 40} other'
        user = await StructureEngine.parse(text, User)
        assert user.name == "David"
        assert user.age == 40

    @pytest.mark.asyncio
    async def test_engine_parse_bracket_array_json(self):
        """顶层数组 JSON 解析（[...]）"""
        text = 'prefix [{"name": "Eva", "age": 22}] suffix'

        # 使用 Pydantic v2 推荐的 RootModel，而不是 BaseModel + __root__
        class UserList(RootModel[list[User]]):
            pass

        # 直接调用底层 extractor 测试数组场景
        result = json_extractor.extract_structured_data(
            text,
            UserList,
        )

        # RootModel 的数据在 .root 中
        assert isinstance(result.root, list)
        assert result.root[0].name == "Eva"
        assert result.root[0].age == 22

    @pytest.mark.asyncio
    async def test_engine_parse_invalid_raises_parse_error(self):
        """语法错误 JSON，最终应抛 StructureParseError"""
        with pytest.raises(StructureParseError):
            await StructureEngine.parse("{invalid json}", User)

    @pytest.mark.asyncio
    async def test_engine_parse_missing_brace_fast_fail(self):
        """不包含 '{' 或 '['，触发 fast-fail"""
        with pytest.raises(StructureParseError) as exc:
            await StructureEngine.parse("no json here", User)
        assert "missing '{' or '['" in str(exc.value)

    @pytest.mark.asyncio
    async def test_engine_parse_auto_fix_trailing_comma(self):
        """自动修复尾逗号场景"""
        dirty_json = """
        {
            "name": "Eve",
            "age": 28,
        }
        """
        user = await StructureEngine.parse(dirty_json, User, auto_fix=True)
        assert user.name == "Eve"
        assert user.age == 28

    @pytest.mark.asyncio
    async def test_engine_parse_validation_error_path(self):
        """JSON 语法正确但模型校验失败的路径"""
        invalid = '{"name": "OnlyName"}'  # 缺少 age
        with pytest.raises(StructureParseError) as exc:
            await StructureEngine.parse(invalid, User)
        # 验证 attempts 中记录了 direct_json 策略错误
        assert any(a["strategy"] == "direct_json" for a in exc.value.attempts)

    def test_extract_structured_data_dos_protection(self):
        """超长文本截断 + warning 日志测试"""
        # 构造超长文本，同时刻意缺少 age 字段以触发校验失败
        huge_text = '{"name": "Huge"}' + " " * 200000

        with patch.object(json_extractor, "logger") as mock_logger:
            # 预期：解析失败，最终抛出 StructureParseError
            with pytest.raises(StructureParseError):
                json_extractor.extract_structured_data(
                    huge_text,
                    User,
                    max_text_length=1000,
                )

            # 验证触发 warning 日志，并携带长度信息
            mock_logger.warning.assert_called()
            msg = mock_logger.warning.call_args[0][0]
            kwargs = mock_logger.warning.call_args[1]
            assert "truncating" in msg
            assert kwargs.get("original_length") > kwargs.get("max_length")


# ==========================
# Strategy 插件机制测试
# ==========================

class TestPluginStrategies:
    """Strategy 插件机制测试"""

    def test_custom_plugin_success(self):
        """自定义插件成功解析场景"""
        # 备份当前插件列表
        original_strategies = list(json_extractor._EXTRA_STRATEGIES)
        json_extractor._EXTRA_STRATEGIES.clear()

        # 自定义插件：忽略文本，直接构造模型
        def plugin_func(text, model_class):
            return model_class(name="PluginUser", age=99)

        strategy = ExtractionStrategy(name="test_plugin", func=plugin_func)
        register_extraction_strategy(strategy)

        # 构造无法被 JSON 正常解析的文本，但包含 '[' 以绕过 fast-fail
        text = "not json but [trigger] for plugin"

        result = json_extractor.extract_structured_data(text, User)
        assert result.name == "PluginUser"
        assert result.age == 99

        # 恢复原插件列表
        json_extractor._EXTRA_STRATEGIES.clear()
        json_extractor._EXTRA_STRATEGIES.extend(original_strategies)

    def test_custom_plugin_failure_accumulates_attempt(self):
        """自定义插件失败时，attempts 中应记录 plugin_xxx 策略"""
        original_strategies = list(json_extractor._EXTRA_STRATEGIES)
        json_extractor._EXTRA_STRATEGIES.clear()

        def bad_plugin(text, model_class):
            raise RuntimeError("plugin failed")

        strategy = ExtractionStrategy(name="bad_plugin", func=bad_plugin)
        register_extraction_strategy(strategy)

        text = "still not {valid} json but [trigger]"

        with pytest.raises(StructureParseError) as exc:
            json_extractor.extract_structured_data(text, User)

        assert any(
            a["strategy"] == "plugin_bad_plugin" for a in exc.value.attempts
        )

        json_extractor._EXTRA_STRATEGIES.clear()
        json_extractor._EXTRA_STRATEGIES.extend(original_strategies)

    def test_yaml_plugin_if_available(self):
        """
        如果环境安装了 PyYAML，则应存在 yaml_fulltext 策略并可正常工作；
        如果未安装，则跳过本测试。
        """
        pytest.importorskip("yaml")  # 无 PyYAML 时自动 skip

        import gecko.core.structure.json_extractor as je_reload
        importlib.reload(je_reload)

        yaml_strategies = [
            s for s in je_reload._EXTRA_STRATEGIES if s.name == "yaml_fulltext"
        ]
        assert yaml_strategies, "yaml_fulltext strategy not registered"

        yaml_strategy = yaml_strategies[0]

        class YamlModel(BaseModel):
            host: str
            port: int

        yaml_text = "host: localhost\nport: 8080\n"

        result = yaml_strategy.func(yaml_text, YamlModel)
        assert result.host == "localhost" # type: ignore
        assert result.port == 8080 # type: ignore


# ==========================
# Sync 工具函数测试
# ==========================

class TestSyncHelpers:
    """同步辅助函数测试"""

    def test_extract_json_from_text_basic(self):
        """从普通文本中提取内嵌 JSON 对象"""
        text = 'Here is data: {"key": "value"} end'
        result = extract_json_from_text(text)
        assert result is not None
        assert result["key"] == "value"

    def test_extract_json_from_text_array_pick_first_object(self):
        """文本中包含数组 JSON 时，选取第一个对象"""
        text = 'prefix [{"k": 1}, {"k": 2}] suffix'
        result = extract_json_from_text(text)
        assert result is not None
        assert result["k"] == 1

    def test_extract_json_from_text_none(self):
        """确实没有 JSON 时返回 None"""
        text = "No JSON here"
        result = extract_json_from_text(text)
        assert result is None

    def test_parse_structured_output_sync_ok(self):
        """同步 parse_structured_output 正常解析"""
        content = '{"name": "SyncUser", "age": 18}'
        user = parse_structured_output(content, User)
        assert user.name == "SyncUser"
        assert user.age == 18

    @pytest.mark.asyncio
    async def test_parse_structured_output_in_async_raises(self):
        """在已有事件循环中调用 parse_structured_output 应抛 RuntimeError"""
        content = '{"name": "AsyncUser", "age": 20}'
        with pytest.raises(RuntimeError):
            parse_structured_output(content, User)


# ==========================
# StructureParseError 测试
# ==========================

class TestStructureParseError:
    """异常类型测试"""

    @pytest.mark.asyncio
    async def test_detailed_error_from_engine(self):
        """通过 Engine 抛出的 StructureParseError 的详细信息"""
        try:
            await StructureEngine.parse("invalid json content", User)
        except StructureParseError as e:
            detailed = e.get_detailed_error()
            assert "结构化解析失败" in detailed
            assert "尝试的解析策略" in detailed
            assert len(e.attempts) > 0

    def test_detailed_error_manual(self):
        """手动构造 StructureParseError 测试 get_detailed_error"""
        err = StructureParseError(
            "test error",
            attempts=[{"strategy": "s1", "error": "e1"}],
            raw_content="raw\ncontent",
        )
        msg = err.get_detailed_error()
        assert "test error" in msg
        assert "s1" in msg
        # 原始内容中的换行应被转义为 \n
        assert "raw\\ncontent" in msg
