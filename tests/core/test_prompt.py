# tests/core/test_prompt.py
import pytest
from pathlib import Path

from gecko.core.prompt import (
    # 核心模板 & 模板库
    PromptTemplate,
    PromptLibrary,
    # 组合器
    PromptSection,
    PromptComposer,
    # 验证 / Lint
    IssueSeverity,
    PromptIssue,
    PromptValidator,
    lint_prompt,
    # 注册中心 / 版本管理
    PromptRecord,
    PromptRegistry,
    default_registry,
    register_prompt,
    get_prompt,
    list_prompts,
    # Jinja2 环境（含兼容别名）
    get_jinja2_env,
    check_jinja2,
    _get_jinja2_env,
    _check_jinja2,
)


# ============================================================
# PromptTemplate 测试
# ============================================================


class TestPromptTemplate:
    """PromptTemplate 核心功能测试"""

    # ===== 基础功能 =====

    def test_basic_format(self):
        """测试基础格式化"""
        template = PromptTemplate(
            template="Hello, {{ name }}!",
            input_variables=["name"],
        )

        result = template.format(name="Alice")
        assert result == "Hello, Alice!"

    def test_multiple_variables(self):
        """测试多个变量"""
        template = PromptTemplate(
            template="{{ greeting }}, {{ name }}! You are {{ age }}.",
            input_variables=["greeting", "name", "age"],
        )

        result = template.format(greeting="Hi", name="Bob", age=25)
        assert "Hi" in result
        assert "Bob" in result
        assert "25" in result

    def test_missing_variable(self):
        """测试缺少必需变量"""
        template = PromptTemplate(
            template="Hello, {{ name }}!",
            input_variables=["name"],
        )

        with pytest.raises(ValueError, match="缺少必需的模板变量"):
            template.format()

    # ===== Jinja2 功能 =====

    def test_conditional(self):
        """测试条件语句"""
        template = PromptTemplate(
            template="""
{% if premium %}
Premium User
{% else %}
Regular User
{% endif %}
            """,
            input_variables=["premium"],
        )

        result1 = template.format(premium=True)
        assert "Premium User" in result1

        result2 = template.format(premium=False)
        assert "Regular User" in result2

    def test_loop(self):
        """测试循环"""
        template = PromptTemplate(
            template="""
{% for item in items %}
- {{ item }}
{% endfor %}
            """,
            input_variables=["items"],
        )

        result = template.format(items=["a", "b", "c"])
        assert "- a" in result
        assert "- b" in result
        assert "- c" in result

    def test_jinja_undefined_variable_error_message(self):
        """[New] 测试 Jinja2 未定义变量的友好错误信息"""
        template = PromptTemplate(
            template="Hello {{ name }} and {{ unknown }}",
            input_variables=["name"],
        )

        with pytest.raises(ValueError, match="模板变量 'unknown' 未定义"):
            template.format(name="Alice")

    # ===== 变量提取 =====

    def test_extract_variables(self):
        """测试变量提取"""
        template = PromptTemplate(
            template="User {{ user }} asked {{ question }}",
        )

        variables = template.get_variables_from_template()
        assert "user" in variables
        assert "question" in variables

        # 再调用一次，走变量缓存分支（主要为了覆盖内部缓存逻辑）
        variables2 = template.get_variables_from_template()
        assert variables2 == variables

    # ===== 部分填充（partial） =====

    def test_partial_semantics(self):
        """测试部分填充（预绑定变量语义）"""
        template = PromptTemplate(
            template="{{ a }} and {{ b }}",
            input_variables=["a", "b"],
        )

        partial = template.partial(a="fixed")

        # 1）必需变量只剩下 b
        assert partial.input_variables == ["b"]

        # 2）partial 预绑定的值在渲染时生效
        result = partial.format(b="value")
        assert result.strip() == "fixed and value"

        # 3）原模板不受影响
        assert template.input_variables == ["a", "b"]

        # 4）调用时传入参数可以覆盖 partial 的预绑定值
        result2 = partial.format(a="override", b="value2")
        assert result2.strip() == "override and value2"

    # ===== 安全格式化 =====

    def test_format_safe(self):
        """测试安全格式化"""
        template = PromptTemplate(
            template="Hello {{ name }}, you are {{ age }}",
            input_variables=["name", "age"],
        )

        result = template.format_safe(name="Alice")
        assert "Alice" in result
        assert "<MISSING: age>" in result

    def test_format_safe_strategies(self):
        """[New] 测试 format_safe 的缺省值填充策略"""
        template = PromptTemplate(
            template="History: {{ history }}\nUser: {{ input }}\nMissing: {{ unknown }}",
            input_variables=["input"],
        )

        result = template.format_safe(input="Hi")

        # 验证智能默认值
        assert "History: []" in result  # 列表类型默认 []
        assert "User: Hi" in result
        assert "Missing: <MISSING: unknown>" in result  # 未知类型默认标记

    # ===== 工厂方法 =====

    def test_from_examples(self):
        """测试从示例创建 few-shot 模板"""
        examples = [
            {"input": "1+1", "output": "2"},
            {"input": "2+2", "output": "4"},
        ]

        template = PromptTemplate.from_examples(examples)
        assert "1+1" in template.template
        assert "2" in template.template

    def test_from_file_auto_detect_variables(self, tmp_path: Path):
        """[New] 测试 from_file 自动提取变量"""
        file_path = tmp_path / "template.jinja"
        file_path.write_text("Hello {{ who }}", encoding="utf-8")

        tpl = PromptTemplate.from_file(str(file_path))
        assert tpl.template == "Hello {{ who }}"
        # 自动提取到变量
        assert tpl.input_variables == ["who"]

    # ===== 克隆 =====

    def test_clone(self):
        """测试克隆"""
        original = PromptTemplate(
            template="Hello {{ name }}",
            input_variables=["name"],
        )

        cloned = original.clone()
        assert cloned.template == original.template
        assert cloned is not original
        assert cloned.input_variables == original.input_variables

    # ===== 字符串表示 =====

    def test_str(self):
        """测试字符串表示"""
        template = PromptTemplate(template="Test")
        str_repr = str(template)
        assert "PromptTemplate" in str_repr

    # ===== f-string / str.format 模式 =====

    def test_fstring_validation(self):
        """测试 f-string（str.format）语法校验"""
        # 有效
        p1 = PromptTemplate(
            template="Hello {name}",
            template_format="f-string",
        )
        assert p1.input_variables == []  # 默认不自动提取

        # 无效：大括号不匹配
        with pytest.raises(ValueError, match="语法错误"):
            PromptTemplate(
                template="Hello {name",
                template_format="f-string",
            )

        with pytest.raises(ValueError, match="语法错误"):
            PromptTemplate(
                template="Hello name}",
                template_format="f-string",
            )

    def test_fstring_format_and_missing_key(self):
        """[New] 测试 f-string 格式化和缺少变量时的错误信息"""
        # 只声明 name 为必需变量，age 不在 input_variables 中，
        # 这样 pre-check 不会因为 age 缺失而报“缺少必需的模板变量”，
        # 而是交给 str.format 触发 KeyError，再由 _format_fstring 包装成“缺少变量”。
        tpl = PromptTemplate(
            template="Hello {name}, {age}",
            template_format="f-string",
            input_variables=["name"],
        )

        # 正常格式化（提供 name 和 age 都没问题）
        result = tpl.format(name="Alice", age=30)
        assert "Alice" in result
        assert "30" in result

        # 缺少 age -> 触发 _format_fstring 中的 KeyError 分支，
        # 抛出的 ValueError 文案里包含“缺少变量”
        with pytest.raises(ValueError, match="缺少变量"):
            tpl.format(name="OnlyName")



# ============================================================
# PromptLibrary 测试
# ============================================================


class TestPromptLibrary:
    """PromptLibrary 预定义模板测试"""

    def test_get_react_prompt(self):
        """测试 ReAct 模板"""
        template = PromptLibrary.get_react_prompt()

        assert template is not None
        assert "question" in template.input_variables
        assert "tools" in template.input_variables

    def test_get_chat_prompt(self):
        """测试对话模板（含 partial 默认值）"""
        template = PromptLibrary.get_chat_prompt()

        assert template is not None
        # get_chat_prompt 经过 partial 后，只要求 user_input
        assert template.input_variables == ["user_input"]

        # 测试1：完整参数，覆盖 partial + format 路径
        result1 = template.format(
            user_input="Hello",
            system="You are helpful",
            history=[
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"},
            ],
        )
        assert "Hello" in result1
        assert "You are helpful" in result1
        assert "user: Hi".lower() in result1.lower()

        # 测试2：最小参数（依赖 partial 绑定的 system/history 默认值）
        result2 = template.format(user_input="World")
        assert "User: World" in result2

        # 测试3：format_safe 也能正常工作
        result3 = template.format_safe(user_input="HiSafe")
        assert "HiSafe" in result3

    def test_get_summarization_prompt(self):
        """测试摘要模板"""
        template = PromptLibrary.get_summarization_prompt()

        result = template.format(
            text="Long text here...",
            max_words=50,
        )
        assert "Long text here..." in result
        assert "50" in result

    def test_get_extraction_prompt(self):
        """[New] 测试信息提取模板"""
        template = PromptLibrary.get_extraction_prompt()

        result = template.format(
            fields=["name", "age"],
            text="Tom is 18 years old.",
        )
        assert "name" in result
        assert "age" in result
        assert "Tom is 18 years old." in result

    def test_get_translation_prompt(self):
        """[New] 测试翻译模板"""
        template = PromptLibrary.get_translation_prompt()

        result = template.format(
            source_lang="Chinese",
            target_lang="English",
            text="你好",
        )
        assert "Chinese" in result
        assert "English" in result
        assert "你好" in result


# ============================================================
# Edge Cases / 边缘情况测试
# ============================================================


class TestEdgeCases:
    """边缘情况测试"""

    def test_empty_template(self):
        """测试空模板"""
        template = PromptTemplate(template="")
        result = template.format()
        assert result == ""

    def test_no_variables(self):
        """测试无变量模板"""
        template = PromptTemplate(template="Static text")
        result = template.format()
        assert result == "Static text"

    def test_unicode(self):
        """测试 Unicode 字符"""
        template = PromptTemplate(
            template="你好，{{ name }}！",
            input_variables=["name"],
        )

        result = template.format(name="世界")
        assert "你好" in result
        assert "世界" in result


# ============================================================
# PromptComposer 测试
# ============================================================


class TestPromptComposer:
    """Prompt 组合器测试"""

    def test_composer_render_and_to_template(self):
        """测试 composer 的 render 与 to_template 行为"""
        composer = PromptComposer()

        # system 纯文本 Section
        composer.add_text_section(
            name="system",
            text="You are a helpful assistant.",
        )

        # 任务模板 Section
        task_tpl = PromptTemplate(
            template="User question: {{ question }}",
            input_variables=["question"],
        )
        composer.add_template_section(
            name="task",
            template=task_tpl,
        )

        # 直接渲染字符串
        result = composer.render(question="What is AI?")
        assert "You are a helpful assistant." in result
        assert "What is AI?" in result

        # 转为新的 PromptTemplate
        combined_tpl = composer.to_template()
        assert "You are a helpful assistant." in combined_tpl.template
        assert "User question: {{ question }}" in combined_tpl.template
        # input_variables 合并为并集
        assert combined_tpl.input_variables == ["question"]

        # 用新模板再渲染
        result2 = combined_tpl.format(question="Hello?")
        assert "Hello?" in result2

    def test_composer_disable_section(self):
        """测试禁用 Section 不参与渲染"""
        composer = PromptComposer(
            global_separator="\n---\n",
        )

        composer.add_text_section(
            name="system",
            text="SYS",
        )
        tpl = PromptTemplate(
            template="Q: {{ q }}",
            input_variables=["q"],
        )
        composer.add_template_section(
            name="task",
            template=tpl,
        )

        # 先渲染全部
        result1 = composer.render(q="hi")
        assert "SYS" in result1
        assert "Q: hi" in result1

        # 禁用 system，再渲染
        composer.disable_section("system")
        result2 = composer.render(q="hi")
        assert "SYS" not in result2
        assert "Q: hi" in result2


# ============================================================
# PromptValidator / lint_prompt 测试
# ============================================================


class TestPromptValidator:
    """Prompt 验证 / Lint 测试"""

    def test_validator_undeclared_and_unused(self):
        """测试未声明变量和未使用变量规则"""
        tpl = PromptTemplate(
            template="Hello {{ used }}",
            input_variables=["used", "unused"],
        )

        validator = PromptValidator()
        issues = validator.validate(tpl)

        # 应该至少有一个“未使用变量”的 INFO
        codes = {issue.code for issue in issues}
        assert PromptValidator.RULE_UNUSED_INPUT_VAR in codes

    def test_validator_unknown_vars_length_and_banned(self):
        """测试 unknown_vars + length + banned_phrases 规则"""
        tpl = PromptTemplate(
            template="bad phrase {{ secret }} {{ allowed }}",
            input_variables=["allowed", "secret"],
        )

        validator = PromptValidator(
            max_length=10,  # 故意设小触发 PROMPT_TOO_LONG
            length_severity=IssueSeverity.WARNING,
            banned_phrases=["bad phrase"],
        )

        issues = validator.validate(
            tpl,
            allowed_variables={"allowed"},  # secret 会被判定为 unknown
        )

        codes = {issue.code for issue in issues}
        assert PromptValidator.RULE_UNKNOWN_VAR in codes
        assert PromptValidator.RULE_PROMPT_TOO_LONG in codes
        assert PromptValidator.RULE_BANNED_PHRASE in codes

    def test_lint_prompt_helper(self):
        """测试 lint_prompt 便捷函数"""
        tpl = PromptTemplate(
            template="Hello {{ x }}",
            input_variables=[],
        )

        issues = lint_prompt(tpl)
        # 至少会有“未声明变量”或“未使用输入变量”
        assert any(isinstance(i, PromptIssue) for i in issues)


# ============================================================
# PromptRegistry / default_registry 测试
# ============================================================


class TestPromptRegistry:
    """Prompt 注册中心 / 版本管理测试"""

    def test_registry_register_get_list_remove(self):
        """测试独立 PromptRegistry 的完整生命周期"""
        registry = PromptRegistry()

        tpl_v1 = PromptTemplate(
            template="Hello {{ name }} (v1)",
            input_variables=["name"],
        )
        tpl_v2 = PromptTemplate(
            template="Hello {{ name }} (v2)",
            input_variables=["name"],
        )

        # 注册两个版本
        registry.register(
            name="greeting.simple",
            version="v1",
            template=tpl_v1,
            description="v1",
            tags={"greeting"},
        )
        registry.register(
            name="greeting.simple",
            version="latest",
            template=tpl_v2,
            description="v2",
            tags={"greeting", "latest"},
        )

        # 未指定版本时，应走 resolve_version -> latest
        tpl = registry.get("greeting.simple")
        assert tpl is tpl_v2

        # 指定版本获取
        tpl1 = registry.get("greeting.simple", version="v1")
        assert tpl1 is tpl_v1

        # 按标签列出
        records = registry.list_records(tags={"greeting"})
        assert len(records) == 2

        # 删除单个版本
        registry.remove("greeting.simple", version="v1")
        with pytest.raises(KeyError):
            registry.get("greeting.simple", version="v1")

        # 删除所有版本
        registry.remove("greeting.simple")
        with pytest.raises(KeyError):
            registry.get("greeting.simple", raise_if_missing=True)

    def test_default_registry_helpers(self):
        """测试全局 default_registry + 便捷函数"""
        name = "test.default.registry"

        # 先确保干净
        default_registry.remove(name)

        tpl = PromptTemplate(
            template="Hi {{ who }}",
            input_variables=["who"],
        )

        # 注册
        record = register_prompt(
            name=name,
            version="v1",
            template=tpl,
            description="测试用模板",
            tags={"test"},
        )
        assert isinstance(record, PromptRecord)
        assert record.name == name

        # 获取
        tpl_got = get_prompt(name, version="v1")
        assert tpl_got is tpl

        # 使用 name + 默认版本解析（目前由于只有 v1，会解析到 v1）
        tpl_got2 = get_prompt(name)
        assert tpl_got2 is tpl

        # 列出
        records = list_prompts(name=name)
        assert len(records) == 1

        # 清理
        default_registry.remove(name)


# ============================================================
# Jinja 环境 / 兼容别名测试
# ============================================================


class TestJinjaEnvCompat:
    """Jinja2 环境与兼容函数测试"""

    def test_jinja_env_singleton_and_alias(self):
        """测试 get_jinja2_env / check_jinja2 以及兼容别名"""
        # 仅测试在正常安装 jinja2 的环境下能返回对象
        assert check_jinja2() is True
        env1 = get_jinja2_env()
        env2 = get_jinja2_env()
        assert env1 is env2  # 单例

        # 兼容别名也能正常调用
        assert _check_jinja2() is True
        env3 = _get_jinja2_env()
        assert env3 is env1
