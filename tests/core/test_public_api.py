# tests/core/test_public_api.py
"""
公共 API 稳定性测试

目的：
1. 验证 gecko 顶层包是否按 v1.0 规范导出核心对象
2. 验证 __version__ 存在且为字符串
3. 为后续 API 稳定性回归提供基础保障
"""

def test_gecko_top_level_exports():
    """
    [新增] 顶层导出对象存在性测试

    若这些对象在未来被重命名/移动，这个测试会第一时间报警，
    提醒我们更新《Gecko 核心 API v1.0 稳定接口规范》与实际代码实现。
    """
    import gecko

    # 版本号
    assert hasattr(gecko, "__version__")
    assert isinstance(gecko.__version__, str)
    assert gecko.__version__  # 非空字符串

    # Agent & Builder
    assert hasattr(gecko, "Agent")
    assert hasattr(gecko, "AgentBuilder")

    # 消息与角色
    assert hasattr(gecko, "Message")
    assert hasattr(gecko, "Role")

    # 输出与 Token 统计
    assert hasattr(gecko, "AgentOutput")
    assert hasattr(gecko, "TokenUsage")

    # 记忆模块
    assert hasattr(gecko, "TokenMemory")
    assert hasattr(gecko, "SummaryTokenMemory")

    # 结构化输出
    assert hasattr(gecko, "StructureEngine")

    # 工作流与多智能体
    assert hasattr(gecko, "Workflow")
    assert hasattr(gecko, "step")
    assert hasattr(gecko, "Next")
    assert hasattr(gecko, "Team")


def test_gecko_version_single_source():
    """
    [新增] 版本号单一来源一致性测试

    确保：
    - gecko.__version__ 与 gecko.version.__version__ 一致
    - 为未来 CLI 版本输出与核心版本号对齐打基础
    """
    import gecko
    from gecko.version import __version__ as internal_version

    assert gecko.__version__ == internal_version
