# gecko/core/prompt/registry.py
"""
Prompt 注册中心 / 版本管理模块

职责：
- 提供一个可编程访问的 Prompt 配置中心，支持：
    - 按 name + version 注册、获取 PromptTemplate；
    - 为 Prompt 附带描述、标签、元数据；
    - 简单的“最新版本”解析逻辑；
- 提供一个全局 default_registry 和若干便捷函数，
  便于在项目中统一管理所有 Prompt。

设计目标：
- 初期以“内存级 Registry”为主，便于快速开发与测试；
- 结构设计为将来扩展为：
    - 从文件 / 数据库加载；
    - 与配置中心 / KV 存储对接；
    - 多环境（dev / staging / prod）隔离 等。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from gecko.core.logging import get_logger
from gecko.core.prompt.template import PromptTemplate

logger = get_logger(__name__)


@dataclass
class PromptRecord:
    """
    单条 Prompt 注册记录。

    字段说明：
        name:
            Prompt 名称，建议使用有层级语义的字符串，例如：
                - "chat.default"
                - "summarization.short"
                - "code.review"
        version:
            版本号字符串，例如：
                - "v1"
                - "1.0.0"
                - "2025-11-29"
              没有特别限定格式，由上层约定。
        template:
            实际的 PromptTemplate 实例。
        description:
            人类可读的描述，用于管理界面或文档展示。
        tags:
            标签集合，例如 {"chat", "production", "zh-CN"}；
        metadata:
            任意额外信息，例如 {"owner": "BMS team", "domain": "battery"}。
    """

    name: str
    version: str
    template: PromptTemplate
    description: Optional[str] = None
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)


class PromptRegistry:
    """
    Prompt 注册中心（Registry）

    内部数据结构：
        - _store: Dict[name, Dict[version, PromptRecord]]

    核心方法：
        - register()       : 注册或更新某个 name/version 的记录；
        - get()            : 获取指定 name/version 的 PromptTemplate；
        - get_record()     : 获取完整 PromptRecord 对象；
        - remove()         : 删除指定记录；
        - list_records()   : 按条件列出所有记录；
        - resolve_version(): 决定在未指定 version 时选择哪一个版本。
    """

    def __init__(self) -> None:
        self._store: Dict[str, Dict[str, PromptRecord]] = {}

    # ========== 注册与更新 ==========

    def register(
        self,
        name: str,
        template: PromptTemplate,
        version: str = "default",
        description: Optional[str] = None,
        tags: Optional[Iterable[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        overwrite: bool = True,
    ) -> PromptRecord:
        """
        注册或更新一个 Prompt 记录。

        参数：
            name:
                Prompt 名称；
            template:
                PromptTemplate 实例；
            version:
                版本号字符串，默认 "default"；
            description:
                描述信息；
            tags:
                标签集合（可选）；
            metadata:
                任意元数据字典；
            overwrite:
                若为 False 且同名同版本已存在，则抛出 ValueError；
                若为 True，则覆盖原有记录。

        返回：
            新注册（或更新后）的 PromptRecord。
        """
        if name not in self._store:
            self._store[name] = {}

        existing = self._store[name].get(version)
        if existing and not overwrite:
            raise ValueError(
                f"Prompt '{name}' version '{version}' 已存在，且 overwrite=False。"
            )

        record = PromptRecord(
            name=name,
            version=version,
            template=template,
            description=description,
            tags=set(tags or []),
            metadata=dict(metadata or {}),
        )
        self._store[name][version] = record

        logger.info(
            "Registered prompt",
            extra={
                "name": name,
                "version": version,
                "tags": list(record.tags),
            },
        )

        return record

    # ========== 获取与删除 ==========

    def get(
        self,
        name: str,
        version: Optional[str] = None,
        raise_if_missing: bool = True,
    ) -> Optional[PromptTemplate]:
        """
        获取指定 name + version 的 PromptTemplate。

        参数：
            name:
                Prompt 名称；
            version:
                版本号；若为 None，则调用 resolve_version(name) 决定采用哪个版本；
            raise_if_missing:
                若为 True 且未找到，则抛 KeyError；
                若为 False 则返回 None。

        返回：
            PromptTemplate 实例或 None。
        """
        record = self.get_record(
            name=name,
            version=version,
            raise_if_missing=raise_if_missing,
        )
        return record.template if record else None

    def get_record(
        self,
        name: str,
        version: Optional[str] = None,
        raise_if_missing: bool = True,
    ) -> Optional[PromptRecord]:
        """
        获取完整 PromptRecord 记录。

        逻辑与 get() 类似，但返回的是包含描述、标签等信息的对象。
        """
        if name not in self._store:
            if raise_if_missing:
                raise KeyError(f"Prompt '{name}' 不存在。")
            return None

        versions_map = self._store[name]
        if version is None:
            # 未指定版本时调用内部版本解析逻辑
            version = self.resolve_version(name, versions_map)
            if version is None:
                if raise_if_missing:
                    raise KeyError(f"Prompt '{name}' 没有任何可用版本。")
                return None

        record = versions_map.get(version)
        if record is None and raise_if_missing:
            raise KeyError(f"Prompt '{name}' 的版本 '{version}' 不存在。")

        return record

    def remove(
        self,
        name: str,
        version: Optional[str] = None,
    ) -> None:
        """
        删除指定 name + version 的记录。

        行为：
            - 若 version 为 None，则删除该 name 下的所有版本；
            - 否则仅删除指定版本；
            - 若删除后该 name 无任何版本，会从 _store 中移除该键。
        """
        if name not in self._store:
            return

        if version is None:
            del self._store[name]
            logger.info("Removed all versions of prompt", extra={"name": name})
            return

        versions_map = self._store[name]
        if version in versions_map:
            del versions_map[version]
            logger.info(
                "Removed prompt version",
                extra={"name": name, "version": version},
            )

        if not versions_map:
            del self._store[name]

    # ========== 列表与搜索 ==========

    def list_records(
        self,
        name: Optional[str] = None,
        tags: Optional[Iterable[str]] = None,
    ) -> List[PromptRecord]:
        """
        列出所有匹配条件的 PromptRecord。

        参数：
            name:
                若提供，仅返回该 name 相关的记录；
            tags:
                若提供，仅返回包含所有这些标签的记录（子集关系）。

        返回：
            PromptRecord 列表（不保证顺序）。
        """
        results: List[PromptRecord] = []
        tag_set = set(tags or [])

        def record_matches(rec: PromptRecord) -> bool:
            if tag_set and not tag_set.issubset(rec.tags):
                return False
            return True

        if name:
            versions_map = self._store.get(name, {})
            for rec in versions_map.values():
                if record_matches(rec):
                    results.append(rec)
        else:
            for versions_map in self._store.values():
                for rec in versions_map.values():
                    if record_matches(rec):
                        results.append(rec)

        return results

    # ========== 版本解析策略 ==========

    def resolve_version(
        self,
        name: str,
        versions_map: Dict[str, PromptRecord],
    ) -> Optional[str]:
        """
        当调用 get(name, version=None) 时，决定选用哪个版本。

        默认策略：
            1. 若存在 "latest" 版本，则优先返回 "latest"；
            2. 若存在 "default" 版本，则返回 "default"；
            3. 否则，对版本号字符串做一次简单排序，返回“最大”的那个；
               （如果版本号为 "v1"、"v2" 或 "1.0"、"1.2.3"，通常能得到一个合理结果）

        可扩展：
            - 子类可重写本方法，实现更复杂的版本选择逻辑；
            - 例如解析 Semantic Version（semver）并按语义排序。
        """
        if not versions_map:
            return None

        if "latest" in versions_map:
            return "latest"
        if "default" in versions_map:
            return "default"

        # 简单按字典序排序版本号字符串
        all_versions = sorted(versions_map.keys())
        return all_versions[-1]


# ===== 全局默认 Registry + 便捷函数 =====

# 全局默认注册中心，适合大多场景使用。
default_registry = PromptRegistry()


def register_prompt(
    name: str,
    template: PromptTemplate,
    version: str = "default",
    description: Optional[str] = None,
    tags: Optional[Iterable[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    overwrite: bool = True,
) -> PromptRecord:
    """
    使用全局 default_registry 注册一个 Prompt。

    用法示例：
        ```python
        from gecko.core.prompt import (
            PromptTemplate,
            register_prompt,
        )

        tpl = PromptTemplate(
            template="Hello {{ name }}",
            input_variables=["name"],
        )
        register_prompt(
            name="greeting.simple",
            version="v1",
            template=tpl,
            description="简单打招呼模板",
            tags={"greeting", "demo"},
        )
        ```
    """
    return default_registry.register(
        name=name,
        template=template,
        version=version,
        description=description,
        tags=tags,
        metadata=metadata,
        overwrite=overwrite,
    )


def get_prompt(
    name: str,
    version: Optional[str] = None,
    raise_if_missing: bool = True,
) -> Optional[PromptTemplate]:
    """
    从全局 default_registry 获取 PromptTemplate。

    参数：
        name:
            Prompt 名称；
        version:
            版本号；若为 None，则走默认版本解析策略；
        raise_if_missing:
            若为 True 且未找到，则抛 KeyError；否则返回 None。
    """
    return default_registry.get(
        name=name,
        version=version,
        raise_if_missing=raise_if_missing,
    )


def list_prompts(
    name: Optional[str] = None,
    tags: Optional[Iterable[str]] = None,
) -> List[PromptRecord]:
    """
    从全局 default_registry 列出所有匹配条件的 PromptRecord。
    """
    return default_registry.list_records(name=name, tags=tags)


__all__ = [
    "PromptRecord",
    "PromptRegistry",
    "default_registry",
    "register_prompt",
    "get_prompt",
    "list_prompts",
]
