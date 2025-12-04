# Gecko Compose: Workflow Engine (v0.3)

Gecko Workflow 是一个生产级、基于 DAG（有向无环图）的智能体编排引擎。v0.3 版本经过深度重构，采用了模块化架构，专注于**高并发**、**低 I/O 开销**和**断点续传**能力。

## 🌟 核心特性 (v0.3 新增)

*   **模块化架构**：将核心逻辑拆分为 `Graph`（拓扑）、`Executor`（执行）、`Persistence`（存储）和 `Models`（数据），解耦清晰。
*   **Context Slimming (上下文瘦身)**：
    *   在持久化时自动剥离监控数据（Traces）并裁剪冗余历史。
    *   在大规模长流程中，存储体积减少 **80%+**，显著降低 Redis/DB 的 I/O 压力。
*   **两阶段提交 (Two-Phase Commit)**：
    *   节点执行前保存 `RUNNING` 状态，执行后保存 `SUCCESS` 状态。
    # Gecko Compose: Workflow Engine (v0.4)

    Gecko Compose Workflow 引擎经过 v0.4 版本优化，面向生产级落地：提升并发稳定性、内存与性能效率，并增强可观测性与恢复能力。

    **发布日期**: 2025-12-04

    ---

    ## 🌟 v0.4 核心更新（本次迭代精华）

    - **P0 修复（关键 bug）**:
        - P0-1: 赛马模式（Race）竞态原子性修复，使用异步锁保证 winner 选取原子性。
        - P0-2: Race 失败返回一致性：失败时返回 `MemberResult[]`（包含错误信息），不再返回空列表。
        - P0-3: 动态跳转 `Next` 的 input=None 情形修复：当 `Next.input is None` 时保留上一步 `last_output`，避免被 None 覆盖。
        - P0-4: 跳过节点（条件不满足）记录为 `NodeStatus.SKIPPED`，不写入历史或覆盖 last_output。

    - **P1 性能与内存优化**:
        - 引入轻量级 Copy-On-Write 包装器 `_COWDict`：并行节点读 O(1)（无深拷贝），写时在局部覆盖，仅在合并时取修改键。
        - 历史清理（History Cleanup）：默认有界保留（可配置），避免长时间运行工作流导致 Context 无界增长。

    - **稳定性与可靠性增强**:
        - 实时超时保护：在 `Workflow.execute(..., timeout=seconds)` 与 `Team.run(..., timeout=seconds)` 中支持 `timeout` 参数，基于 `anyio.move_on_after` 实现实时中断并返回或抛出合理错误。
        - 异常与日志增强：节点层与成员执行层记录更丰富的上下文（节点名、session_id、preview），`NodeExecutor` 将异常统一包装为 `WorkflowError`，便于上层统一处理和可观察性。

    - **测试 & 基准**:
        - 单元测试：新增 7 个测试，当前测试套件 `pytest` 全量通过（297/297）。
        - 性能基准：新增基准脚本并生成报告，深历史场景下在 51–501 节点规模上观察到 6–15x 的性能改进。

    ---

    ## 🔧 变更要点与使用说明

    **版本提示**: 本次为向后兼容升级，外部 API 保持稳定，但新增 `timeout` 参数与更合理的并发/错误语义，建议在升级前在测试环境完成一次完整回归。

    ### 1) `Workflow.execute` 支持超时

    ```python
    # 同步/异步皆可，timeout 单位为秒
    result = await wf.execute(input_data, session_id="s1", timeout=30.0)
    ```

    - 超时触发后，工作流会中断当前任务（使用 anyio 的 cancel 机制），并抛出 `WorkflowError`（消息包含发生超时时的步骤计数）。
    - 若使用持久化（`session_id`），中断前的 Pre-Commit/Post-Commit checkpoint 能够保证后续可从最近的 checkpoint 恢复。

    ### 2) `Team.run` / `Team.__call__` 支持超时

    ```python
    team = Team(members=[a1, a2], strategy=ExecutionStrategy.RACE)
    results = await team.run(input_payload, timeout=5.0)
    # 或者
    results = await team(input_payload, timeout=5.0)
    ```

    - 在 `RACE` 模式下如果超时未产生 winner，会抛出超时错误或返回已完成的成员结果（按配置与调用上下文而定）。

    ### 3) Copy-On-Write（并行状态隔离）

    - 引擎在并行执行单层节点时为每个节点创建 `node_context.state = _COWDict(main_context.state)`。
    - 节点读取优先从本地 overlay，写入只修改 overlay；在层合并时仅将 overlay（get_diff）合并到主 `state`。
    - 优点：避免大 history 深拷贝，显著节省内存与拷贝时间，特别是在深历史场景（数十至数百步骤）中表现明显。

    ### 4) History Cleanup（有界保留）

    - 默认行为：`max_history_retention`（或 persistence.history_retention）决定保留最近 N 步的历史（建议默认 20）。
    - 设计：`last_output` 始终保留，便于下一层继续使用；其余旧记录按时间/顺序裁剪。

    ### 5) 日志与异常策略

    - 节点异常：`NodeExecutor` 内部会记录 `logger.exception` 包含节点名、错误摘要与 preview 并抛出 `WorkflowError`，上层 `engine` 会在 TaskGroup 层捕获並记录执行上下文（包括 `session_id`）。
    - 成员执行：`Team._safe_execute_member` 在出现异常时会先调用 `logger.error`（兼容旧测试/告警）再调用 `logger.exception`（记录堆栈），并返回 `MemberResult`（`is_success=False`）。

    ---

    ## 🧪 测试与基准（简要）

    - 单元测试：覆盖关键修复点（P0）与 COW 行为，所有测试通过。
    - 基准脚本位于 `benchmarks/`，包含：
        - `compose_cow_benchmark.py`（快速压力）
        - `compose_cow_detailed_benchmark.py`（多配置对比）
        - `visualize_results.py`（结果可视化）
        - 结果报告 `benchmarks/PERFORMANCE_REPORT.md` 与 `benchmarks/results_cow_performance.json`

    基准摘要示例（部分）：
    - Small (51 nodes): 浅历史 82.1ms → 深历史 5.2ms（≈15.8x）
    - Medium (201 nodes): 浅历史 262.0ms → 深历史 18.3ms（≈14.3x）
    - Large (501 nodes): 浅历史 311.5ms → 深历史 47.5ms（≈6.6x）

    ---

    ## ✅ 升级建议

    - 在升级到 v0.4 之前，请在测试环境中执行：
        1. 完整单元测试：`pytest -q`。
        2. 对关键流程做一次短时压力测试（推荐使用 `benchmarks/compose_cow_benchmark.py`）。
        3. 如需 24 小时稳定性验证，请在独立环境执行长期基准并观察内存与 checkpoint 成长情况。

    - 建议配置：
        - `max_history_retention` 设为 10–20 （根据业务保留历史深度调整）。
        - 在关键外部调用点（模型/远程 API），设置合理的 `timeout`（例如 5–30s）并结合 retry 策略。

    ---

    ## 📦 变更清单（摘录）

    - `gecko/compose/workflow/engine.py`:
        - 添加 `_COWDict` 支持并将节点上下文 state 包装为 COW
        - 添加 `timeout` 参数与实时超时保护
        - 增强 `merge` 與 `next` 语义，防止 None 覆盖
        - 历史清理逻辑（bounded retention）

    - `gecko/compose/team.py`:
        - Race 模式原子性修复（winner lock）
        - 增加 `timeout` 支持與超時处理
        - 改进成员异常日志与返回语义（返回 `MemberResult[]`）

    - `gecko/compose/workflow/executor.py`:
        - 增强异常捕获，记录节点级别 preview，并统一抛出 `WorkflowError`

    - 新增/更新测试與基准脚本（`tests/`, `benchmarks/`）及性能报告

    ---

    ## 联系与贡献

    若在升级或运行中遇到问题，请在仓库中打开 Issue，或在 PR 中附上最小复现示例与日志（`session_id`、`execution_plan` 片段、错误栈）。

    ---

    © 2025 Gecko Project
