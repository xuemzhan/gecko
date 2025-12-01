docs/
├── index.md                    # [Landing] 项目简介、核心价值与架构图
├── install.md                  # [Guide] 安装指南 (依赖管理、可选包)
├── quickstart.md               # [Tutorial] 5分钟快速上手 (Agent & RAG)
├── configuration.md            # [Reference] 环境变量与配置项详解
├── concepts/                   # [Explanation] 核心概念详解
│   ├── architecture.md         # 架构设计 (Core/Compose/Plugin 分层)
│   ├── agents_and_tools.md     # Agent 组装与工具箱机制
│   └── memory_and_state.md     # 记忆管理 (TokenWindow) 与 状态持久化
├── guides/                     # [How-to] 核心功能指南
│   ├── workflows.md            # 工作流编排、分支循环与断点恢复 (Resume)
│   ├── rag_pipeline.md         # 知识库构建 (Ingestion) 与检索
│   ├── prompt_management.md    # 模块化 Prompt (Composer/Registry/Lint)
│   ├── structured_output.md    # 结构化输出与自动修复
│   └── observability.md        # OpenTelemetry 追踪与日志
├── plugins/                    # [Reference] 插件生态
│   ├── models.md               # 模型适配 (OpenAI/Zhipu/Ollama)
│   └── storage.md              # 存储后端 (SQLite/Redis/Chroma/LanceDB)
└── api/                        # [Reference] API 字典 (自动生成存根)
    └── public_api.md           # L1 稳定接口列表