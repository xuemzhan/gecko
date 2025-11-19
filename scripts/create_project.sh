#!/usr/bin/env bash
# scripts/create_project.sh
# 一键生成 Gecko v0.1 完整工程目录结构
# 在 GitHub Codespaces 终端直接运行：bash scripts/create_project.sh

set -e  # 遇到错误立即退出

PROJECT_ROOT="gecko"
echo "🚀 正在创建 Gecko 项目结构..."

mkdir -p ${PROJECT_ROOT}/gecko/{core,plugins/{models,tools,storage,knowledge,guardrails},compose}
mkdir -p ${PROJECT_ROOT}/examples
mkdir -p ${PROJECT_ROOT}/tests/{unit,integration,benchmark}
mkdir -p ${PROJECT_ROOT}/scripts
mkdir -p ${PROJECT_ROOT}/gecko-compat
mkdir -p ${PROJECT_ROOT}/docs
mkdir -p ${PROJECT_ROOT}/.github/workflows

# core
touch ${PROJECT_ROOT}/gecko/core/__init__.py
touch ${PROJECT_ROOT}/gecko/core/{agent.py,builder.py,runner.py,session.py,message.py,events.py,output.py,exceptions.py}

# plugins
touch ${PROJECT_ROOT}/gecko/plugins/{__init__.py,registry.py,base.py}
touch ${PROJECT_ROOT}/gecko/plugins/models/{__init__.py,litellm.py,openai.py,anthropic.py,gemini.py,groq.py}
touch ${PROJECT_ROOT}/gecko/plugins/tools/{__init__.py,search.py,calculator.py,file.py,python.py}
touch ${PROJECT_ROOT}/gecko/plugins/storage/{__init__.py,postgres_pgvector.py,chroma.py,qdrant.py,milvus.py,redis.py}
touch ${PROJECT_ROOT}/gecko/plugins/knowledge/{__init__.py,base.py,default.py}
touch ${PROJECT_ROOT}/gecko/plugins/guardrails/{__init__.py,pii.py}

# compose
touch ${PROJECT_ROOT}/gecko/compose/{__init__.py,team.py,workflow.py,nodes.py}

# root package
touch ${PROJECT_ROOT}/gecko/__init__.py

# examples
touch ${PROJECT_ROOT}/examples/{basic_agent.py,rag_agent.py,multi_tool.py,team_of_agents.py,workflow_dag.py}

# tests
touch ${PROJECT_ROOT}/tests/__init__.py

# compat & scripts
touch ${PROJECT_ROOT}/gecko-compat/__init__.py
touch ${PROJECT_ROOT}/scripts/create_project.sh  # 自举

# docs & ci
touch ${PROJECT_ROOT}/{pyproject.toml,README.md,LICENSE,CHANGELOG.md,.gitignore}
touch ${PROJECT_ROOT}/.github/workflows/ci.yml

echo "✅ Gecko 项目目录创建完成！"
echo "下一步操作建议："
echo "1. cd gecko && rye init --py 3.12"
echo "2. rye add pydantic anyio litellm networkx opentelemetry"
echo "3. 复制本脚本到 scripts/ 目录（已完成）"
echo "4. 让 AI 开始生成 gecko/core/agent.py 等核心文件"