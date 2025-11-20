#!/bin/bash

# =============================================================================
# Gecko 项目单元测试运行脚本 (Rye Managed)
# 用法: ./scripts/run_tests.sh [module_name|all] [pytest_options]
# =============================================================================

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 检查 rye 是否安装
if ! command -v rye &> /dev/null; then
    echo -e "${RED}Error: 'rye' not found. Please install rye first.${NC}"
    exit 1
fi

# 基础配置
PROJECT_ROOT=$(pwd)
TEST_ROOT="tests"
COV_SETTINGS="--cov=gecko --cov-report=term-missing --cov-report=html:htmlcov"

# 帮助函数
usage() {
    echo -e "${YELLOW}Usage: $0 <target> [extra_args]${NC}"
    echo ""
    echo "Targets:"
    echo "  all         运行所有测试 (Default)"
    echo "  core        仅运行核心模块测试 (tests/core)"
    echo "  compose     仅运行编排模块测试 (tests/compose)"
    echo "  plugins     仅运行插件模块测试 (tests/plugins)"
    echo "  unit        运行基础单元测试 (tests/unit)"
    echo "  integration 运行集成测试 (tests/integration)"
    echo "  utils       运行工具类测试 (tests/utils)"
    echo ""
    echo "Examples:"
    echo "  $0 all"
    echo "  $0 core -v"
    echo "  $0 compose -s -k 'test_workflow'"
    exit 1
}

# 1. 解析目标模块
TARGET=$1
if [ -z "$TARGET" ]; then
    TARGET="all"
else
    shift # 移除第一个参数，剩下的传给 pytest
fi

# 2. 映射测试目录
case "$TARGET" in
    "all")
        TEST_PATH="$TEST_ROOT"
        DESC="All Tests"
        ;;
    "core")
        TEST_PATH="$TEST_ROOT/core"
        DESC="Core Module Tests"
        ;;
    "compose")
        TEST_PATH="$TEST_ROOT/compose"
        DESC="Compose/Workflow Tests"
        ;;
    "plugins")
        TEST_PATH="$TEST_ROOT/plugins"
        DESC="Plugins Tests"
        ;;
    "unit")
        TEST_PATH="$TEST_ROOT/unit"
        DESC="Basic Unit Tests"
        ;;
    "integration")
        TEST_PATH="$TEST_ROOT/integration"
        DESC="Integration Tests"
        ;;
    "utils")
        TEST_PATH="$TEST_ROOT/utils"
        DESC="Utility Tests"
        ;;
    "-h"|"--help")
        usage
        ;;
    *)
        echo -e "${RED}Unknown target: $TARGET${NC}"
        usage
        ;;
esac

# 3. 清理旧的覆盖率数据 (可选)
echo -e "${YELLOW}Cleaning up old coverage data...${NC}"
rm -f .coverage
rm -rf htmlcov

# 4. 构建并执行命令
# rye run pytest <目录> <覆盖率参数> <用户传入的其他参数>
CMD="rye run pytest $TEST_PATH $COV_SETTINGS $@"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}🚀 Running: $DESC${NC}"
echo -e "${GREEN}📂 Target: $TEST_PATH${NC}"
echo -e "${GREEN}💻 Command: $CMD${NC}"
echo -e "${GREEN}========================================${NC}"

# 执行命令
$CMD
EXIT_CODE=$?

# 5. 结果摘要
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}✅ Tests Passed! Coverage report generated.${NC}"
    echo -e "${GREEN}📄 Open 'htmlcov/index.html' to view details.${NC}"
    echo -e "${GREEN}========================================${NC}"
else
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}❌ Tests Failed with exit code $EXIT_CODE.${NC}"
    echo -e "${RED}========================================${NC}"
fi

exit $EXIT_CODE