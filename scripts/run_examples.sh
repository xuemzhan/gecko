#!/bin/bash

# Python示例执行工具
# 用于执行examples目录中的Python脚本

# 不使用 set -e，我们手动处理错误
set -u
set -o pipefail

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 脚本目录和项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
EXAMPLES_DIR="${PROJECT_ROOT}/examples"

# 显示帮助信息
show_help() {
    cat << EOF
${BLUE}Python示例执行工具${NC}

用法:
    $0 [选项] [模块/文件]

选项:
    -h, --help              显示帮助信息
    -l, --list              列出所有可用的模块和脚本
    -a, --all               执行所有脚本
    -m, --module <模块名>   执行指定模块中的所有脚本
    -f, --file <文件路径>   执行指定的脚本文件
    -v, --verbose           显示详细输出
    -s, --skip-errors       遇到错误继续执行

示例:
    $0 --list                           # 列出所有模块和脚本
    $0 --all                            # 执行所有脚本
    $0 -m core                          # 执行core模块中的所有脚本
    $0 -f fast_dev_demo.py              # 执行指定脚本
    $0 -f core/engine_base_demo.py      # 执行指定路径的脚本
    $0 --all --skip-errors              # 执行所有脚本,忽略错误

EOF
}

# 检查examples目录是否存在
check_examples_dir() {
    if [ ! -d "$EXAMPLES_DIR" ]; then
        echo -e "${RED}错误: examples目录不存在: $EXAMPLES_DIR${NC}"
        exit 1
    fi
}

# 列出所有模块和脚本
list_scripts() {
    echo -e "${BLUE}=== 可用的模块和脚本 ===${NC}\n"
    
    # 列出模块
    echo -e "${GREEN}模块:${NC}"
    for dir in "$EXAMPLES_DIR"/*/ ; do
        if [ -d "$dir" ]; then
            module_name=$(basename "$dir")
            echo "  - $module_name"
        fi
    done
    
    echo ""
    
    # 列出根目录下的脚本
    echo -e "${GREEN}根目录脚本:${NC}"
    for file in "$EXAMPLES_DIR"/*.py ; do
        if [ -f "$file" ]; then
            echo "  - $(basename "$file")"
        fi
    done
    
    echo ""
    
    # 详细列出每个模块中的脚本
    for dir in "$EXAMPLES_DIR"/*/ ; do
        if [ -d "$dir" ]; then
            module_name=$(basename "$dir")
            echo -e "${YELLOW}[$module_name]${NC}"
            for file in "$dir"/*.py ; do
                if [ -f "$file" ]; then
                    echo "  - $(basename "$file")"
                fi
            done
            echo ""
        fi
    done
}

# 执行单个Python脚本
run_script() {
    local script_path=$1
    local script_name=$(basename "$script_path")
    local relative_path=${script_path#$EXAMPLES_DIR/}
    
    echo -e "${BLUE}>>> 执行: $relative_path${NC}"
    echo ""
    
    # 直接执行，不使用 set +e / set -e
    python3 "$script_path"
    local exit_code=$?
    
    echo ""
    
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}✓ 成功: $relative_path${NC}"
        echo ""
        return 0
    else
        echo -e "${RED}✗ 失败: $relative_path (退出码: $exit_code)${NC}"
        echo ""
        return 1
    fi
}

# 执行指定模块中的所有脚本
run_module() {
    local module_name=$1
    local module_path="$EXAMPLES_DIR/$module_name"
    
    if [ ! -d "$module_path" ]; then
        echo -e "${RED}错误: 模块不存在: $module_name${NC}"
        echo -e "${RED}路径: $module_path${NC}"
        exit 1
    fi
    
    echo -e "${YELLOW}=== 执行模块: $module_name ===${NC}"
    echo -e "${YELLOW}模块路径: $module_path${NC}"
    echo ""
    
    local total=0
    local success=0
    local failed=0
    
    # 先列出找到的文件进行调试
    echo -e "${BLUE}查找Python文件...${NC}"
    local file_count=$(find "$module_path" -maxdepth 1 -name "*.py" -type f | wc -l)
    echo -e "${BLUE}找到 $file_count 个Python文件${NC}"
    echo ""
    
    # 使用简单的 for 循环
    for script in "$module_path"/*.py; do
        # 检查文件是否真实存在
        if [ ! -f "$script" ]; then
            continue
        fi
        
        ((total++))
        
        # 调用 run_script 并检查返回值
        if run_script "$script"; then
            ((success++))
        else
            ((failed++))
            # 如果不是跳过错误模式，则退出
            if [ "${SKIP_ERRORS:-false}" != "true" ]; then
                echo -e "${RED}执行失败,停止运行${NC}"
                exit 1
            fi
        fi
    done
    
    if [ $total -eq 0 ]; then
        echo -e "${YELLOW}警告: 模块 $module_name 中没有找到Python文件${NC}"
        echo ""
        return
    fi
    
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}模块 $module_name 执行完成:${NC}"
    echo -e "  总计: $total"
    echo -e "  ${GREEN}成功: $success${NC}"
    echo -e "  ${RED}失败: $failed${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
}

# 执行所有脚本
run_all() {
    echo -e "${YELLOW}=== 执行所有脚本 ===${NC}"
    echo ""
    
    local total=0
    local success=0
    local failed=0
    
    # 先执行根目录下的脚本
    while IFS= read -r -d '' script; do
        ((total++))
        if run_script "$script"; then
            ((success++))
        else
            ((failed++))
            if [ "$SKIP_ERRORS" != true ]; then
                echo -e "${RED}执行失败,停止运行${NC}"
                exit 1
            fi
        fi
    done < <(find "$EXAMPLES_DIR" -maxdepth 1 -name "*.py" -type f -print0 | sort -z)
    
    # 再执行各模块中的脚本
    while IFS= read -r -d '' dir; do
        local module_name=$(basename "$dir")
        echo -e "${YELLOW}[模块: $module_name]${NC}"
        echo ""
        
        while IFS= read -r -d '' script; do
            ((total++))
            if run_script "$script"; then
                ((success++))
            else
                ((failed++))
                if [ "$SKIP_ERRORS" != true ]; then
                    echo -e "${RED}执行失败,停止运行${NC}"
                    exit 1
                fi
            fi
        done < <(find "$dir" -maxdepth 1 -name "*.py" -type f -print0 | sort -z)
    done < <(find "$EXAMPLES_DIR" -maxdepth 1 -type d ! -path "$EXAMPLES_DIR" -print0 | sort -z)
    
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}执行完成统计:${NC}"
    echo -e "  总计: $total"
    echo -e "  ${GREEN}成功: $success${NC}"
    echo -e "  ${RED}失败: $failed${NC}"
    echo -e "${BLUE}========================================${NC}"
}

# 执行指定文件
run_file() {
    local file_path=$1
    
    # 如果是相对路径,尝试在examples目录中查找
    if [[ ! "$file_path" = /* ]]; then
        if [ -f "$EXAMPLES_DIR/$file_path" ]; then
            file_path="$EXAMPLES_DIR/$file_path"
        elif [ -f "$file_path" ]; then
            file_path=$(realpath "$file_path")
        else
            echo -e "${RED}错误: 文件不存在: $file_path${NC}"
            exit 1
        fi
    fi
    
    if [ ! -f "$file_path" ]; then
        echo -e "${RED}错误: 文件不存在: $file_path${NC}"
        exit 1
    fi
    
    run_script "$file_path"
}

# 主函数
main() {
    check_examples_dir
    
    # 解析参数
    VERBOSE=false
    SKIP_ERRORS=false
    ACTION=""
    TARGET=""
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -l|--list)
                ACTION="list"
                shift
                ;;
            -a|--all)
                ACTION="all"
                shift
                ;;
            -m|--module)
                ACTION="module"
                TARGET="$2"
                shift 2
                ;;
            -f|--file)
                ACTION="file"
                TARGET="$2"
                shift 2
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -s|--skip-errors)
                SKIP_ERRORS=true
                shift
                ;;
            *)
                echo -e "${RED}未知选项: $1${NC}"
                show_help
                exit 1
                ;;
        esac
    done
    
    # 执行相应操作
    case $ACTION in
        list)
            list_scripts
            ;;
        all)
            run_all
            ;;
        module)
            if [ -z "$TARGET" ]; then
                echo -e "${RED}错误: 请指定模块名${NC}"
                exit 1
            fi
            run_module "$TARGET"
            ;;
        file)
            if [ -z "$TARGET" ]; then
                echo -e "${RED}错误: 请指定文件路径${NC}"
                exit 1
            fi
            run_file "$TARGET"
            ;;
        *)
            echo -e "${RED}错误: 请指定操作选项${NC}\n"
            show_help
            exit 1
            ;;
    esac
}

# 运行主函数
main "$@"