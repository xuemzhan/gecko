#!/usr/bin/env bash
set -euo pipefail

########################################
# 0. 从 git 配置中自动获取作者和邮箱
########################################

AUTHOR="$(git config --get user.name || true)"
EMAIL="$(git config --get user.email || true)"

if [[ -z "$AUTHOR" || -z "$EMAIL" ]]; then
  echo "错误: 未在 git 配置中找到 user.name 或 user.email。"
  echo "请先配置，例如:"
  echo "  git config --global user.name \"Your Name\""
  echo "  git config --global user.email \"you@example.com\""
  exit 1
fi

########################################
# 1. 配置区
########################################

# Salt 必须和你第一次加水印时使用的一致
# 推荐: export WATERPRINT_SALT=你的盐
SALT="${WATERPRINT_SALT:-404222}"

# waterprint.py 的路径 (相对项目根目录)
WATERPRINT_SCRIPT="scripts/waterprint.py"

# Python 解释器
PYTHON_BIN="python3"

########################################
# 2. 参数检查
########################################

if [ $# -lt 1 ]; then
  echo "用法: $0 \"commit message\""
  echo "示例: $0 \"fix: 修复签名验证逻辑\""
  exit 1
fi

COMMIT_MSG="$1"

# 确认在 git 仓库内
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "错误: 当前目录不是 git 仓库"
  exit 1
fi

# 检查 waterprint.py 是否存在
if [ ! -f "$WATERPRINT_SCRIPT" ]; then
  echo "错误: 找不到水印工具脚本: $WATERPRINT_SCRIPT"
  exit 1
fi

########################################
# 3. 找出变更的 Python 文件 (相对 HEAD)
########################################

mapfile -t PY_FILES < <(
  {
    git diff --name-only --diff-filter=ACMRTUXB HEAD || true
    git ls-files --others --exclude-standard || true
  } | sort -u | grep -E '\.py$' || true
)

if [ ${#PY_FILES[@]} -eq 0 ]; then
  echo "未检测到变更的 Python 文件, 跳过水印处理。"
else
  echo "检测到以下变更的 Python 文件:"
  for f in "${PY_FILES[@]}"; do
    echo "  - $f"
  done
  echo

  echo "开始为变更的 Python 文件更新/添加水印签名..."
  for f in "${PY_FILES[@]}"; do
    if [ ! -f "$f" ]; then
      echo "  (跳过, 文件不存在) $f"
      continue
    fi

    echo "  -> 处理 $f"
    "$PYTHON_BIN" "$WATERPRINT_SCRIPT" \
      -f "$f" \
      -a "$AUTHOR" \
      -e "$EMAIL" \
      -s "$SALT" \
      --ensure \
      --no-backup
  done
fi

########################################
# 4. git add .
########################################

echo
echo "执行: git add ."
git add .

########################################
# 5. git commit -m "<comments>"
########################################

echo "执行: git commit -m \"$COMMIT_MSG\""
git commit -m "$COMMIT_MSG"

########################################
# 6. git push
########################################

echo "执行: git push"
git push

echo
echo "✅ 提交完成。"
echo "Author: $AUTHOR"
echo "Email : $EMAIL"
echo "Salt  : $SALT"
