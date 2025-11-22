# Python 文件水印 & Git 提交流水线 SOP

本仓库集成了一套 **Python 源码水印工具 + Git 提交自动签名流水线**，用于在提交前为 `.py` 文件自动添加作者水印和内容签名（Hash）。

核心目标：

- 每个 Python 文件开头带有统一格式的水印：
  - Author / Email / Copyright / License / Signature
- `Signature` 基于“代码内容 + 文件路径 + Salt”生成，可用于简单完整性校验。
- 开发过程中改动代码后，在提交前自动同步更新 `Signature`。

---

## 1. 文件结构概览

推荐结构：

```text
.
├── commit.sh                  # 一键提交脚本（可选）
├── scripts
│   └── waterprint.py          # 水印工具
└── .git
    └── hooks
        └── pre-commit         # git 钩子：提交前自动更新水印
````

---

## 2. 环境准备

### 2.1 Git 基本信息

确保 git 已配置用户名和邮箱（会写入水印）：

```bash
git config --global user.name  "Your Name"
git config --global user.email "you@example.com"
```

> `waterprint.py`、`commit.sh` 和 `pre-commit` 都会从 `git config` 中自动读取 `Author` / `Email`。

### 2.2 配置 Salt（签名盐）

签名中会用到一个 `Salt`，用于增强签名唯一性。
**重要：整个仓库生命周期请保持同一个 Salt，否则签名会全部重算。**

推荐在 shell 配置文件（例如 `~/.bashrc` 或 `~/.zshrc`）中加入：

```bash
export WATERPRINT_SALT="404222"   # 改成你自己的一串随机值
```

然后重新加载：

```bash
source ~/.bashrc
# 或
source ~/.zshrc
```

如果不配置环境变量，则脚本会使用默认值 `404222`。

---

## 3. 安装步骤

1. **拷贝脚本**

   * 将 `scripts/waterprint.py` 放到仓库根目录下的 `scripts/` 目录；
   * 将 `commit.sh` 放到仓库根目录；
   * 将 `pre-commit` 放到 `.git/hooks/pre-commit`；

2. **赋予执行权限**

   ```bash
   chmod +x commit.sh
   chmod +x .git/hooks/pre-commit
   ```

3. **（可选）先为整个项目批量加一次水印**

   在项目根目录执行：

   ```bash
   python3 scripts/waterprint.py \
     -d . \
     -a "$(git config --get user.name)" \
     -e "$(git config --get user.email)" \
     -s "${WATERPRINT_SALT:-404222}"
   ```

   这一步会为仓库内所有 `.py` 文件添加水印和初始签名。

---

## 4. 日常开发流程

### 4.1 正常使用（推荐）

安装好 `pre-commit` 钩子后，你可以像平时一样使用 Git：

```bash
git add path/to/your_file.py
git commit -m "feat: xxx"
git push
```

pre-commit 钩子会在 `git commit` 之前自动执行：

1. 找出 **本次已暂存的 `.py` 文件**；

2. 对每个文件调用：

   ```bash
   python3 scripts/waterprint.py \
     -f <file> \
     -a "$(git config --get user.name)" \
     -e "$(git config --get user.email)" \
     -s "${WATERPRINT_SALT:-404222}" \
     --ensure \
     --no-backup
   ```

   * 如该文件**已有你的水印** → 只更新 `Signature`；
   * 如该文件**还没有你的水印** → 自动加上完整水印（含签名）；

3. 自动重新 `git add` 这些文件，保证提交的是最新签名版本。

### 4.2 使用 `commit.sh` 一键提交（可选）

如果你喜欢用脚本统一提交，可以使用：

```bash
./commit.sh "feat: 某某功能"
```

其流程是：

1. 找出相对 `HEAD` 变化的 `.py` 文件（包括新增）；
2. 调用 `waterprint.py --ensure` 为这些文件添加或更新水印；
3. `git add .`
4. `git commit -m "你的 message"`
5. `git push`

> 注意：
>
> * 如果同时启用了 `pre-commit`，`commit.sh` 里的 ensure + pre-commit 里的 ensure 会**各跑一遍**，但 `--ensure` 是幂等的，不会出问题，只是多几次 I/O。
> * 你也可以把 `commit.sh` 改成只做 `git commit && git push`，把水印逻辑完全交给 pre-commit。

---

## 5. 典型命令示例

### 5.1 单文件添加水印

```bash
python3 scripts/waterprint.py \
  -f scripts/publish.py \
  -a "Xuemin" \
  -e "zxm0813@gmail.com" \
  -s "404222"
```

### 5.2 单文件 ensure（有则更新签名，无则添加水印）

```bash
python3 scripts/waterprint.py \
  -f scripts/publish.py \
  -a "Xuemin" \
  -e "zxm0813@gmail.com" \
  -s "404222" \
  --ensure
```

### 5.3 目录批量验证签名

```bash
python3 scripts/waterprint.py \
  -d . \
  -a "Xuemin" \
  -s "404222" \
  --verify
```

### 5.4 目录批量移除水印（带签名校验）

```bash
python3 scripts/waterprint.py \
  -d . \
  -a "Xuemin" \
  -s "404222" \
  --remove
```

### 5.5 强制移除水印（不校验签名，慎用）

```bash
python3 scripts/waterprint.py \
  -d . \
  -a "Xuemin" \
  --force-remove
```

---

## 6. 注意事项 & 建议

1. **Salt 一定要稳定**
   不要在同一仓库中频繁更换 `WATERPRINT_SALT`，否则历史文件的签名验证都会失败。

2. **多人协作时的约定**

   * 建议团队内统一使用此工具；
   * 每个人使用自己的 `git user.name / user.email`，水印中会自然记录是谁改的；
   * `Salt` 可以团队统一一个，例如由项目 Owner 生成并通知团队成员配置。

3. **与代码格式化工具（black/isort 等）共存**
   签名的计算是基于“规范化后的无水印源码”，对末尾空白和多余空行不敏感，可以和 black/isort 等 Formatter 共存。

4. **有需要时可以扩展的方向**

   * 在 CI 里加一步：验证所有 Python 文件的签名是否有效；
   * 为水印增加更多字段，如仓库名、分支名、版本号等；
   * 为每次提交在 CI 生成一个签名报告。