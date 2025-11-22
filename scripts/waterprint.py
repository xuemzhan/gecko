#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Python文件水印工具
为开源项目的Python文件批量添加/验证/移除/更新作者信息水印
"""

import os
import re
import argparse
import shutil
import hashlib
import secrets
from datetime import datetime

# 全局水印块正则：匹配如下结构：
#   # ========
#   # ...
#   # ========
#   (后面跟1~2个换行)
WATERMARK_REGEX = re.compile(
    r'# =+\n(?:# .*\n)+# =+\n\n?'
)


class PythonWatermarkTool:
    def __init__(self, author, email=None, license_type=None, year=None, salt=None):
        """
        初始化水印工具

        Args:
            author: 作者名称
            email: 作者邮箱(可选)
            license_type: 许可证类型(可选)
            year: 版权年份(可选,默认当前年份)
            salt: 用于生成随机码的盐值(可选,默认随机生成)
        """
        self.author = author
        self.email = email
        self.license_type = license_type
        self.year = year or datetime.now().year
        self.salt = salt or secrets.token_hex(16)

    # ======================== 签名 & 规范化 ========================

    def _normalize_for_signature(self, content: str) -> str:
        """
        为签名计算做内容规范化：
        - 去除每行末尾多余空白
        - 折叠连续空行为单个空行
        - 去掉文件末尾多余空行
        - 统一以 '\n' 结尾

        这样即使加/删水印或编辑时空行略有变化，签名仍可稳定一致。
        """
        lines = content.splitlines()
        normalized = []
        last_blank = False

        for line in lines:
            if line.strip() == "":
                # 连续空行折叠为1行
                if not last_blank:
                    normalized.append("")
                    last_blank = True
            else:
                normalized.append(line.rstrip())
                last_blank = False

        # 去掉末尾多余空行
        while normalized and normalized[-1] == "":
            normalized.pop()

        # 统一加一个结尾换行，保证稳定
        return "\n".join(normalized) + "\n"

    def generate_file_signature(self, normalized_content: str, file_path: str) -> str:
        """
        生成文件的唯一签名码

        要求传入 normalized_content 是已经过 _normalize_for_signature
        处理后的“无水印内容”。

        Args:
            normalized_content: 规范化后的无水印内容
            file_path: 文件(相对)路径

        Returns:
            16字符的十六进制签名码(大写)
        """
        data = f"{normalized_content}{file_path}{self.salt}".encode("utf-8")
        hash_obj = hashlib.sha256(data)
        full_hash = hash_obj.hexdigest()
        return full_hash[:16].upper()

    # ======================== 水印生成 & 检测 ========================

    def generate_watermark(self, signature=None):
        """生成水印文本

        Args:
            signature: 文件签名码(可选)
        """
        lines = [
            "# " + "=" * 70,
            f"# Author: {self.author}",
        ]

        if self.email:
            lines.append(f"# Email: {self.email}")

        lines.append(f"# Copyright (c) {self.year} {self.author}")

        if self.license_type:
            lines.append(f"# License: {self.license_type}")

        if signature:
            lines.append(f"# Signature: {signature}")

        lines.append("# " + "=" * 70)

        return "\n".join(lines) + "\n"

    def has_watermark(self, content: str) -> bool:
        """检查文件是否已包含水印(基于 Author / Copyright 关键字)"""
        watermark_patterns = [
            rf"# Author:\s*{re.escape(self.author)}",
            rf"# Copyright.*{re.escape(self.author)}",
        ]

        for pattern in watermark_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return True
        return False

    # ======================== 文件头特殊注释处理 ========================

    def extract_special_comments(self, content):
        """提取文件开头的特殊注释(shebang和编码声明)"""
        lines = content.split("\n")
        special_lines = []
        start_idx = 0

        # shebang
        if lines and lines[0].startswith("#!"):
            special_lines.append(lines[0])
            start_idx = 1

        # 编码声明
        if start_idx < len(lines):
            encoding_pattern = r"#.*?coding[:=]\s*([-\w.]+)"
            if re.match(encoding_pattern, lines[start_idx]):
                special_lines.append(lines[start_idx])
                start_idx += 1

        # 跳过紧接着的空行（也算在special_lines中）
        while start_idx < len(lines) and not lines[start_idx].strip():
            special_lines.append(lines[start_idx])
            start_idx += 1

        remaining_content = "\n".join(lines[start_idx:])
        return special_lines, remaining_content

    # ======================== 添加水印 ========================

    def add_watermark_to_file(self, file_path, backup=True, base_dir=None):
        """
        为单个文件添加水印

        Args:
            file_path: 文件路径
            backup: 是否备份原文件
            base_dir: 基准目录,用于计算相对路径(用于签名生成)

        Returns:
            (success, message): 处理结果
        """
        try:
            # 读取原始内容
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            # 已有水印则跳过
            if self.has_watermark(content):
                return False, "已存在水印,跳过"

            # 备份原文件
            if backup:
                backup_path = str(file_path) + ".bak"
                shutil.copy2(file_path, backup_path)

            # 计算相对路径
            if base_dir:
                rel_path = os.path.relpath(file_path, base_dir)
            else:
                rel_path = os.path.basename(file_path)

            # 为签名计算做规范化（基于“无水印原始内容”）
            normalized_content = self._normalize_for_signature(content)
            signature = self.generate_file_signature(normalized_content, rel_path)

            # 提取特殊注释
            special_lines, remaining_content = self.extract_special_comments(content)

            # 生成水印块（包含签名）
            watermark = self.generate_watermark(signature)

            # 组合新内容：特殊注释 + 空行 + 水印 + 剩余代码
            new_content_parts = []

            if special_lines:
                new_content_parts.append("\n".join(special_lines))
                new_content_parts.append("")  # 特意加一行空行，让水印更清晰

            new_content_parts.append(watermark)

            if remaining_content.strip():
                new_content_parts.append(remaining_content)

            new_content = "\n".join(new_content_parts)

            # 写回文件
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(new_content)

            return True, f"成功添加水印 (签名: {signature})"

        except Exception as e:
            return False, f"处理失败: {str(e)}"

    def process_directory(self, directory, backup=True, exclude_dirs=None):
        """
        批量处理目录下的所有Python文件（添加水印）
        """
        if exclude_dirs is None:
            exclude_dirs = [".git", "__pycache__", "venv", ".venv", "env"]

        stats = {
            "total": 0,
            "success": 0,
            "skipped": 0,
            "failed": 0,
            "files": [],
        }

        for root, dirs, files in os.walk(directory):
            dirs[:] = [d for d in dirs if d not in exclude_dirs]

            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    stats["total"] += 1

                    success, message = self.add_watermark_to_file(
                        file_path, backup, base_dir=directory
                    )

                    if success:
                        stats["success"] += 1
                        status = "✓"
                    elif "跳过" in message:
                        stats["skipped"] += 1
                        status = "-"
                    else:
                        stats["failed"] += 1
                        status = "✗"

                    rel_path = os.path.relpath(file_path, directory)
                    stats["files"].append((status, rel_path, message))

        return stats

    # ======================== 移除水印 ========================

    def remove_watermark_from_file(
        self, file_path, backup=True, verify_signature=True, base_dir=None
    ):
        """
        从单个文件中移除水印

        Args:
            file_path: 文件路径
            backup: 是否备份原文件
            verify_signature: 是否验证签名(True则只删除签名匹配的水印)
            base_dir: 基准目录

        Returns:
            (success, message): 处理结果
        """
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            # 没有水印则跳过
            if not self.has_watermark(content):
                return False, "未找到水印,跳过"

            # 签名验证
            if verify_signature:
                valid, verify_msg = self.verify_file_signature(file_path, base_dir)
                if not valid:
                    return False, f"签名验证失败,拒绝删除: {verify_msg}"

            # 备份
            if backup:
                backup_path = str(file_path) + ".bak"
                shutil.copy2(file_path, backup_path)

            # 移除水印块
            new_content, count = WATERMARK_REGEX.subn("", content, count=1)
            if count == 0:
                # 正则未匹配到水印块，防止误删
                return False, "未匹配到水印块,可能水印格式已被修改"

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(new_content)

            return True, "成功移除水印"

        except Exception as e:
            return False, f"移除失败: {str(e)}"

    def remove_watermarks_from_directory(
        self, directory, backup=True, verify_signature=True, exclude_dirs=None
    ):
        """
        批量移除目录下所有Python文件的水印
        """
        if exclude_dirs is None:
            exclude_dirs = [".git", "__pycache__", "venv", ".venv", "env"]

        stats = {
            "total": 0,
            "success": 0,
            "skipped": 0,
            "failed": 0,
            "files": [],
        }

        for root, dirs, files in os.walk(directory):
            dirs[:] = [d for d in dirs if d not in exclude_dirs]

            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    stats["total"] += 1

                    success, message = self.remove_watermark_from_file(
                        file_path,
                        backup=backup,
                        verify_signature=verify_signature,
                        base_dir=directory,
                    )

                    if success:
                        stats["success"] += 1
                        status = "✓"
                    elif "跳过" in message:
                        stats["skipped"] += 1
                        status = "-"
                    else:
                        stats["failed"] += 1
                        status = "✗"

                    rel_path = os.path.relpath(file_path, directory)
                    stats["files"].append((status, rel_path, message))

        return stats

    def print_removal_report(self, stats):
        """打印移除水印报告"""
        print("\n" + "=" * 70)
        print("水印移除报告")
        print("=" * 70)
        print(f"总文件数: {stats['total']}")
        print(f"成功移除: {stats['success']}")
        print(f"无水印(跳过): {stats['skipped']}")
        print(f"失败/拒绝: {stats['failed']}")
        print("\n文件详情:")
        print("-" * 70)

        for status, file_path, message in stats["files"]:
            print(f"{status} {file_path:50s} {message}")

        print("=" * 70)

    # ======================== 验证签名 ========================

    def verify_file_signature(self, file_path, base_dir=None):
        """
        验证文件签名是否匹配

        Args:
            file_path: 文件路径
            base_dir: 基准目录

        Returns:
            (valid, message): 验证结果
        """
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            # 提取签名字段
            signature_match = re.search(
                r"# Signature:\s*([A-F0-9]{16})", content
            )
            if not signature_match:
                return False, "未找到签名"

            stored_signature = signature_match.group(1)

            # 去掉水印块, 得到“无水印内容”
            content_without_watermark, count = WATERMARK_REGEX.subn("", content, count=1)
            if count == 0:
                return False, "未匹配到水印块,无法验证签名"

            # 计算相对路径
            if base_dir:
                rel_path = os.path.relpath(file_path, base_dir)
            else:
                rel_path = os.path.basename(file_path)

            # 同样对“无水印内容”做规范化，再计算签名
            normalized = self._normalize_for_signature(content_without_watermark)
            calculated_signature = self.generate_file_signature(normalized, rel_path)

            if stored_signature == calculated_signature:
                return True, "签名验证通过"
            else:
                return (
                    False,
                    f"签名不匹配 (存储: {stored_signature}, 计算: {calculated_signature})",
                )

        except Exception as e:
            return False, f"验证失败: {str(e)}"

    def verify_directory(self, directory, exclude_dirs=None):
        """
        批量验证目录下所有Python文件的签名
        """
        if exclude_dirs is None:
            exclude_dirs = [".git", "__pycache__", "venv", ".venv", "env"]

        stats = {
            "total": 0,
            "valid": 0,
            "invalid": 0,
            "no_signature": 0,
            "files": [],
        }

        for root, dirs, files in os.walk(directory):
            dirs[:] = [d for d in dirs if d not in exclude_dirs]

            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    stats["total"] += 1

                    valid, message = self.verify_file_signature(
                        file_path, base_dir=directory
                    )
                    rel_path = os.path.relpath(file_path, directory)

                    if "未找到签名" in message:
                        stats["no_signature"] += 1
                        status = "?"
                    elif valid:
                        stats["valid"] += 1
                        status = "✓"
                    else:
                        stats["invalid"] += 1
                        status = "✗"

                    stats["files"].append((status, rel_path, message))

        return stats

    # ======================== 更新签名 ========================

    def update_signature_for_file(
        self, file_path, backup=True, base_dir=None, skip_if_valid=True
    ):
        """
        更新单个文件水印中的签名码
        使用场景：作者修改了代码，准备提交前希望让水印中的签名
        与当前代码内容重新对齐。

        Args:
            file_path: 文件路径
            backup: 是否备份原文件
            base_dir: 基准目录(用于计算相对路径)
            skip_if_valid: 如果当前签名已匹配，则跳过

        Returns:
            (success, message)
        """
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            if not self.has_watermark(content):
                return False, "未找到水印,跳过"

            sig_match = re.search(r"# Signature:\s*([A-F0-9]{16})", content)
            if not sig_match:
                return False, "水印中未找到签名行"

            old_sig = sig_match.group(1)

            # 删除水印块，得到“无水印内容”
            content_without_watermark, count = WATERMARK_REGEX.subn("", content, count=1)
            if count == 0:
                return False, "未匹配到水印块,无法更新签名"

            # 计算相对路径
            if base_dir:
                rel_path = os.path.relpath(file_path, base_dir)
            else:
                rel_path = os.path.basename(file_path)

            # 对无水印内容做规范化后重新计算签名
            normalized = self._normalize_for_signature(content_without_watermark)
            new_sig = self.generate_file_signature(normalized, rel_path)

            if skip_if_valid and new_sig == old_sig:
                return False, "签名已是最新,跳过"

            # 备份
            if backup:
                backup_path = str(file_path) + ".bak"
                shutil.copy2(file_path, backup_path)

            # 只更新 Signature 行，不动其他水印内容
            new_content = re.sub(
                r"(# Signature:\s*)([A-F0-9]{16})",
                r"\1" + new_sig,
                content,
                count=1,
            )

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(new_content)

            return True, f"签名已更新: {old_sig} -> {new_sig}"

        except Exception as e:
            return False, f"更新失败: {str(e)}"

    def update_signatures_in_directory(
        self, directory, backup=True, exclude_dirs=None, skip_if_valid=True
    ):
        """
        批量更新目录下所有Python文件的水印签名
        """
        if exclude_dirs is None:
            exclude_dirs = [".git", "__pycache__", "venv", ".venv", "env"]

        stats = {
            "total": 0,
            "updated": 0,
            "skipped": 0,
            "failed": 0,
            "files": [],
        }

        for root, dirs, files in os.walk(directory):
            dirs[:] = [d for d in dirs if d not in exclude_dirs]

            for file in files:
                if not file.endswith(".py"):
                    continue

                file_path = os.path.join(root, file)
                stats["total"] += 1

                success, message = self.update_signature_for_file(
                    file_path,
                    backup=backup,
                    base_dir=directory,
                    skip_if_valid=skip_if_valid,
                )

                if success:
                    stats["updated"] += 1
                    status = "✓"
                elif "跳过" in message:
                    stats["skipped"] += 1
                    status = "-"
                else:
                    stats["failed"] += 1
                    status = "✗"

                rel_path = os.path.relpath(file_path, directory)
                stats["files"].append((status, rel_path, message))

        return stats

    def print_update_report(self, stats):
        """打印签名更新报告"""
        print("\n" + "=" * 70)
        print("签名更新报告")
        print("=" * 70)
        print(f"总文件数: {stats['total']}")
        print(f"已更新签名: {stats['updated']}")
        print(f"已是最新/无水印(跳过): {stats['skipped']}")
        print(f"失败: {stats['failed']}")
        print("\n文件详情:")
        print("-" * 70)

        for status, file_path, message in stats["files"]:
            print(f"{status} {file_path:50s} {message}")

        print("=" * 70)

    # ======================== 通用处理报告 ========================

    def print_report(self, stats):
        """打印批量添加水印处理报告"""
        print("\n" + "=" * 70)
        print("处理报告")
        print("=" * 70)
        print(f"总文件数: {stats['total']}")
        print(f"成功添加: {stats['success']}")
        print(f"已存在(跳过): {stats['skipped']}")
        print(f"失败: {stats['failed']}")
        print("\n文件详情:")
        print("-" * 70)

        for status, file_path, message in stats["files"]:
            print(f"{status} {file_path:50s} {message}")

        print("=" * 70)


# =====================================================================
# CLI 主入口
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="为Python项目文件批量添加作者信息水印",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 添加水印
  %(prog)s -d ./my_project -a "张三" -e "zhangsan@example.com" -s "my_salt"
  
  # 验证水印
  %(prog)s -d ./my_project -a "张三" -s "my_salt" --verify
  
  # 移除水印(需要正确的salt)
  %(prog)s -d ./my_project -a "张三" -s "my_salt" --remove
  
  # 强制移除水印(不验证签名)
  %(prog)s -d ./my_project -a "张三" --force-remove
  
  # 更新签名(代码改动后重算签名, 不改水印其它信息)
  %(prog)s -d ./my_project -a "张三" -s "my_salt" --update-signature
  
  # 处理单个文件
  %(prog)s -f test.py -a "李四" -e "lisi@example.com"
        """,
    )

    parser.add_argument("-d", "--directory", help="要处理的目录路径")
    parser.add_argument("-f", "--file", help="要处理的单个文件路径")
    parser.add_argument("-a", "--author", required=True, help="作者名称(必需)")
    parser.add_argument("-e", "--email", help="作者邮箱")
    parser.add_argument(
        "-l",
        "--license",
        help="许可证类型(如: MIT, Apache-2.0, GPL-3.0)",
    )
    parser.add_argument(
        "-y",
        "--year",
        type=int,
        help="版权年份(默认当前年份)",
    )
    parser.add_argument(
        "-s",
        "--salt",
        help="用于生成签名的盐值(可选,建议使用固定值以便后续验证)",
    )
    parser.add_argument(
        "--no-backup", action="store_true", help="不备份原文件"
    )
    parser.add_argument(
        "--exclude",
        nargs="+",
        help="要排除的目录名称(如: .git __pycache__ venv)",
    )
    parser.add_argument(
        "--verify", action="store_true", help="验证模式:检查文件签名是否有效"
    )
    parser.add_argument(
        "--remove",
        action="store_true",
        help="移除模式:删除水印(需要正确的salt验证)",
    )
    parser.add_argument(
        "--force-remove",
        action="store_true",
        help="强制移除水印,不验证签名(谨慎使用)",
    )
    parser.add_argument(
        "--update-signature",
        action="store_true",
        help="更新水印中的签名码(代码修改后使用)",
    )
    parser.add_argument(
        "--ensure",
        action="store_true",
        help="确保文件存在最新水印: 无则添加, 有则更新签名",
    )
    parser.add_argument(
        "--show-salt", action="store_true", help="显示当前使用的salt值"
    )

    args = parser.parse_args()

    if not args.directory and not args.file:
        parser.error("必须指定 -d/--directory 或 -f/--file")

    # 创建水印工具实例
    tool = PythonWatermarkTool(
        author=args.author,
        email=args.email,
        license_type=args.license,
        year=args.year,
        salt=args.salt,
    )

    # 显示salt值
    if args.show_salt:
        print(f"当前使用的Salt值: {tool.salt}")
        print("请保存此值以便后续验证文件签名!")
        print()

    backup = not args.no_backup

    # ============ 批量移除水印(目录模式) ============
    if (args.remove or args.force_remove) and args.directory and not args.file:
        if not os.path.exists(args.directory):
            print(f"错误: 目录不存在: {args.directory}")
            return

        verify_sig = not args.force_remove

        print(f"开始移除水印: {args.directory}")
        print(f"Salt: {tool.salt}")
        print(f"签名验证: {'是' if verify_sig else '否(强制模式)'}")
        print(f"备份: {'是' if backup else '否'}")

        if args.force_remove:
            print("\n⚠️  警告: 强制模式将删除所有水印,不验证签名!")
            confirm = input("确定要继续吗? (输入 'yes' 确认): ")
            if confirm.lower() != "yes":
                print("已取消操作")
                return

        print()

        stats = tool.remove_watermarks_from_directory(
            args.directory,
            backup=backup,
            verify_signature=verify_sig,
            exclude_dirs=args.exclude,
        )

        tool.print_removal_report(stats)
        return

    # ============ 批量更新签名(目录模式) ============
    if args.update_signature and args.directory and not args.file:
        if not os.path.exists(args.directory):
            print(f"错误: 目录不存在: {args.directory}")
            return

        print(f"开始更新目录签名: {args.directory}")
        print(f"Salt: {tool.salt}")
        print(f"备份: {'是' if backup else '否'}")
        print("提示: 请确保使用与初次加水印相同的 Salt 和 Author")
        print()

        stats = tool.update_signatures_in_directory(
            args.directory,
            backup=backup,
            exclude_dirs=args.exclude,
            skip_if_valid=True,
        )

        tool.print_update_report(stats)
        return

    # ============ 验证模式(目录) ============
    if args.verify:
        if not args.directory:
            parser.error("验证模式需要指定 -d/--directory")

        if not os.path.exists(args.directory):
            print(f"错误: 目录不存在: {args.directory}")
            return

        print(f"开始验证目录: {args.directory}")
        print(f"Salt: {tool.salt}")
        print()

        stats = tool.verify_directory(args.directory, exclude_dirs=args.exclude)

        print("\n" + "=" * 70)
        print("验证报告")
        print("=" * 70)
        print(f"总文件数: {stats['total']}")
        print(f"签名有效: {stats['valid']}")
        print(f"签名无效: {stats['invalid']}")
        print(f"无签名: {stats['no_signature']}")
        print("\n文件详情:")
        print("-" * 70)

        for status, file_path, message in stats["files"]:
            print(f"{status} {file_path:50s} {message}")

        print("=" * 70)
        return

    # ============ 处理单个文件 ============
    if args.file:
        if not os.path.exists(args.file):
            print(f"错误: 文件不存在: {args.file}")
            return

        # 单文件 ensure: 有水印则更新签名, 无水印则添加
        if args.ensure:
            success, message = tool.update_signature_for_file(
                args.file,
                backup=backup,
                base_dir=None,
                skip_if_valid=True,
            )
            # 如果提示“未找到水印”, 则改为添加水印
            if (not success) and ("未找到水印" in str(message)):
                success, message = tool.add_watermark_to_file(
                    args.file,
                    backup=backup,
                    base_dir=None,
                )

            print(f"{'✓' if success else '✗'} {args.file}: {message}")
            return

        # 单文件更新签名(只更新, 不自动添加)
        if args.update_signature:
            success, message = tool.update_signature_for_file(
                args.file,
                backup=backup,
                base_dir=None,
                skip_if_valid=True,
            )
            print(f"{'✓' if success else '✗'} {args.file}: {message}")
            return

        # 单文件移除
        if args.remove or args.force_remove:
            verify_sig = not args.force_remove

            if args.force_remove:
                print("⚠️  警告: 强制模式将删除水印,不验证签名!")
                confirm = input("确定要继续吗? (输入 'yes' 确认): ")
                if confirm.lower() != "yes":
                    print("已取消操作")
                    return

            success, message = tool.remove_watermark_from_file(
                args.file, backup=backup, verify_signature=verify_sig
            )
            print(f"{'✓' if success else '✗'} {args.file}: {message}")
            return

        # 单文件添加
        success, message = tool.add_watermark_to_file(
            args.file, backup=backup
        )
        print(f"{'✓' if success else '✗'} {args.file}: {message}")
        return

    # ============ 目录批量添加水印 ============
    if not os.path.exists(args.directory):
        print(f"错误: 目录不存在: {args.directory}")
        return

    print(f"开始处理目录: {args.directory}")
    print(f"作者: {args.author}")
    if args.email:
        print(f"邮箱: {args.email}")
    if args.license:
        print(f"许可证: {args.license}")
    print(f"Salt: {tool.salt}")
    print(f"备份: {'是' if backup else '否'}")
    print("提示: 请保存Salt值以便后续验证文件签名!")
    print()

    stats = tool.process_directory(
        args.directory,
        backup=backup,
        exclude_dirs=args.exclude,
    )

    tool.print_report(stats)


if __name__ == "__main__":
    main()
