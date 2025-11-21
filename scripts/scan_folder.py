import os
from pathlib import Path


def generate_tree(directory, prefix="", is_last=True, output_lines=None, 
                  show_hidden=False, max_depth=None, current_depth=0, exclude_dirs=None):
    """
    生成目录树结构
    
    参数:
        directory: 目录路径
        prefix: 当前行的前缀
        is_last: 是否是最后一个项目
        output_lines: 输出行列表
        show_hidden: 是否显示隐藏文件
        max_depth: 最大深度限制
        current_depth: 当前深度
        exclude_dirs: 要排除的文件夹名称集合
    """
    if exclude_dirs is None:
        exclude_dirs = set()
    if output_lines is None:
        output_lines = []
    
    directory = Path(directory)
    
    # 检查深度限制
    if max_depth is not None and current_depth >= max_depth:
        return output_lines
    
    try:
        # 获取目录内容并排序(目录在前,文件在后)
        items = sorted(directory.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
        
        # 过滤隐藏文件
        if not show_hidden:
            items = [item for item in items if not item.name.startswith('.')]
        
        # 过滤排除的文件夹
        items = [item for item in items if not (item.is_dir() and item.name in exclude_dirs)]
        
        for index, item in enumerate(items):
            is_last_item = (index == len(items) - 1)
            
            # 确定连接符号
            connector = "└── " if is_last_item else "├── "
            
            # 添加当前项
            if item.is_dir():
                output_lines.append(f"{prefix}{connector}{item.name}/")
                
                # 递归处理子目录
                extension = "    " if is_last_item else "│   "
                generate_tree(item, prefix + extension, is_last_item, 
                            output_lines, show_hidden, max_depth, current_depth + 1, exclude_dirs)
            else:
                output_lines.append(f"{prefix}{connector}{item.name}")
    
    except PermissionError:
        output_lines.append(f"{prefix}[Permission Denied]")
    
    return output_lines


def save_tree_to_file(directory, output_file, show_hidden=False, max_depth=None, exclude_dirs=None):
    """
    将目录树保存到文件
    
    参数:
        directory: 要扫描的目录
        output_file: 输出文件路径
        show_hidden: 是否显示隐藏文件
        max_depth: 最大深度限制
        exclude_dirs: 要排除的文件夹名称集合
    """
    if exclude_dirs is None:
        exclude_dirs = set()
    directory = Path(directory).resolve()
    
    if not directory.exists():
        print(f"错误: 目录 '{directory}' 不存在!")
        return
    
    if not directory.is_dir():
        print(f"错误: '{directory}' 不是一个目录!")
        return
    
    print(f"正在扫描目录: {directory}")
    if exclude_dirs:
        print(f"排除的文件夹: {', '.join(sorted(exclude_dirs))}")
    
    # 生成目录树
    tree_lines = [f"{directory.name}/"]
    generate_tree(directory, "", True, tree_lines, show_hidden, max_depth, 0, exclude_dirs)
    
    # 写入文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(tree_lines))
    
    print(f"成功! 共扫描 {len(tree_lines)} 行")
    print(f"目录树已保存到: {output_file}")
    
    # 同时在控制台显示
    print("\n目录结构预览:")
    print("=" * 50)
    for line in tree_lines[:50]:  # 只显示前50行
        print(line)
    if len(tree_lines) > 50:
        print(f"... (还有 {len(tree_lines) - 50} 行,请查看输出文件)")


def main():
    """主函数"""
    print("目录结构扫描工具")
    print("=" * 50)
    
    # 获取目录路径
    directory = input("请输入要扫描的目录路径: ").strip()
    
    if not directory:
        print("错误: 未输入目录路径!")
        return
    
    # 获取输出文件名
    output_file = input("请输入输出文件名 (默认: directory_tree.txt): ").strip()
    if not output_file:
        output_file = "directory_tree.txt"
    
    # 是否显示隐藏文件
    show_hidden_input = input("是否显示隐藏文件? (y/n, 默认: n): ").strip().lower()
    show_hidden = show_hidden_input == 'y'
    
    # 深度限制
    max_depth_input = input("最大扫描深度 (留空表示无限制): ").strip()
    max_depth = None
    if max_depth_input.isdigit():
        max_depth = int(max_depth_input)
    
    # 排除的文件夹
    exclude_input = input("要排除的文件夹名称 (多个用逗号或空格分隔,留空表示不排除): ").strip()
    exclude_dirs = set()
    if exclude_input:
        # 支持逗号或空格分隔
        if ',' in exclude_input:
            exclude_dirs = {name.strip() for name in exclude_input.split(',') if name.strip()}
        else:
            exclude_dirs = {name.strip() for name in exclude_input.split() if name.strip()}
        print(f"将排除以下文件夹: {', '.join(sorted(exclude_dirs))}")
    
    print("\n开始扫描...")
    save_tree_to_file(directory, output_file, show_hidden, max_depth, exclude_dirs)


if __name__ == "__main__":
    main()