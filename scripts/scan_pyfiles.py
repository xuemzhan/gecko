import os
from pathlib import Path


def scan_python_files(source_paths, output_file):
    """
    扫描指定目录或文件列表中的所有Python文件,并将内容整合到一个文件中
    
    参数:
        source_paths: 要扫描的源目录或文件路径列表
        output_file: 输出文件路径
    """
    # 收集所有Python文件
    python_files = []
    
    for source in source_paths:
        source_path = Path(source).resolve()
        
        # 检查路径是否存在
        if not source_path.exists():
            print(f"警告: 路径 '{source}' 不存在,跳过...")
            continue
        
        # 如果是文件
        if source_path.is_file():
            if source_path.suffix == '.py' or source_path.suffix == '.toml':
                python_files.append(source_path)
            else:
                print(f"警告: '{source}' 不是Python文件,跳过...")
        # 如果是目录
        elif source_path.is_dir():
            for root, dirs, files in os.walk(source_path):
                for file in files:
                    if file.endswith('.py'):
                        file_path = Path(root) / file
                        python_files.append(file_path)
    
    # 去重并排序
    python_files = sorted(set(python_files))
    
    if not python_files:
        print(f"没有找到任何Python文件")
        return
    
    # 获取当前工作目录
    current_dir = Path.cwd()
    
    # 写入输出文件
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for index, py_file in enumerate(python_files, start=1):
            # 计算相对于当前工作目录的路径
            try:
                display_path = py_file.relative_to(current_dir)
            except ValueError:
                # 如果无法计算相对路径，使用绝对路径
                display_path = py_file
            
            # 写入文件头信息
            out_f.write(f"[{index}] {display_path}\n")
            out_f.write("```python\n")
            
            # 读取并写入Python文件内容
            try:
                with open(py_file, 'r', encoding='utf-8') as py_f:
                    content = py_f.read()
                    out_f.write(content)
                    # 确保内容以换行结束
                    if content and not content.endswith('\n'):
                        out_f.write('\n')
            except Exception as e:
                out_f.write(f"# 读取文件时出错: {str(e)}\n")
            
            out_f.write("```\n\n")
    
    print(f"成功! 共处理 {len(python_files)} 个Python文件")
    print(f"输出文件: {output_file}")


def main():
    """主函数"""
    print("Python文件扫描整合工具")
    print("=" * 50)
    print("支持输入多个目录或文件路径,用空格、逗号或分号分隔")
    print("例如: /path/to/dir1 /path/to/file.py /path/to/dir2")
    print("或者: /path/to/dir1, /path/to/file.py, /path/to/dir2")
    print("=" * 50)
    
    # 获取输入路径
    paths_input = input("\n请输入要扫描的目录或文件路径: ").strip()
    
    # 分割路径(支持逗号、分号或空格分隔)
    if ',' in paths_input:
        source_paths = [p.strip() for p in paths_input.split(',')]
    elif ';' in paths_input:
        source_paths = [p.strip() for p in paths_input.split(';')]
    else:
        # 使用空格分割
        source_paths = paths_input.split()
    
    # 过滤空路径
    source_paths = [p for p in source_paths if p]
    
    if not source_paths:
        print("错误: 未输入任何路径!")
        return
    
    print(f"\n将扫描以下 {len(source_paths)} 个路径:")
    for i, p in enumerate(source_paths, 1):
        print(f"  {i}. {p}")
    
    # 获取输出文件名
    output_filename = input("\n请输入输出文件名 (默认: python_files_collection.txt): ").strip()
    
    # 设置默认输出文件名
    if not output_filename:
        output_filename = "python_files_collection.txt"
    
    print(f"\n开始扫描...")
    # 执行扫描
    scan_python_files(source_paths, output_filename)


if __name__ == "__main__":
    main()