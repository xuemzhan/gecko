# scripts/publish.py
import subprocess
import sys

def main():
    print("🚀 发布 Gecko v0.1.0 到 PyPI...")
    subprocess.run(["rye", "build"], check=True)
    subprocess.run(["rye", "publish"], check=True)
    print("✅ 发布成功！https://pypi.org/project/gecko-ai/")

if __name__ == "__main__":
    main()