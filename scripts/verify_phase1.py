# scripts/verify_phase1.py
"""
Phase 1 改进版验证脚本
运行所有测试并生成报告
"""
import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd: list[str], description: str) -> bool:
    """运行命令并打印结果"""
    print(f"\n{'='*60}")
    print(f"🔍 {description}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode == 0:
        print(f"✅ {description} - PASSED")
        return True
    else:
        print(f"❌ {description} - FAILED")
        return False

def main():
    """主验证流程"""
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    results = {}
    
    # 1. 导入测试
    results["import"] = run_command(
        [sys.executable, "-c", "import gecko; print('✅ Import successful')"],
        "Import Test"
    )
    
    # 2. 单元测试
    results["unit_tests"] = run_command(
        [sys.executable, "-m", "pytest", "tests/unit/", "-v"],
        "Unit Tests"
    )
    
    # 3. 集成测试
    results["integration_tests"] = run_command(
        [sys.executable, "-m", "pytest", "tests/integration/", "-v"],
        "Integration Tests"
    )
    
    # 4. 性能测试
    results["performance_tests"] = run_command(
        [sys.executable, "-m", "pytest", "tests/performance/", "-v", "-s"],
        "Performance Tests"
    )
    
    # 5. 覆盖率测试
    results["coverage"] = run_command(
        [
            sys.executable, "-m", "pytest",
            "tests/",
            "--cov=gecko",
            "--cov-report=term-missing",
            "--cov-report=html"
        ],
        "Coverage Test"
    )
    
    # 生成报告
    print(f"\n{'='*60}")
    print("📊 验证报告")
    print(f"{'='*60}")
    
    total = len(results)
    passed = sum(results.values())
    
    for name, status in results.items():
        icon = "✅" if status else "❌"
        print(f"{icon} {name.replace('_', ' ').title()}")
    
    print(f"\n总计: {passed}/{total} 通过")
    
    if passed == total:
        print("\n🎉 所有验证通过！Phase 1 改进版完成。")
        return 0
    else:
        print(f"\n⚠️  {total - passed} 项验证失败，请检查。")
        return 1

if __name__ == "__main__":
    sys.exit(main())