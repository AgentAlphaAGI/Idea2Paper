"""
一键运行完整Pipeline
从数据抽取到知识图谱构建的完整流程
"""

import os
import sys
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent


def run_step(step_num: int, name: str, script: str):
    """运行单个步骤"""
    print(f"\n{'='*60}")
    print(f"📌 Step {step_num}: {name}")
    print(f"{'='*60}")
    
    script_path = SCRIPT_DIR / script
    if not script_path.exists():
        print(f"❌ 脚本不存在: {script_path}")
        return False
    
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(SCRIPT_DIR)
    )
    
    if result.returncode != 0:
        print(f"❌ Step {step_num} 失败")
        return False
    
    print(f"✅ Step {step_num} 完成")
    return True


def main():
    print("="*60)
    print("🚀 知识图谱Pipeline - 一键运行")
    print("="*60)
    
    steps = [
        # (0, "数据抽取", "extract_paper_review.py"),  # 已完成，数据在data/
        # (1, "Pattern聚类", "generate_patterns.py"),  # 已完成，结果在output/
        # (2, "构建entity", "build_entity.py"),  # 已完成，结果在output/
        (3, "运行召回", "simple_recall_demo.py"),
    ]
    
    print("\n📋 将执行以下步骤:")
    print("   1. 数据抽取 (已完成 - 结果在 data/)")
    print("   2. Pattern聚类 (已完成 - 结果在 output/patterns_structured.json)")
    print("   3. 构建知识图谱(已完成 - 结果在 output/nodes_xxx.json)")
    print("   4. idea召回")
    
    for step_num, name, script in steps:
        if not run_step(step_num, name, script):
            print(f"\n❌ Pipeline在Step {step_num}中断")
            sys.exit(1)
    
    print("\n" + "="*60)
    print("🎉 Pipeline完成!")
    print("="*60)


if __name__ == '__main__':
    main()
