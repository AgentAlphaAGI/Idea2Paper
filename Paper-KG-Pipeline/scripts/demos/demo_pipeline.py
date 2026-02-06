"""
Idea2Story Pipeline 演示脚本

展示如何快速使用 Pipeline，包含：
1. 基础使用
2. 自定义配置
3. 批量处理
"""

import json
import sys
from pathlib import Path

# 添加脚本目录到路径
SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = SCRIPTS_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / "output"

sys.path.insert(0, str(SCRIPTS_DIR))


# ===================== 示例 1: 基础使用 =====================
def demo_basic_usage():
    """示例1: 基础使用"""
    print("\n" + "=" * 80)
    print("📚 示例 1: 基础使用")
    print("=" * 80)

    from idea2story_pipeline import Idea2StoryPipeline
    
    # 重新定义路径，避免循环引用
    NODES_PATTERN = OUTPUT_DIR / "nodes_pattern.json"
    NODES_PAPER = OUTPUT_DIR / "nodes_paper.json"

    # 用户 Idea
    user_idea = "使用对比学习改进小样本文本分类，并在医疗领域数据集上验证"

    print(f"\n【用户 Idea】\n{user_idea}\n")

    # 加载数据
    print("📂 加载数据...")
    with open(NODES_PATTERN, 'r', encoding='utf-8') as f:
        patterns_data = json.load(f)
    with open(NODES_PAPER, 'r', encoding='utf-8') as f:
        papers_data = json.load(f)

    # 模拟召回结果（简化版）
    print("🔍 运行召回...")
    pattern_map = {p['pattern_id']: p for p in patterns_data}

    # 这里简化为直接使用前 10 个 Pattern
    recalled_patterns = [
        (pid, pattern_map[pid], 0.8 - i * 0.05)
        for i, pid in enumerate(list(pattern_map.keys())[:10])
    ]

    print(f"   召回 {len(recalled_patterns)} 个 Pattern\n")

    # 创建 Pipeline
    print("🚀 启动 Pipeline...")
    pipeline = Idea2StoryPipeline(user_idea, recalled_patterns, papers_data)

    # 运行
    result = pipeline.run()

    # 输出结果
    print("\n" + "=" * 80)
    print("📊 执行结果")
    print("=" * 80)
    print(f"✅ 状态: {'成功' if result['success'] else '需审核'}")
    print(f"📈 迭代次数: {result['iterations']}")
    print(f"📝 最终标题: {result['final_story']['title']}")

    return result


# ===================== 示例 2: 自定义配置 =====================
def demo_custom_config():
    """示例2: 自定义配置"""
    print("\n" + "=" * 80)
    print("📚 示例 2: 自定义配置")
    print("=" * 80)

    from idea2story_pipeline import PipelineConfig

    # 修改配置
    print("\n🔧 修改配置:")
    print(f"   PASS_SCORE: 6.0 → 5.0（降低通过门槛）")
    print(f"   MAX_REFINE_ITERATIONS: 3 → 5（增加迭代次数）")
    print(f"   COLLISION_THRESHOLD: 0.75 → 0.85（放宽查重）")

    original_pass_score = PipelineConfig.PASS_SCORE
    original_max_iter = PipelineConfig.MAX_REFINE_ITERATIONS
    original_threshold = PipelineConfig.COLLISION_THRESHOLD

    PipelineConfig.PASS_SCORE = 5.0
    PipelineConfig.MAX_REFINE_ITERATIONS = 5
    PipelineConfig.COLLISION_THRESHOLD = 0.85

    print("\n💡 提示: 修改后的配置会应用到所有 Pipeline 实例")

    # 恢复原配置
    PipelineConfig.PASS_SCORE = original_pass_score
    PipelineConfig.MAX_REFINE_ITERATIONS = original_max_iter
    PipelineConfig.COLLISION_THRESHOLD = original_threshold


# ===================== 示例 3: 批量处理 =====================
def demo_batch_processing():
    """示例3: 批量处理多个 Idea"""
    print("\n" + "=" * 80)
    print("📚 示例 3: 批量处理")
    print("=" * 80)

    # 多个 Idea
    ideas = [
        "使用知识蒸馏压缩BERT模型用于移动端部署",
        "基于强化学习的对话系统策略优化",
        "多模态融合用于情感分析任务"
    ]

    print(f"\n📋 待处理 Idea 列表: {len(ideas)} 个")
    for i, idea in enumerate(ideas, 1):
        print(f"   {i}. {idea[:40]}...")

    print("\n💡 批量处理示例代码:")
    print("""
    results = []
    for i, idea in enumerate(ideas):
        print(f"\\n处理 {i+1}/{len(ideas)}: {idea[:30]}...")

        # 运行召回
        recalled_patterns = run_recall(idea)

        # 运行 Pipeline
        pipeline = Idea2StoryPipeline(idea, recalled_patterns, papers)
        result = pipeline.run()

        # 保存结果
        results.append(result)
        with open(f"output/story_{i+1}.json", 'w') as f:
            json.dump(result['final_story'], f, ensure_ascii=False, indent=2)

    print(f"\\n✅ 批量处理完成，成功 {sum(r['success'] for r in results)} 个")
    """)


# ===================== 示例 4: 查看中间结果 =====================
def demo_inspect_intermediate():
    """示例4: 查看中间结果"""
    print("\n" + "=" * 80)
    print("📚 示例 4: 查看中间结果")
    print("=" * 80)

    # 检查是否有 pipeline_result.json
    result_file = OUTPUT_DIR / "pipeline_result.json"

    if not result_file.exists():
        print("\n⚠️  未找到 pipeline_result.json")
        print("   请先运行: python scripts/idea2story_pipeline.py")
        return

    # 加载结果
    with open(result_file, 'r', encoding='utf-8') as f:
        result = json.load(f)

    print(f"\n📊 执行历史分析:")
    print(f"   用户 Idea: {result['user_idea'][:50]}...")
    print(f"   总迭代次数: {result['iterations']}")
    print(f"   最终状态: {'✅ 成功' if result['success'] else '❌ 失败'}")

    print(f"\n📋 选择的 Patterns:")
    for ptype, pid in result['selected_patterns'].items():
        print(f"   - {ptype}: {pid}")

    print(f"\n📝 评审历史:")
    review_summary = result['review_summary']
    print(f"   总评审轮数: {review_summary['total_reviews']}")
    print(f"   最终得分: {review_summary['final_score']:.2f}/10")

    print(f"\n🔧 修正历史:")
    refinement_summary = result['refinement_summary']
    print(f"   总修正次数: {refinement_summary['total_refinements']}")
    if refinement_summary['issues_addressed']:
        print(f"   修正的问题: {', '.join(refinement_summary['issues_addressed'])}")

    print(f"\n🔎 查重结果:")
    verification = result['verification_summary']
    print(f"   检测到撞车: {'是' if verification['collision_detected'] else '否'}")
    print(f"   最高相似度: {verification['max_similarity']:.2f}")


# ===================== 示例 5: 导出 Markdown =====================
def demo_export_markdown():
    """示例5: 导出为 Markdown"""
    print("\n" + "=" * 80)
    print("📚 示例 5: 导出为 Markdown")
    print("=" * 80)

    # 检查是否有 log
    story_file = OUTPUT_DIR / "log"

    if not story_file.exists():
        print("\n⚠️  未找到 log")
        print("   请先运行: python scripts/idea2story_pipeline.py")
        return

    # 加载 Story
    with open(story_file, 'r', encoding='utf-8') as f:
        story = json.load(f)

    # 生成 Markdown
    md_content = f"""# {story['title']}

## Abstract

{story['abstract']}

## Problem Definition

{story['problem_definition']}

## Method Skeleton

{story['method_skeleton']}

## Innovation Claims

{chr(10).join([f"- {claim}" for claim in story['innovation_claims']])}

## Experiments Plan

{story['experiments_plan']}

---

*Generated by Idea2Story Pipeline*
"""

    # 保存
    md_file = OUTPUT_DIR / "final_story.md"
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write(md_content)

    print(f"\n✅ Markdown 已保存到: {md_file}")
    print("\n预览:")
    print("-" * 80)
    print(md_content[:500] + "...")
    print("-" * 80)


# ===================== 主函数 =====================
def main():
    """运行所有演示"""
    print("=" * 80)
    print("🎓 Idea2Story Pipeline 演示")
    print("=" * 80)

    print("\n选择演示:")
    print("  1. 基础使用（完整流程）")
    print("  2. 自定义配置")
    print("  3. 批量处理")
    print("  4. 查看中间结果")
    print("  5. 导出 Markdown")
    print("  0. 运行所有演示")

    choice = input("\n请输入选项 (0-5): ").strip()

    if choice == '1':
        demo_basic_usage()
    elif choice == '2':
        demo_custom_config()
    elif choice == '3':
        demo_batch_processing()
    elif choice == '4':
        demo_inspect_intermediate()
    elif choice == '5':
        demo_export_markdown()
    elif choice == '0':
        # 运行所有（跳过耗时的基础使用）
        demo_custom_config()
        demo_batch_processing()
        demo_inspect_intermediate()
        demo_export_markdown()
    else:
        print("\n⚠️  无效选项")

    print("\n" + "=" * 80)
    print("✅ 演示完成!")
    print("=" * 80)
    print("\n💡 提示:")
    print("  - 运行完整流程: python scripts/idea2story_pipeline.py")
    print("  - 查看文档: docs/QUICK_START_PIPELINE.md")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
