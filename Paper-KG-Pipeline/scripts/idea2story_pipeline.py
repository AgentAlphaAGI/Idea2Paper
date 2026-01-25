"""
Idea2Story Pipeline - 从用户 Idea 到可发表的 Paper Story

实现流程:
  Phase 1: Pattern Selection (策略选择)
  Phase 2: Story Generation (结构化生成)
  Phase 3: Multi-Agent Critic & Refine (评审与修正)
  Phase 4: RAG Verification & Pivot (查重与规避)

使用方法:
  python scripts/idea2story_pipeline.py "你的Idea描述"
"""

import json
import sys

# 导入 Pipeline 模块
try:
    from pipeline import Idea2StoryPipeline, OUTPUT_DIR
except ImportError:
    # 如果直接运行脚本，尝试添加当前目录到 path
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from pipeline import Idea2StoryPipeline, OUTPUT_DIR

# ===================== 主函数 =====================
def main():
    """主函数"""
    # 获取用户输入
    if len(sys.argv) > 1:
        user_idea = " ".join(sys.argv[1:])
    else:
        user_idea = "LLM-Assisted Domain Data Extraction and Cleaning"

    # 加载召回结果（调用 simple_recall_demo 的结果）
    print("📂 加载数据...")

    try:
        # 加载节点数据
        with open(OUTPUT_DIR / "nodes_pattern.json", 'r', encoding='utf-8') as f:
            patterns = json.load(f)
        with open(OUTPUT_DIR / "nodes_paper.json", 'r', encoding='utf-8') as f:
            papers = json.load(f)

        print(f"  ✓ 加载 {len(patterns)} 个 Pattern")
        print(f"  ✓ 加载 {len(papers)} 个 Paper")

        # 运行召回（复用 simple_recall_demo 的逻辑）
        # 注意：这里为了复用逻辑，直接导入了 simple_recall_demo
        # 在生产环境中，建议将召回逻辑封装为独立的类

        # 临时保存原始 argv
        original_argv = sys.argv.copy()
        sys.argv = ['simple_recall_demo.py', user_idea]

        # 运行召回（使用 RecallSystem 类，支持两阶段优化）
        print("\n🔍 运行召回系统...")
        print("-" * 80)

        # 【优化】直接使用 RecallSystem 类（支持两阶段召回，大幅提速）
        from recall_system import RecallSystem

        print("  初始化召回系统...")
        recall_system = RecallSystem()

        print("\n  执行三路召回（优化版，支持两阶段加速）...")
        recall_results = recall_system.recall(user_idea, verbose=True)

        # 【关键修复】加载完整的 patterns_structured.json 以合并数据
        patterns_structured_file = OUTPUT_DIR / "patterns_structured.json"
        if patterns_structured_file.exists():
            with open(patterns_structured_file, 'r', encoding='utf-8') as f:
                patterns_structured = json.load(f)

            # 构建 pattern_id -> structured_data 的映射
            structured_map = {}
            for p in patterns_structured:
                pattern_id = f"pattern_{p.get('pattern_id')}"
                structured_map[pattern_id] = p

            # 合并 skeleton_examples 和 common_tricks 到召回结果
            merged_results = []
            for pattern_id, pattern_info, score in recall_results:
                merged_pattern = dict(pattern_info)
                if pattern_id in structured_map:
                    merged_pattern['skeleton_examples'] = structured_map[pattern_id].get('skeleton_examples', [])
                    merged_pattern['common_tricks'] = structured_map[pattern_id].get('common_tricks', [])
                merged_results.append((pattern_id, merged_pattern, score))

            recalled_patterns = merged_results
        else:
            # 如果没有 patterns_structured.json，直接使用召回结果
            recalled_patterns = recall_results

        # 加载 papers 数据 (Pipeline 需要用于 RAG 查重)
        print("\n  加载 Papers 数据用于查重...")
        with open(OUTPUT_DIR / "nodes_paper.json", 'r', encoding='utf-8') as f:
            papers = json.load(f)

        # 恢复 argv
        sys.argv = original_argv

        print("-" * 80)
        print(f"✅ 召回完成: Top-{len(recalled_patterns)} Patterns\n")

        # 运行 Pipeline（传递 user_idea 用于 Pattern 智能分类）
        pipeline = Idea2StoryPipeline(user_idea, recalled_patterns, papers)
        result = pipeline.run()

        # 保存结果
        output_file = OUTPUT_DIR / "final_story.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result['final_story'], f, ensure_ascii=False, indent=2)

        print(f"\n💾 最终 Story 已保存到: {output_file}")

        # 保存完整结果
        full_result_file = OUTPUT_DIR / "pipeline_result.json"
        with open(full_result_file, 'w', encoding='utf-8') as f:
            json.dump({
                'user_idea': user_idea,
                'success': result['success'],
                'iterations': result['iterations'],
                'selected_patterns': result['selected_patterns'],
                'final_story': result['final_story'],
                'review_history': result['review_history'],
                'review_summary': {
                    'total_reviews': len(result['review_history']),
                    'final_score': result['review_history'][-1]['avg_score'] if result['review_history'] else 0
                },
                'refinement_summary': {
                    'total_refinements': len(result['refinement_history']),
                    'issues_addressed': [r['issue'] for r in result['refinement_history']]
                },
                'verification_summary': {
                    'collision_detected': result['verification_result']['collision_detected'],
                    'max_similarity': result['verification_result']['max_similarity']
                }
            }, f, ensure_ascii=False, indent=2)

        print(f"💾 完整结果已保存到: {full_result_file}")

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

