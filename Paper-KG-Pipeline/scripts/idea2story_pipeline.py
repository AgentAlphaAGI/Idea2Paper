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
import pickle
import sys
from collections import defaultdict

import numpy as np

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
        user_idea = "使用蒸馏技术做Transformer跨领域文本分类任务"

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

        # 运行召回（捕获输出以保持控制台整洁）
        print("\n🔍 运行召回系统...")
        print("-" * 80)

        # 直接导入召回逻辑
        from simple_recall_demo import (
            NODES_IDEA, NODES_PATTERN, NODES_DOMAIN, NODES_PAPER, GRAPH_FILE,
            compute_similarity, TOP_K_IDEAS, TOP_K_DOMAINS, TOP_K_PAPERS,
            FINAL_TOP_K, PATH1_WEIGHT, PATH2_WEIGHT, PATH3_WEIGHT
        )

        # 加载数据
        with open(NODES_IDEA, 'r', encoding='utf-8') as f:
            ideas = json.load(f)
        with open(NODES_PATTERN, 'r', encoding='utf-8') as f:
            patterns_data = json.load(f)
        with open(NODES_DOMAIN, 'r', encoding='utf-8') as f:
            domains = json.load(f)
        with open(NODES_PAPER, 'r', encoding='utf-8') as f:
            papers_data = json.load(f)
        with open(GRAPH_FILE, 'rb') as f:
            G = pickle.load(f)

        # 【关键修复】加载完整的 patterns_structured.json 以获取 skeleton_examples
        patterns_structured_file = OUTPUT_DIR / "patterns_structured.json"
        with open(patterns_structured_file, 'r', encoding='utf-8') as f:
            patterns_structured = json.load(f)

        # 构建 pattern_id -> structured_data 的映射
        structured_map = {}
        for p in patterns_structured:
            pattern_id = f"pattern_{p.get('pattern_id')}"
            structured_map[pattern_id] = p

        # 构建索引并合并完整的 skeleton_examples
        idea_map = {i['idea_id']: i for i in ideas}
        pattern_map = {}
        for p in patterns_data:
            pattern_id = p['pattern_id']
            # 合并 nodes_pattern 和 patterns_structured 的数据
            merged_pattern = dict(p)  # 复制基础数据
            if pattern_id in structured_map:
                # 补充完整的 skeleton_examples 和 common_tricks
                merged_pattern['skeleton_examples'] = structured_map[pattern_id].get('skeleton_examples', [])
                merged_pattern['common_tricks'] = structured_map[pattern_id].get('common_tricks', [])
            pattern_map[pattern_id] = merged_pattern

        domain_map = {d['domain_id']: d for d in domains}
        paper_map = {p['paper_id']: p for p in papers_data}

        # 路径1
        path1_scores = defaultdict(float)
        similarities = [(idea['idea_id'], compute_similarity(user_idea, idea['description']))
                       for idea in ideas if compute_similarity(user_idea, idea['description']) > 0]
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_ideas = similarities[:TOP_K_IDEAS]

        for idea_id, similarity in top_ideas:
            idea = idea_map[idea_id]
            pattern_ids = idea.get('pattern_ids', [])
            for pid in pattern_ids:
                path1_scores[pid] += similarity

        # 路径2
        path2_scores = defaultdict(float)
        top_idea = idea_map[top_ideas[0][0]] if top_ideas else None
        domain_scores = []

        if top_idea and G.has_node(top_idea['idea_id']):
            for successor in G.successors(top_idea['idea_id']):
                edge_data = G[top_idea['idea_id']][successor]
                if edge_data.get('relation') == 'belongs_to':
                    domain_id = successor
                    weight = edge_data.get('weight', 0.5)
                    domain_scores.append((domain_id, weight))

        domain_scores.sort(key=lambda x: x[1], reverse=True)
        top_domains = domain_scores[:TOP_K_DOMAINS]

        for domain_id, domain_weight in top_domains:
            for predecessor in G.predecessors(domain_id):
                edge_data = G[predecessor][domain_id]
                if edge_data.get('relation') == 'works_well_in':
                    pattern_id = predecessor
                    effectiveness = edge_data.get('effectiveness', 0.0)
                    confidence = edge_data.get('confidence', 0.0)
                    path2_scores[pattern_id] += domain_weight * max(effectiveness, 0.1) * confidence

        # 路径3
        path3_scores = defaultdict(float)
        similarities = []
        for paper in papers_data:
            paper_idea = paper.get('idea', {}).get('core_idea', '') or paper.get('abstract', '')[:100]
            if not paper_idea:
                continue

            sim = compute_similarity(user_idea, paper_idea)
            if sim > 0.1 and G.has_node(paper['paper_id']):
                reviews = paper.get('reviews', [])
                if reviews:
                    scores = [r.get('rating', 5) for r in reviews]
                    avg_score = np.mean(scores)
                    quality = (avg_score - 1) / 9
                else:
                    quality = 0.5

                combined = sim * quality
                similarities.append((paper['paper_id'], sim, quality, combined))

        similarities.sort(key=lambda x: x[3], reverse=True)
        top_papers = similarities[:TOP_K_PAPERS]

        for paper_id, similarity, quality, combined_weight in top_papers:
            if not G.has_node(paper_id):
                continue
            for successor in G.successors(paper_id):
                edge_data = G[paper_id][successor]
                if edge_data.get('relation') == 'uses_pattern':
                    pattern_id = successor
                    pattern_quality = edge_data.get('quality', 0.5)
                    path3_scores[pattern_id] += combined_weight * pattern_quality

        # 融合
        all_patterns = set(path1_scores.keys()) | set(path2_scores.keys()) | set(path3_scores.keys())
        final_scores = {}
        for pattern_id in all_patterns:
            score1 = path1_scores.get(pattern_id, 0.0) * PATH1_WEIGHT
            score2 = path2_scores.get(pattern_id, 0.0) * PATH2_WEIGHT
            score3 = path3_scores.get(pattern_id, 0.0) * PATH3_WEIGHT
            final_scores[pattern_id] = score1 + score2 + score3

        ranked = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        top_k = ranked[:FINAL_TOP_K]

        # 构建召回结果
        recalled_patterns = [
            (pattern_id, pattern_map.get(pattern_id, {}), score)
            for pattern_id, score in top_k
        ]

        # 恢复 argv
        sys.argv = original_argv

        print("-" * 80)
        print(f"✅ 召回完成: Top-{len(recalled_patterns)} Patterns\n")

        # 运行 Pipeline
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

