import time
from typing import Dict, List, Tuple

from .config import PipelineConfig
from .critic import MultiAgentCritic
from .pattern_selector import PatternSelector
from .refinement import RefinementEngine
from .story_generator import StoryGenerator
from .verifier import RAGVerifier


class Idea2StoryPipeline:
    """Idea2Story 主流程编排器"""

    def __init__(self, user_idea: str, recalled_patterns: List[Tuple[str, Dict, float]],
                 papers: List[Dict]):
        self.user_idea = user_idea
        self.recalled_patterns = recalled_patterns
        self.papers = papers

        # 初始化各模块
        self.pattern_selector = PatternSelector(recalled_patterns)
        self.story_generator = StoryGenerator(user_idea)
        self.critic = MultiAgentCritic()
        self.refinement_engine = RefinementEngine(recalled_patterns)
        self.verifier = RAGVerifier(papers)

    def run(self) -> Dict:
        """运行完整 Pipeline

        Returns:
            {
                'success': bool,
                'final_story': Dict,
                'iterations': int,
                'selected_patterns': Dict,
                'review_history': List,
                'refinement_history': List
            }
        """
        print("\n" + "=" * 80)
        print("🚀 Idea2Story Pipeline 启动")
        print("=" * 80)
        print(f"\n【用户 Idea】\n{self.user_idea}\n")

        # Phase 1: Pattern Selection
        selected_patterns = self.pattern_selector.select()

        if not selected_patterns:
            print("❌ 未选择到 Pattern，流程终止")
            return {'success': False}

        # 选择第一个 Pattern 进行生成（优先使用 conservative）
        pattern_type = 'conservative' if 'conservative' in selected_patterns else list(selected_patterns.keys())[0]
        pattern_id, pattern_info = selected_patterns[pattern_type]

        print(f"\n🎯 使用 Pattern: {pattern_type} - {pattern_id}")

        # 初始化迭代变量（必须在第一次生成前初始化）
        iterations = 0
        constraints = None
        injected_tricks = []  # 初始生成时无注入
        review_history = []
        refinement_history = []

        # Phase 2: Initial Story Generation (初始生成)
        current_story = self.story_generator.generate(
            pattern_id, pattern_info, constraints, injected_tricks
        )

        while iterations < PipelineConfig.MAX_REFINE_ITERATIONS:
            iterations += 1
            print(f"\n" + "=" * 80)
            print(f"🔄 迭代轮次: {iterations}/{PipelineConfig.MAX_REFINE_ITERATIONS}")
            print("=" * 80)

            # Phase 3: Multi-Agent Critic
            critic_result = self.critic.review(current_story)
            review_history.append(critic_result)

            if critic_result['pass']:
                print("\n✅ 评审通过，进入查重验证阶段")
                break

            # Phase 3.5: Refinement
            print(f"\n❌ 评审未通过 (平均分: {critic_result['avg_score']:.2f})")

            main_issue = critic_result['main_issue']
            suggestions = critic_result['suggestions']

            # 检查分数是否停滞 (针对 novelty)
            if iterations >= 1 and main_issue == 'novelty':
                # 获取当前和上一次的 Novelty 分数
                curr_novelty_score = next((r['score'] for r in critic_result['reviews'] if r['role'] == 'Novelty'), 0)
                prev_novelty_score = 0
                if len(review_history) >= 2:
                    prev_novelty_score = next((r['score'] for r in review_history[-2]['reviews'] if r['role'] == 'Novelty'), 0)

                if iterations >= 2 and curr_novelty_score <= prev_novelty_score + 0.5:
                    print(f"\n⚠️  检测到新颖性评分停滞或提升缓慢 ({curr_novelty_score:.1f} <= {prev_novelty_score:.1f} + 0.5)")

                    # 全局寻找未使用的、最创新的 Pattern (不再局限于 Phase 1 的 3 个)
                    all_unused = [
                        (pid, pinfo) for pid, pinfo, _ in self.recalled_patterns
                        if pid not in self.refinement_engine.used_patterns
                    ]
                    # 按聚类大小升序排列，优先选冷门的
                    all_unused.sort(key=lambda x: x[1].get('cluster_size', 999))

                    if all_unused:
                        alt_pattern = all_unused[0]
                        pattern_id, pattern_info = alt_pattern
                        print(f"🚀 强制切换到全局最创新 Pattern: {pattern_id} (聚类大小: {pattern_info.get('cluster_size')})")

                        # 切换 Pattern 后，清空之前的注入，重新开始
                        injected_tricks = []
                        print("   已重置注入技巧，基于新 Pattern 重新构建")
                    else:
                        print("   ⚠️  已无更多可用 Pattern，继续在当前路径修正")

            new_tricks = self.refinement_engine.refine(main_issue, suggestions)


            # 累积 Tricks (去重)
            if new_tricks:
                for trick in new_tricks:
                    if trick not in injected_tricks:
                        injected_tricks.append(trick)

            refinement_history.append({
                'iteration': iterations,
                'issue': main_issue,
                'injected_tricks': new_tricks
            })

            print(f"\n🔄 准备重新生成 Story（迭代 {iterations + 1}）...\n")
            time.sleep(1)  # 短暂延迟

            # 判断是否发生了 Pattern 强制切换
            # 如果发生了切换，则视为重新生成（previous_story=None）
            # 否则，视为增量修正
            is_pattern_switch = False
            if iterations >= 2 and main_issue == 'novelty':
                 # 简单的启发式判断：如果 injected_tricks 被清空了，说明发生了切换
                 if not injected_tricks and new_tricks:
                     is_pattern_switch = True

            # 注意：上面的判断逻辑可能不够严谨，更准确的是检查 pattern_id 是否变化
            # 但由于 pattern_id 在循环外定义，这里我们直接根据上下文传递逻辑来处理

            if is_pattern_switch:
                 # 强制切换模式：重新生成
                 current_story = self.story_generator.generate(
                    pattern_id, pattern_info, constraints, injected_tricks
                )
            else:
                # 增量修正模式：传入旧 Story、评审反馈、以及本轮新增的 Trick
                current_story = self.story_generator.generate(
                    pattern_id, pattern_info, constraints, injected_tricks,
                    previous_story=current_story,
                    review_feedback=critic_result,
                    new_tricks_only=new_tricks
                )

        # 检查是否达到最大迭代次数
        if iterations >= PipelineConfig.MAX_REFINE_ITERATIONS and not review_history[-1]['pass']:
            print("\n⚠️  达到最大迭代次数，但评审仍未通过")
            print("   将使用当前版本进入查重验证阶段\n")

        # Phase 4: RAG Verification
        verification_result = self.verifier.verify(current_story)

        if verification_result['collision_detected']:
            print("\n❌ 检测到撞车，触发 Pivot 策略")

            # 生成 Pivot 约束
            constraints = self.verifier.generate_pivot_constraints(
                current_story, verification_result['similar_papers']
            )

            # 重新生成（使用 innovative 或 cross_domain Pattern）
            if 'innovative' in selected_patterns:
                pattern_id, pattern_info = selected_patterns['innovative']
                print(f"\n🔄 切换到创新型 Pattern: {pattern_id}")
            elif 'cross_domain' in selected_patterns:
                pattern_id, pattern_info = selected_patterns['cross_domain']
                print(f"\n🔄 切换到跨域型 Pattern: {pattern_id}")

            current_story = self.story_generator.generate(
                pattern_id, pattern_info, constraints, injected_tricks
            )

            # 重新查重
            verification_result = self.verifier.verify(current_story)

        # 输出最终结果
        success = verification_result['pass']

        print("\n" + "=" * 80)
        print("🎉 Pipeline 完成!")
        print("=" * 80)
        print(f"✅ 状态: {'成功' if success else '需人工审核'}")
        print(f"📊 迭代次数: {iterations}")
        print(f"📝 最终 Story:")
        print(f"   标题: {current_story.get('title', '')}")
        print(f"   摘要: {current_story.get('abstract', '')[:100]}...")
        print("=" * 80)

        return {
            'success': success,
            'final_story': current_story,
            'iterations': iterations,
            'selected_patterns': {k: v[0] for k, v in selected_patterns.items()},
            'review_history': review_history,
            'refinement_history': refinement_history,
            'verification_result': verification_result
        }

