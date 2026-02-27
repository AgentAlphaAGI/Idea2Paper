import time
from typing import Dict, List, Tuple

from idea2paper.config import (
    NOVELTY_ENABLE,
    NOVELTY_ACTION,
    NOVELTY_MAX_PIVOTS,
    NOVELTY_REQUIRE_EMBEDDING,
    OUTPUT_DIR,
)
from idea2paper.config import PipelineConfig
from idea2paper.infra.run_context import get_logger
from idea2paper.novelty.novelty_checker import NoveltyChecker
from idea2paper.pipeline.pattern_selector import PatternSelector
from idea2paper.pipeline.refinement import RefinementEngine
from idea2paper.pipeline.story_generator import StoryGenerator
from idea2paper.pipeline.story_reflector import StoryReflector
from idea2paper.pipeline.verification_adapter import verification_from_novelty_report
from idea2paper.pipeline.verifier import RAGVerifier
from idea2paper.review.critic import MultiAgentCritic
from idea2paper.review.review_index import ReviewIndex


class Idea2StoryPipeline:
    """Idea2Story 主流程编排器"""

    def __init__(self, user_idea: str, recalled_patterns: List[Tuple[str, Dict, float]],
                 papers: List[Dict], run_id: str | None = None,
                 idea_brief: Dict | None = None):
        self.user_idea = user_idea
        self.raw_idea = user_idea
        self.idea_brief = idea_brief
        self.recalled_patterns = recalled_patterns
        self.papers = papers
        self.run_id = run_id

        # 初始化各模块（传递 user_idea 给 PatternSelector 用于智能分类）
        self.pattern_selector = PatternSelector(recalled_patterns, user_idea, idea_brief=idea_brief)
        self.story_generator = StoryGenerator(user_idea, idea_brief=idea_brief)
        self.story_reflector = StoryReflector()  # 新增：故事反思器
        self.review_index = ReviewIndex(papers)
        self.critic = MultiAgentCritic(review_index=self.review_index)
        # RefinementEngine 需要在 Pattern Selection 后初始化，以获取分类结果
        self.refinement_engine = None  # 延迟初始化
        self.verifier = RAGVerifier(papers)
        self.novelty_checker = NoveltyChecker(
            papers=self.papers,
            nodes_paper_path=OUTPUT_DIR / "nodes_paper.json",
            logger=get_logger()
        )
        self.pattern_info_map = {pid: info for pid, info, _ in recalled_patterns}

    def _build_critic_context(self, fallback_pattern_id: str, fallback_pattern_info: Dict) -> Dict:
        current_pid = fallback_pattern_id
        if self.refinement_engine and getattr(self.refinement_engine, 'current_pattern_id', None):
            current_pid = self.refinement_engine.current_pattern_id
        pattern_info = self.pattern_info_map.get(current_pid, fallback_pattern_info)
        return {
            'pattern_id': current_pid,
            'pattern_info': pattern_info,
            'idea_brief': self.idea_brief
        }

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
        logger = get_logger()
        print("\n" + "=" * 80)
        print("🚀 Idea2Story Pipeline 启动")
        print("=" * 80)
        print(f"\n【用户 Idea】\n{self.raw_idea}\n")

        # Phase 1: Pattern Selection (多维度评分与排序)
        ranked_patterns = self.pattern_selector.select()

        if not ranked_patterns or all(len(v) == 0 for v in ranked_patterns.values()):
            print("❌ 未选择到 Pattern，流程终止")
            return {'success': False}

        # 选择第一个 Pattern 进行生成（优先使用 stability 维度的第一个 KG pattern）
        # Agentic Search pattern（source='agentic_search'，size=1）质量未经社区验证，
        # 不宜作为初始 Story 的基座 pattern。应优先选 KG pattern 保证稳定性；
        # Agentic pattern 的新颖性价值留给后续 Refinement 阶段通过 idea fusion 注入。
        def _is_agentic(pid: str, pinfo: Dict) -> bool:
            return (
                str(pinfo.get("source", "")).lower() == "agentic_search"
                or str(pid).startswith("p4_")
            )

        pattern_id = None
        pattern_info = None
        metadata = None
        dimension_type = None

        if ranked_patterns.get('stability'):
            # 优先选 stability 维度里第一个 KG pattern
            for pid, pinfo, meta in ranked_patterns['stability']:
                if not _is_agentic(pid, pinfo):
                    pattern_id, pattern_info, metadata = pid, pinfo, meta
                    dimension_type = 'stability'
                    break
            # 若 stability 维度全是 agentic pattern，退回到第一个（不应出现但做保底）
            if pattern_id is None:
                pattern_id, pattern_info, metadata = ranked_patterns['stability'][0]
                dimension_type = 'stability'

        if pattern_id is None and ranked_patterns.get('novelty'):
            pattern_id, pattern_info, metadata = ranked_patterns['novelty'][0]
            dimension_type = 'novelty'

        if pattern_id is None:
            first_dim = list(ranked_patterns.keys())[0]
            pattern_id, pattern_info, metadata = ranked_patterns[first_dim][0]
            dimension_type = first_dim

        is_agentic_initial = _is_agentic(pattern_id, pattern_info)
        print(f"\n🎯 使用 Pattern: {dimension_type} 维度 - {pattern_id}"
              + (" [KG]" if not is_agentic_initial else " [Agentic-fallback]"))
        if logger:
            logger.log_event("pattern_selected", {
                "pattern_id": pattern_id,
                "dimension": dimension_type
            })

        # 初始化 RefinementEngine（传入分类结果和 user_idea 以支持 idea fusion）
        self.refinement_engine = RefinementEngine(self.recalled_patterns, ranked_patterns, self.user_idea)

        # 初始化迭代变量（必须在第一次生成前初始化）
        iterations = 0
        constraints = None
        injected_tricks = []  # 初始生成时无注入
        review_history = []
        refinement_history = []

        # 【新增】分数退化检测和回滚机制
        last_story_before_refinement = None  # 用于回滚
        last_issue_type = None  # 记录上一轮的 issue 类型
        pattern_failure_map = {}  # 记录 pattern 对各类 issue 的失败情况

        # 【新增】新颖性模式相关
        novelty_mode_active = False  # 是否进入新颖性模式
        novelty_pattern_iterations = 0  # 新颖性模式的迭代次数
        novelty_pattern_results = []  # 存储所有新颖性 pattern 的生成结果
        best_novelty_result = None  # 最佳新颖性结果
        novelty_mode_base_iteration = None  # 记录触发新颖性模式时的迭代次数

        # 【新增】全局最佳版本追踪
        global_best_story = None  # 整个迭代过程中得分最高的Story
        global_best_score = 0.0  # 对应的最高分数
        global_best_critic_result = None  # 对应的Critic结果
        global_best_iteration = 0  # 对应的迭代轮次

        # Phase 2: Initial Story Generation (初始生成)
        current_story = self.story_generator.generate(
            pattern_id, pattern_info, constraints, injected_tricks
        )

        while iterations < PipelineConfig.MAX_REFINE_ITERATIONS or novelty_mode_active:
            iterations += 1
            print(f"\n" + "=" * 80)
            if novelty_mode_active:
                print(f"🔄 迭代轮次: {novelty_mode_base_iteration} (新颖性模式 - 遍历Pattern #{novelty_pattern_iterations + 1})")
            else:
                print(f"🔄 迭代轮次: {iterations}/{PipelineConfig.MAX_REFINE_ITERATIONS}")
            print("=" * 80)
            if logger:
                logger.log_event("iteration", {
                    "iteration": iterations,
                    "novelty_mode": novelty_mode_active,
                    "novelty_pattern_iterations": novelty_pattern_iterations
                })

            # Phase 3: Multi-Agent Critic
            critic_context = self._build_critic_context(pattern_id, pattern_info)
            critic_result = self.critic.review(current_story, context=critic_context)
            if logger:
                logger.log_event("critic_result", {
                    "avg_score": critic_result.get("avg_score"),
                    "pass": critic_result.get("pass"),
                    "main_issue": critic_result.get("main_issue")
                })

            # 【核心】分数退化检测 - 检查是否应该回滚
            if len(review_history) > 0 and last_issue_type:
                issue_to_role = {
                    'novelty': 'Novelty',
                    'stability': 'Methodology',
                    'domain_distance': 'Storyteller'
                }
                role_name = issue_to_role.get(last_issue_type, last_issue_type)
                curr_issue_score = next((r['score'] for r in critic_result['reviews'] if r['role'] == role_name), 0)
                prev_issue_score = next((r['score'] for r in review_history[-1]['reviews'] if r['role'] == role_name), 0)

                # 如果该维度的分数下降，则触发回滚
                if curr_issue_score < prev_issue_score - 0.1:  # 允许 0.1 的浮动误差
                    print(f"\n" + "=" * 80)
                    print(f"⚠️  【ROLLBACK TRIGGERED】{last_issue_type} 分数下降")
                    print(f"   前一轮: {prev_issue_score:.1f} → 本轮: {curr_issue_score:.1f}")
                    print(f"   最后注入的 Pattern 未能改进，进行完整回滚...")
                    print("=" * 80)
                    if logger:
                        logger.log_event("rollback_triggered", {
                            "issue_type": last_issue_type,
                            "prev_score": prev_issue_score,
                            "curr_score": curr_issue_score
                        })

                    # Step 1: 回滚到前一个版本
                    if last_story_before_refinement:
                        current_story = last_story_before_refinement
                        print(f"   ✅ Step 1: 已回滚 Story 到前一个版本")

                    # Step 2: 标记该 pattern 在该 issue 上失败
                    last_used_pattern = refinement_history[-1].get('pattern_id') if refinement_history else None
                    if last_used_pattern:
                        if last_used_pattern not in pattern_failure_map:
                            pattern_failure_map[last_used_pattern] = set()
                        pattern_failure_map[last_used_pattern].add(last_issue_type)
                        print(f"   ✅ Step 2: 标记 {last_used_pattern} 对 {last_issue_type} 失败")

                    # Step 3: 移除上一轮的 refinement 记录和 tricks，准备重新尝试
                    if refinement_history:
                        removed_refinement = refinement_history.pop()
                        print(f"   ✅ Step 3: 移除 iteration {removed_refinement['iteration']} 的修正记录")

                        # 恢复 injected_tricks（移除本轮注入的技巧）
                        last_tricks = removed_refinement.get('injected_tricks', [])
                        for trick in last_tricks:
                            if trick in injected_tricks:
                                injected_tricks.remove(trick)
                        print(f"   ✅ Step 4: 恢复 injected_tricks（移除 {len(last_tricks)} 个）")

                    # Step 5: 通知 refinement_engine 当前 pattern 失败，选择下一个
                    self.refinement_engine.mark_pattern_failed(last_used_pattern, last_issue_type)
                    print(f"   ✅ Step 5: 通知 RefinementEngine 该 Pattern 失败")

                    # Step 6: 继续到下一轮迭代而不更新 review_history
                    print(f"\n   准备下一轮迭代，自动选择新的 Pattern...\n")
                    continue

            # 【说明】在新颖性模式下，Critic评审已在story生成后立即执行
            # 这里只处理非新颖性模式的情况
            if not novelty_mode_active:
                review_history.append(critic_result)

            # 【新增】更新全局最佳版本
            current_avg_score = critic_result['avg_score']
            if current_avg_score > global_best_score:
                global_best_story = dict(current_story) if current_story else None
                global_best_score = current_avg_score
                global_best_critic_result = dict(critic_result)
                global_best_iteration = iterations if not novelty_mode_active else novelty_mode_base_iteration
                print(f"\n🏆 更新全局最佳版本: 得分 {global_best_score:.2f} (迭代 {global_best_iteration})")

            if critic_result['pass'] and not novelty_mode_active:
                print("\n✅ 评审通过，进入查重验证阶段")
                break

            # Phase 3.5: Refinement
            print(f"\n❌ 评审未通过 (平均分: {critic_result['avg_score']:.2f})")

            main_issue = critic_result['main_issue']
            suggestions = critic_result['suggestions']

            # 【新增】保存当前 story 作为回滚点
            last_story_before_refinement = dict(current_story) if current_story else None

            # 【新增】检查分数是否停滞 (针对 novelty) - 激活新颖性模式
            # 只在首次检测到时激活，避免重复触发
            if iterations >= 1 and main_issue == 'novelty' and not novelty_mode_active and novelty_mode_base_iteration is None:
                # 获取当前和上一次的 Novelty 分数
                curr_novelty_score = next((r['score'] for r in critic_result['reviews'] if r['role'] == 'Novelty'), 0)
                prev_novelty_score = 0
                if len(review_history) >= 2:
                    prev_novelty_score = next((r['score'] for r in review_history[-2]['reviews'] if r['role'] == 'Novelty'), 0)

                if iterations >= 2 and curr_novelty_score <= prev_novelty_score + 0.5:
                    print(f"\n⚠️  检测到新颖性评分停滞或提升缓慢 ({curr_novelty_score:.1f} <= {prev_novelty_score:.1f} + 0.5)")
                    print(f"🎯 激活【新颖性模式】- 遍历所有新颖性 Pattern（可超过最大迭代次数）\n")
                    if logger:
                        logger.log_event("novelty_mode_activated", {
                            "iteration": iterations,
                            "curr_novelty_score": curr_novelty_score,
                            "prev_novelty_score": prev_novelty_score
                        })

                    # 激活新颖性模式
                    novelty_mode_active = True
                    novelty_pattern_iterations = 0
                    novelty_pattern_results = []
                    novelty_mode_base_iteration = iterations  # 记录基准迭代次数
                    # 记录当前Story作为回滚基准
                    last_story_before_refinement = dict(current_story) if current_story else None

            # 【核心创新】使用 Idea Fusion 进行修正
            # 在新颖性模式下，强制遍历下一个Pattern
            force_next = novelty_mode_active and main_issue == 'novelty'
            new_tricks, fused_idea = self.refinement_engine.refine_with_idea_fusion(
                main_issue, suggestions, current_story, force_next_pattern=force_next
            )

            # 【新增】检查是否没有更多Pattern可用（在新颖性模式下）
            if novelty_mode_active and main_issue == 'novelty' and not fused_idea:
                print(f"\n   ⚠️  没有更多新颖性Pattern可用")
                print("   退出新颖性模式，准备启用兜底策略")
                novelty_mode_active = False
                # 跳出当前循环，进入兜底策略
                break

            # 累积 Tricks (去重)
            if new_tricks:
                for trick in new_tricks:
                    if trick not in injected_tricks:
                        injected_tricks.append(trick)

            # 获取本轮使用的 pattern_id（从 refinement engine 获取）
            current_pattern_id = self.refinement_engine.current_pattern_id if hasattr(self.refinement_engine, 'current_pattern_id') else None

            refinement_history.append({
                'iteration': novelty_mode_base_iteration if novelty_mode_active else iterations,
                'issue': main_issue,
                'pattern_id': current_pattern_id,  # 保存使用的 pattern_id
                'injected_tricks': new_tricks,
                'fused_idea': fused_idea  # 保存融合后的 idea
            })

            # 【新增】记录本轮的 issue 类型，用于下一轮检测
            last_issue_type = main_issue

            print(f"\n🔄 准备重新生成 Story（迭代 {novelty_mode_base_iteration if novelty_mode_active else iterations + 1}）...\n")
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
                 new_story = self.story_generator.generate(
                    pattern_id, pattern_info, constraints, injected_tricks
                )
            else:
                # 增量修正模式：传入旧 Story、评审反馈、新增 Trick，以及融合后的 idea
                new_story = self.story_generator.generate(
                    pattern_id, pattern_info, constraints, injected_tricks,
                    previous_story=current_story,
                    review_feedback=critic_result,
                    new_tricks_only=new_tricks,
                    fused_idea=fused_idea  # 传入融合后的概念级创新 idea
                )

            # 【新增】生成后反思：评估融合质量（仅在融合发生时）
            if fused_idea and new_story:
                print("\n" + "=" * 80)
                print("🔍 Phase 3.6: Story Post-Generation Reflection (生成后质量评估)")
                print("=" * 80)

                reflection_result = self.story_reflector.reflect_on_fusion(
                    old_story=last_story_before_refinement or current_story,
                    new_pattern=pattern_info,
                    fused_idea=fused_idea,
                    critic_feedback=critic_result,
                    user_idea=self.user_idea
                )

                fusion_quality = reflection_result.get('fusion_quality_score', 0)
                ready_for_generation = reflection_result.get('ready_for_generation', True)
                print(f"\n   📊 融合质量评分: {fusion_quality:.2f}/1.0")
                print(f"   🎯 准备生成: {'是' if ready_for_generation else '否'}")

                # 打印融合质量诊断信息
                if fusion_quality < 0.65:
                    print(f"   ⚠️  融合质量不足 (< 0.65)")
                    print(f"   融合分析: {reflection_result.get('fusion_insights', '')[:200]}...")
                    print(f"   连贯性问题: {reflection_result.get('coherence_analysis', '')[:150]}...")
                else:
                    print(f"   ✅ 融合质量良好")
                    print(f"   融合优势: {reflection_result.get('fusion_insights', '')[:150]}...")

                # 【关键修正】无论融合质量如何，都应该根据Reflection建议生成Story终稿
                # 这是新颖性Pattern注入的核心步骤：初稿 → Reflection → 终稿
                print(f"\n🔄 Step 2: 根据Reflection建议生成Story终稿...")

                # 提取Reflection建议
                fusion_suggestions = reflection_result.get('fusion_suggestions', {})

                # 将Reflection建议注入到Story生成的约束中
                enhanced_constraints = dict(constraints) if constraints is not None else {}
                enhanced_constraints['reflection_guidance'] = fusion_suggestions

                # 重新生成Story（终稿），传入Reflection建议
                new_story = self.story_generator.generate(
                    pattern_id, pattern_info, enhanced_constraints, injected_tricks,
                    previous_story=new_story,  # 基于初稿进行改进
                    review_feedback=critic_result,
                    fused_idea=fused_idea,
                    reflection_guidance=fusion_suggestions  # 传入Reflection建议
                )

                print(f"   ✅ Story终稿已根据Reflection建议生成")

                # 【关键判断】如果融合质量极低（< 0.5），在新颖性模式下可以选择跳过Critic直接尝试下一个Pattern
                # 但这应该是可选的优化策略，不应阻止终稿生成
                if fusion_quality < 0.5 and novelty_mode_active and current_pattern_id:
                    print(f"\n   ⚠️  融合质量极低 (< 0.5)，可能不适合此Pattern")
                    print(f"   💡 提示: 将继续Critic评审，但如果失败可快速切换到下一个Pattern")

            # 接受新生成的 Story
            current_story = new_story

            # 【新增】在新颖性模式下，生成完成后立即进行Critic评审
            if novelty_mode_active and main_issue == 'novelty':
                novelty_pattern_iterations += 1
                print(f"\n" + "=" * 80)
                print(f"🔍 Phase 3: Multi-Agent Critic (评审Pattern #{novelty_pattern_iterations})")
                print("=" * 80)

                # 立即评审新生成的Story
                critic_context = self._build_critic_context(pattern_id, pattern_info)
                new_critic_result = self.critic.review(current_story, context=critic_context)
                if logger:
                    logger.log_event("critic_result", {
                        "avg_score": new_critic_result.get("avg_score"),
                        "pass": new_critic_result.get("pass"),
                        "main_issue": new_critic_result.get("main_issue"),
                        "novelty_mode": True
                    })

                # 记录本次尝试的结果
                novelty_pattern_results.append({
                    'iteration': novelty_mode_base_iteration,
                    'pattern_id': current_pattern_id,
                    'avg_score': new_critic_result['avg_score'],
                    'novelty_score': next((r['score'] for r in new_critic_result['reviews'] if r['role'] == 'Novelty'), 0),
                    'story': dict(current_story)
                })

                print(f"\n   📊 新颖性Pattern尝试 #{novelty_pattern_iterations}:")
                print(f"      Pattern: {current_pattern_id}")
                print(f"      平均分: {new_critic_result['avg_score']:.2f}/10")
                print(f"      新颖度: {novelty_pattern_results[-1]['novelty_score']:.1f}/10")

                # 如果通过评审，退出新颖性模式
                if new_critic_result['pass']:
                    print(f"\n   ✅ 评审通过！找到合适的Pattern")
                    review_history.append(new_critic_result)
                    novelty_mode_active = False
                    break

                # 如果不通过，但分数有提升，也标记为成功
                print(f"\n   ❌ 评审未通过 (平均分: {new_critic_result['avg_score']:.2f})")

                # 检查是否达到新颖性模式的最大尝试次数
                if novelty_pattern_iterations >= PipelineConfig.NOVELTY_MODE_MAX_PATTERNS:
                    print(f"\n   ⚠️  已达到新颖性模式最大尝试次数 ({PipelineConfig.NOVELTY_MODE_MAX_PATTERNS})")
                    print("   退出新颖性模式，准备启用兜底策略")
                    novelty_mode_active = False
                    # 继续到外层循环进行兜底处理
                else:
                    # 继续尝试下一个Pattern
                    print(f"   🔄 继续尝试下一个新颖性Pattern...")
                    # 不要break，继续循环

        # 【新增】新颖性模式下的兜底策略
        if novelty_pattern_results and not review_history[-1]['pass']:
            print("\n" + "=" * 80)
            print("🎯 新颖性模式兜底策略")
            print("=" * 80)
            print(f"\n⚠️  在新颖性模式中尝试了 {novelty_pattern_iterations} 个 Pattern")
            print(f"📊 所有尝试的结果:")

            # 从新颖性模式的所有结果中找到最高分的
            for idx, result in enumerate(novelty_pattern_results):
                print(f"   {idx + 1}. {result['pattern_id']}: 平均分={result['avg_score']:.2f}, 新颖度={result['novelty_score']:.1f}")

            best_result = max(novelty_pattern_results, key=lambda x: x['avg_score'])
            best_novelty_result = best_result
            current_story = best_result['story']

            print(f"\n   ✅ 选择最高分结果: 平均分={best_result['avg_score']:.2f}/10")
            print(f"   📝 Pattern: {best_result['pattern_id']}")
            print(f"   📝 使用该版本作为最终输出")

        # 检查是否达到最大迭代次数
        if not novelty_mode_active and iterations >= PipelineConfig.MAX_REFINE_ITERATIONS and not review_history[-1]['pass']:
            print("\n⚠️  达到最大迭代次数，但评审仍未通过")
            print("   将使用当前版本进入查重验证阶段\n")

        # 【新增】最终版本选择逻辑：通过版本 OR 最佳版本
        final_story = current_story  # 默认使用当前版本
        final_is_passed = review_history[-1]['pass'] if review_history else False

        if not final_is_passed and global_best_story is not None:
            # 如果当前版本未通过，但有全局最佳版本，使用最佳版本
            print("\n" + "=" * 80)
            print("🎯 最终版本选择逻辑")
            print("=" * 80)
            print(f"📊 当前版本: 平均分={critic_result['avg_score']:.2f}, 状态={'通过' if final_is_passed else '未通过'}")
            print(f"🏆 全局最佳版本: 平均分={global_best_score:.2f} (迭代 {global_best_iteration})")

            if global_best_score > critic_result['avg_score']:
                print(f"\n✅ 使用全局最佳版本作为最终输出（得分更高）")
                final_story = global_best_story
            else:
                print(f"\n✅ 使用当前版本作为最终输出（得分相同或更高）")
            print("=" * 80)

        # 本地查新（Novelty Check）+ Pivot
        novelty_report = None
        novelty_action = NOVELTY_ACTION
        pivot_attempts = 0
        if NOVELTY_ENABLE:
            run_id = self.run_id or "run_unknown"
            novelty_report = self.novelty_checker.check(final_story, run_id, self.user_idea)
            if logger:
                logger.log_event("novelty_check_done", {
                    "risk_level": novelty_report.get("risk_level"),
                    "max_similarity": novelty_report.get("max_similarity"),
                    "embedding_available": novelty_report.get("embedding_available"),
                    "report_path": novelty_report.get("report_path")
                })

            if not novelty_report.get("embedding_available", False):
                if NOVELTY_REQUIRE_EMBEDDING:
                    raise RuntimeError("Novelty check requires embeddings, but embedding is unavailable")
                # embedding 不可用时默认不触发 pivot
                if novelty_action == "pivot":
                    novelty_action = "report_only"

            while (
                novelty_report.get("risk_level") == "high"
                and novelty_action == "pivot"
                and pivot_attempts < NOVELTY_MAX_PIVOTS
            ):
                pivot_attempts += 1
                top_title = ""
                if novelty_report.get("candidates"):
                    top_title = novelty_report["candidates"][0].get("title", "")
                if logger:
                    logger.log_event("novelty_pivot_triggered", {
                        "attempt": pivot_attempts,
                        "top_title": top_title,
                        "max_similarity": novelty_report.get("max_similarity")
                    })

                # 生成 Pivot 约束（复用 verifier 的策略）
                if top_title:
                    constraints = self.verifier.generate_pivot_constraints(final_story, [{"title": top_title}])
                else:
                    constraints = [
                        "避免与已有工作使用相同核心技术组合",
                        "将应用场景迁移到新领域",
                        "增加额外约束条件（如无监督、少样本等）"
                    ]

                # 重新生成（使用 novelty 或 domain_distance 维度的 Pattern）
                if ranked_patterns.get('novelty') and len(ranked_patterns['novelty']) > 0:
                    pattern_id, pattern_info, metadata = ranked_patterns['novelty'][0]
                    print(f"\n🔄 [Novelty Pivot] 切换到新颖度维度 Pattern: {pattern_id}")
                elif ranked_patterns.get('domain_distance') and len(ranked_patterns['domain_distance']) > 0:
                    pattern_id, pattern_info, metadata = ranked_patterns['domain_distance'][0]
                    print(f"\n🔄 [Novelty Pivot] 切换到领域距离维度 Pattern: {pattern_id}")
                else:
                    # fallback 使用当前 pattern_info
                    pattern_id, pattern_info = pattern_id, pattern_info

                final_story = self.story_generator.generate(
                    pattern_id, pattern_info, constraints, injected_tricks
                )

                novelty_report = self.novelty_checker.check(final_story, run_id, self.user_idea)
                if logger:
                    logger.log_event("novelty_check_done", {
                        "risk_level": novelty_report.get("risk_level"),
                        "max_similarity": novelty_report.get("max_similarity"),
                        "embedding_available": novelty_report.get("embedding_available"),
                        "report_path": novelty_report.get("report_path"),
                        "pivot_attempt": pivot_attempts
                    })

            if novelty_report.get("risk_level") == "high":
                if novelty_action == "fail":
                    raise RuntimeError("Novelty check high risk after pivots")
                if logger:
                    logger.log_event("novelty_pivot_exhausted", {
                        "attempts": pivot_attempts,
                        "max_similarity": novelty_report.get("max_similarity"),
                        "action": novelty_action
                    })

            if novelty_report is not None:
                novelty_report["pivot_attempts"] = pivot_attempts
                novelty_report["action"] = novelty_action

        # Phase 4: RAG Verification
        print("\n" + "=" * 80)
        print("🔎 Phase 4: RAG Verification (查重验证)")
        print("=" * 80)
        if not PipelineConfig.VERIFICATION_ENABLE:
            print("⚠️  Verification disabled → skip Phase 4 (no collision check / no pivot)")
            print("⚠️  max_similarity shown as 0.00 because verification is disabled (not actual similarity)")
            verification_result = {
                "pass": True,
                "collision_detected": False,
                "similar_papers": [],
                "max_similarity": 0.0,
                "source": "disabled",
                "metric": "disabled",
                "skipped": True,
                "threshold": PipelineConfig.COLLISION_THRESHOLD,
            }
            if logger:
                logger.log_event("verification_skipped", {
                    "reason": "disabled",
                    "verification_enable": False,
                    "threshold": PipelineConfig.COLLISION_THRESHOLD,
                })
        else:
            verification_result = verification_from_novelty_report(
                novelty_report=novelty_report,
                collision_threshold=PipelineConfig.COLLISION_THRESHOLD
            )
            metric = verification_result.get("metric", "unknown")
            print("📌 验证来源: novelty_report")
            print(f"📌 metric: {metric}")
            if novelty_report is None:
                print("⚠️  novelty_report missing → verification metric unknown")
            else:
                embedding_available = novelty_report.get("embedding_available", False)
                print(f"📌 embedding_available: {embedding_available}")
                if metric == "keyword_overlap":
                    print("⚠️  embedding 不可用 → 降级 keyword_overlap")
            print(f"📊 max_similarity: {verification_result.get('max_similarity', 0.0):.2f}")
            if verification_result.get("similar_papers"):
                print("\n   Top-3 相似论文:")
                for i, paper in enumerate(verification_result["similar_papers"], 1):
                    print(f"   {i}. {paper.get('title', '')}")
                    print(f"      相似度: {paper.get('similarity', 0.0):.2f}")
            if logger:
                logger.log_event("verification_from_novelty", {
                    "metric": metric,
                    "max_similarity": verification_result.get("max_similarity"),
                    "collision_detected": verification_result.get("collision_detected"),
                    "embedding_available": novelty_report.get("embedding_available") if novelty_report else None
                })

        if verification_result['collision_detected']:
            print("\n❌ 检测到撞车，触发 Pivot 策略")
            if logger:
                logger.log_event("pivot_triggered", {
                    "collision_detected": True,
                    "max_similarity": verification_result.get("max_similarity")
                })

            # 生成 Pivot 约束
            constraints = self.verifier.generate_pivot_constraints(
                current_story, verification_result['similar_papers']
            )

            # 重新生成（使用 novelty 或 domain_distance 维度的 Pattern）
            if ranked_patterns.get('novelty') and len(ranked_patterns['novelty']) > 0:
                pattern_id, pattern_info, metadata = ranked_patterns['novelty'][0]
                print(f"\n🔄 切换到新颖度维度 Pattern: {pattern_id}")
            elif ranked_patterns.get('domain_distance') and len(ranked_patterns['domain_distance']) > 0:
                pattern_id, pattern_info, metadata = ranked_patterns['domain_distance'][0]
                print(f"\n🔄 切换到领域距离维度 Pattern: {pattern_id}")

            final_story = self.story_generator.generate(
                pattern_id, pattern_info, constraints, injected_tricks
            )

            # 重新查新并查重（复用 novelty_report）
            run_id = self.run_id or "run_unknown"
            novelty_report = self.novelty_checker.check(final_story, run_id, self.user_idea)
            if logger:
                logger.log_event("novelty_check_done", {
                    "risk_level": novelty_report.get("risk_level"),
                    "max_similarity": novelty_report.get("max_similarity"),
                    "embedding_available": novelty_report.get("embedding_available"),
                    "report_path": novelty_report.get("report_path"),
                    "pivot_attempt": "rag_verifier"
                })
            verification_result = verification_from_novelty_report(
                novelty_report=novelty_report,
                collision_threshold=PipelineConfig.COLLISION_THRESHOLD
            )
            metric = verification_result.get("metric", "unknown")
            print("\n🔄 重新查重（基于 novelty_report）")
            print(f"📌 metric: {metric}")
            if novelty_report is not None:
                embedding_available = novelty_report.get("embedding_available", False)
                print(f"📌 embedding_available: {embedding_available}")
                if metric == "keyword_overlap":
                    print("⚠️  embedding 不可用 → 降级 keyword_overlap")
            print(f"📊 max_similarity: {verification_result.get('max_similarity', 0.0):.2f}")

        # 输出最终结果
        success = verification_result['pass']

        print("\n" + "=" * 80)
        print("🎉 Pipeline 完成!")
        print("=" * 80)
        print(f"✅ 状态: {'成功' if success else '需人工审核'}")
        print(f"📊 迭代次数: {iterations}")
        print(f"🏆 最终版本来源: 迭代 {global_best_iteration if final_story == global_best_story else iterations}")
        print(f"📝 最终 Story:")
        print(f"   标题: {final_story.get('title', '')}")
        print(f"   摘要: {final_story.get('abstract', '')[:100]}...")
        print("=" * 80)

        return {
            'success': success,
            'final_story': final_story,
            'final_story_source': {
                'iteration': global_best_iteration if final_story == global_best_story else iterations,
                'score': global_best_score if final_story == global_best_story else critic_result['avg_score'],
                'is_best_across_iterations': final_story == global_best_story
            },
            'iterations': iterations,
            'selected_patterns': {
                k: [pid for pid, _, _ in v[:5]]  # 保存每个维度的前5个 Pattern ID
                for k, v in ranked_patterns.items() if v
            },
            'review_history': review_history,
            'refinement_history': refinement_history,
            'verification_result': verification_result,
            'novelty_report': novelty_report
        }
