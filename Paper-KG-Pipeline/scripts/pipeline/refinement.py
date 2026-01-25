from typing import List, Tuple, Dict, Optional

from .config import PipelineConfig
from .idea_fusion import IdeaFusionEngine


class RefinementEngine:
    """修正引擎: 根据 Critic 反馈进行 Pattern Injection（支持按分类优先级选择）"""

    # 通用/实验性 Trick 列表，这些 Trick 不足以提升技术新颖性
    GENERIC_TRICKS = [
        "消融实验", "多数据集验证", "对比实验", "Case Study", "案例分析",
        "可视化", "Attention 可视化", "参数敏感性分析", "鲁棒性测试",
        "现有方法局限性", "逻辑递进", "叙事结构", "性能提升", "实验验证"
    ]

    def __init__(self, recalled_patterns: List[Tuple[str, Dict, float]],
                 ranked_patterns: Dict[str, List[Tuple[str, Dict, Dict]]] = None,
                 user_idea: Optional[str] = None):
        """
        Args:
            recalled_patterns: 原始召回列表（兼容旧代码）
            ranked_patterns: 按维度排序后的 Pattern（新接口）
            user_idea: 用户的原始 idea（用于 idea fusion）
        """
        self.recalled_patterns = recalled_patterns
        self.ranked_patterns = ranked_patterns or {}
        self.used_patterns = set()  # 追踪已使用过的 Pattern，避免重复
        self.dimension_indices = {'novelty': 0, 'stability': 0, 'domain_distance': 0}  # 每个维度的当前索引
        self.user_idea = user_idea
        self.fusion_engine = IdeaFusionEngine()

        # 【新增】回滚机制相关
        self.current_pattern_id = None  # 记录当前使用的 pattern_id
        self.pattern_failure_map = {}  # {pattern_id: {issue_type1, issue_type2, ...}}
        self.last_failed_pattern = None  # 记录上一个失败的 pattern

    def refine(self, main_issue: str, suggestions: List[str]) -> List[str]:
        """根据问题类型注入 Trick

        Args:
            main_issue: 'novelty' | 'stability' | 'domain_distance'
            suggestions: 建议列表

        Returns:
            injected_tricks: List[str] - 注入的 Trick 描述
        """
        print("\n" + "=" * 80)
        print("🔧 Phase 3.5: Refinement (修正注入)")
        print("=" * 80)
        print(f"📌 诊断问题: {main_issue}")
        print(f"💡 建议策略: {', '.join(suggestions)}")

        if main_issue == 'novelty':
            return self._inject_tail_tricks()
        elif main_issue == 'stability':
            return self._inject_head_tricks()
        elif main_issue == 'domain_distance':
            return self._inject_cross_domain_tricks()
        else:
            return []

    def refine_with_idea_fusion(self, main_issue: str, suggestions: List[str],
                                previous_story: Optional[Dict] = None,
                                force_next_pattern: bool = False) -> Tuple[List[str], Optional[Dict]]:
        """
        【新方法】基于 Idea Fusion 的修正引擎

        不仅提取 Pattern 的技术点，而是在概念层融合两个 idea，生成创新的新想法

        Args:
            main_issue: 诊断问题类型
            suggestions: 建议策略列表
            previous_story: 前一轮生成的 Story
            force_next_pattern: 是否强制使用下一个 Pattern（在新颖性模式中使用）

        Returns:
            (injected_tricks, fused_idea_result)
            - injected_tricks: 兼容旧接口的 trick 列表
            - fused_idea_result: 融合后的 idea 完整结构（给 story_generator 使用）
        """
        print("\n" + "=" * 80)
        print("🔧 Phase 3.5: Refinement (创新融合修正)")
        print("=" * 80)
        print(f"📌 诊断问题: {main_issue}")
        print(f"💡 建议策略: {', '.join(suggestions)}")

        # Step 1: 选择要融合的 Pattern
        selected_pattern = self._select_pattern_for_fusion(main_issue, force_next=force_next_pattern)
        if not selected_pattern:
            print("   ⚠️  未找到合适的 Pattern，降级到传统 trick 注入")
            return self.refine(main_issue, suggestions), None

        pattern_id, pattern_info = selected_pattern

        # Step 2: 进行 Idea Fusion
        print(f"\n   🔄 选中 Pattern: {pattern_id} - {pattern_info.get('name', '')}")
        print(f"      进行概念层创新融合...")

        if not self.user_idea:
            print("   ⚠️  user_idea 未配置，无法进行 idea fusion")
            return self._inject_tail_tricks() if main_issue == 'novelty' else [], None

        fused_result = self.fusion_engine.fuse(
            user_idea=self.user_idea,
            pattern_id=pattern_id,
            pattern_info=pattern_info,
            previous_story=previous_story
        )

        # Step 3: 返回融合结果和兼容的 trick 格式
        injected_tricks = [
            f"【创新融合】{fused_result.get('fused_idea_title', '')}",
            f"【新问题定义】{fused_result.get('problem_framing', '')[:100]}...",
            f"【新创新主张】{fused_result.get('novelty_claim', '')[:100]}...",
        ]

        return injected_tricks, fused_result

    def _select_pattern_for_fusion(self, main_issue: str, force_next: bool = False) -> Optional[Tuple[str, Dict]]:
        """为 idea fusion 选择最合适的 Pattern

        跳过已失败的 pattern（即该 pattern 对该 issue 曾导致分数下降）

        Args:
            main_issue: 问题类型
            force_next: 是否强制选择下一个 pattern（用于新颖性模式的循环遍历）
        """
        # 根据问题类型选择对应维度
        dimension_map = {
            'novelty': 'novelty',
            'stability': 'stability',
            'domain_distance': 'domain_distance'
        }

        dimension = dimension_map.get(main_issue, 'novelty')

        if self.ranked_patterns and dimension in self.ranked_patterns:
            patterns = self.ranked_patterns[dimension]
            idx = self.dimension_indices[dimension]

            # 如果强制使用下一个，跳过已使用的
            if force_next:
                idx = self.dimension_indices[dimension]

            while idx < len(patterns):
                pattern_id, pattern_info, metadata = patterns[idx]

                # 【新增】检查该 pattern 是否对该 issue 已失败
                if self._is_pattern_failed_for_issue(pattern_id, main_issue):
                    print(f"      ⏭️  跳过已失败的 {pattern_id} (对 {main_issue} 无效)")
                    idx += 1
                    self.dimension_indices[dimension] = idx
                    continue

                # 在 force_next 模式下，直接返回当前 pattern（即使已使用过）
                # 在普通模式下，只返回未使用的 pattern
                if force_next or pattern_id not in self.used_patterns:
                    self.used_patterns.add(pattern_id)  # 标记为已使用
                    self.current_pattern_id = pattern_id  # 记录当前使用的 pattern
                    self.dimension_indices[dimension] = idx + 1  # 更新索引，下次从下一个开始
                    print(f"      ✅ 选中{dimension}维度 Pattern: {pattern_id} (索引: {idx})")
                    return (pattern_id, pattern_info)
                idx += 1

        return None

    def _is_pattern_failed_for_issue(self, pattern_id: str, issue_type: str) -> bool:
        """检查该 pattern 是否对该 issue 已失败"""
        if pattern_id in self.pattern_failure_map:
            return issue_type in self.pattern_failure_map[pattern_id]
        return False

    def mark_pattern_failed(self, pattern_id: str, issue_type: str):
        """标记某个 pattern 对某个 issue 类型失败，避免再次使用"""
        if pattern_id:
            if pattern_id not in self.pattern_failure_map:
                self.pattern_failure_map[pattern_id] = set()
            self.pattern_failure_map[pattern_id].add(issue_type)
            self.last_failed_pattern = pattern_id
            print(f"\n   📝 记录失败映射: {pattern_id} 对 {issue_type} 无效")

    def _inject_tail_tricks(self) -> List[str]:
        """长尾注入: 选择冷门但有特色的 Trick - 注入核心方法论（优先从 novelty 维度选择）"""
        print("\n🎯 策略: Tail Injection (长尾注入 - 深度方法论融合)")
        print("   目标: 优先从 novelty 维度中按序选择未使用的 Pattern")

        # 【优化】优先从 novelty 维度中选择（新颖性最高的）
        if self.ranked_patterns and 'novelty' in self.ranked_patterns:
            novelty_patterns = self.ranked_patterns['novelty']
            idx = self.dimension_indices['novelty']

            # 从当前索引开始查找未使用的 Pattern
            while idx < len(novelty_patterns):
                pattern_id, pattern_info, metadata = novelty_patterns[idx]
                self.dimension_indices['novelty'] = idx + 1  # 更新索引

                # 【新增】检查是否已失败
                if self._is_pattern_failed_for_issue(pattern_id, 'novelty'):
                    print(f"      ⏭️  跳过已失败的 {pattern_id}")
                    idx += 1
                    continue

                if pattern_id not in self.used_patterns:
                    self.used_patterns.add(pattern_id)
                    self.current_pattern_id = pattern_id  # 【新增】记录当前 pattern
                    size = pattern_info.get('size', 0)

                    print(f"\n   ✅ 选择 Pattern: {pattern_id} (来自 novelty 维度, 排序位置={idx+1})")
                    print(f"      名称: {pattern_info.get('name', '')}")
                    print(f"      聚类大小: {size} 篇（冷门）")

                    scores = metadata.get('scores', {})
                    if scores:
                        print(f"      新颖度: {scores.get('novelty_score', 0):.2f}")

                    return self._extract_pattern_insights(pattern_id, pattern_info, 'novelty')

                idx += 1

            print("   ⚠️  novelty 维度中的所有 Pattern 已用完，回退到传统方法")

        # 【Fallback】传统方法：筛选候选 Pattern
        start, end = PipelineConfig.TAIL_INJECTION_RANK_RANGE
        candidates = []

        for i in range(start, min(end + 1, len(self.recalled_patterns))):
            pattern_id, pattern_info, score = self.recalled_patterns[i]
            # 避免重复使用已使用过的 Pattern
            if pattern_id in self.used_patterns:
                continue
            size = pattern_info.get('size', 999)

            if size < PipelineConfig.INNOVATIVE_CLUSTER_SIZE_THRESHOLD:
                candidates.append((pattern_id, pattern_info, size))

        if not candidates:
            print("   ⚠️  未找到符合条件的长尾 Pattern，尝试放宽条件...")
            # 放宽条件：在所有召回中找未使用的、聚类最小的
            candidates = [
                (pid, pinfo, pinfo.get('size', 999))
                for pid, pinfo, _ in self.recalled_patterns
                if pid not in self.used_patterns
            ]
            candidates.sort(key=lambda x: x[2])

        if not candidates:
            print("   ⚠️  所有召回 Pattern 已用尽，注入通用创新算子")
            return ["引入对比学习负采样优化策略", "设计多尺度特征融合机制", "添加自适应动态权重分配"]

        # 选择 Cluster Size 最小的
        candidates.sort(key=lambda x: x[2])
        selected_pattern = candidates[0]

        pattern_id, pattern_info, size = selected_pattern
        # 记录已使用的 Pattern
        self.used_patterns.add(pattern_id)

        print(f"\n   ✅ 选择 Pattern: {pattern_id} (Fallback 模式)")
        print(f"      名称: {pattern_info.get('name', '')}")
        print(f"      聚类大小: {size} 篇（冷门）")
        print(f"      已使用 Pattern 数: {len(self.used_patterns)}")

        return self._extract_pattern_insights(pattern_id, pattern_info, 'innovative')

    def _extract_pattern_insights(self, pattern_id: str, pattern_info: Dict, category: str) -> List[str]:
        """提取 Pattern 的核心方法论和技术点"""
        pattern_name = pattern_info.get('name', '')

        # 【关键改进】提取 Pattern 的核心方法论（适配新结构）
        method_insights = []

        # 新结构: 从 summary 字段提取
        summary = pattern_info.get('summary', {})
        if isinstance(summary, dict):
            solution_approaches = summary.get('solution_approaches', [])[:2]  # 取前2个
            story_guides = summary.get('story', [])[:1]
        else:
            solution_approaches = []
            story_guides = []

        # 1. 从 solution_approaches 中提取核心方法论
        if solution_approaches:
            for approach in solution_approaches:
                if approach:
                    method_insights.append(approach[:200])  # 保留更多细节

        # 2. 兼容旧结构: 从 skeleton_examples 中提取
        skeleton_examples = pattern_info.get('skeleton_examples', [])
        if skeleton_examples and not method_insights:
            for ex in skeleton_examples[:2]:
                method_story = ex.get('method_story', '')
                if method_story:
                    method_insights.append(method_story[:150])

        # 3. 从 common_tricks 提取技术点 (兼容旧结构)
        tech_tricks = []
        for trick in pattern_info.get('common_tricks', [])[:5]:
            if isinstance(trick, dict):
                trick_name = trick.get('trick_name', '')
            else:
                trick_name = str(trick)
            # 过滤通用 Trick
            is_generic = any(gt in trick_name for gt in self.GENERIC_TRICKS)
            if is_generic:
                continue
            tech_tricks.append(trick_name)
            if len(tech_tricks) >= 2:
                break

        # 4. 构建注入描述（强调方法论融合）
        injection_instructions = []

        if method_insights:
            # 【核心改进】直接注入方法论的具体描述
            for i, insight in enumerate(method_insights[:1], 1):  # 取最相关的一个
                injection_instructions.append(
                    f"【方法论重构】参考 {pattern_name} 的核心技术路线：{insight}"
                )
                print(f"      注入方法论示例 {i}: {insight[:80]}...")

        if story_guides:
            # 添加写作包装策略
            injection_instructions.append(
                f"【包装策略】{story_guides[0]}"
            )
            print(f"      注入包装策略: {story_guides[0][:80]}...")

        if tech_tricks:
            # 补充具体技术名称
            injection_instructions.append(
                f"【核心技术】融合 {pattern_name} 的关键技术点：{' + '.join(tech_tricks)}"
            )
            for trick in tech_tricks:
                print(f"      注入核心技术: {trick}")

        if not injection_instructions:
            injection_instructions.append(f"融合 {pattern_name} 的核心思路，重构现有方法论")

        return injection_instructions

    def _inject_head_tricks(self) -> List[str]:
        """头部注入: 选择成熟稳健的 Trick - 注入稳定性方法论"""
        print("\n🎯 策略: Head Injection (头部注入 - 稳定性方法论融合)")
        print(f"   目标: 从 Rank 1-3 中选择 Cluster Size > {PipelineConfig.HEAD_INJECTION_CLUSTER_THRESHOLD} 的成熟 Pattern，提取稳定性技术")

        # 筛选候选 Pattern
        start, end = PipelineConfig.HEAD_INJECTION_RANK_RANGE
        candidates = []

        for i in range(start, min(end + 1, len(self.recalled_patterns))):
            pattern_id, pattern_info, score = self.recalled_patterns[i]
            # 避免重复使用已使用过的 Pattern
            if pattern_id in self.used_patterns:
                continue
            size = pattern_info.get('size', 0)

            if size > PipelineConfig.HEAD_INJECTION_CLUSTER_THRESHOLD:
                candidates.append((pattern_id, pattern_info, size))

        if not candidates:
            # 如果没有符合条件的，选择 Cluster Size 最大的（且未使用过）
            candidates = [
                (pid, pinfo, pinfo.get('size', 0))
                for i, (pid, pinfo, _) in enumerate(self.recalled_patterns[:3])
                if pid not in self.used_patterns
            ]
            candidates.sort(key=lambda x: x[2], reverse=True)

        if not candidates:
            # 如果所有头部 Pattern 都用过了，从中间范围选择
            print("   ⚠️  头部 Pattern 已用完，尝试中间范围...")
            candidates = [
                (pid, pinfo, pinfo.get('size', 0))
                for i, (pid, pinfo, _) in enumerate(self.recalled_patterns[3:6])
                if pid not in self.used_patterns
            ]
            candidates.sort(key=lambda x: x[2], reverse=True)

        if not candidates:
            print("   ⚠️  未找到符合条件的头部 Pattern")
            return []

        selected_pattern = candidates[0]
        pattern_id, pattern_info, size = selected_pattern
        # 记录已使用的 Pattern
        self.used_patterns.add(pattern_id)

        pattern_name = pattern_info.get('name', '')
        skeleton_examples = pattern_info.get('skeleton_examples', [])

        print(f"\n   ✅ 选择 Pattern: {pattern_id}")
        print(f"      名称: {pattern_name}")
        print(f"      聚类大小: {size} 篇（成熟）")
        print(f"      已使用 Pattern 数: {len(self.used_patterns)}")

        # 【关键改进】提取稳定性相关的核心技术和方法论（适配新结构）
        injection_instructions = []

        # 新结构: 从 summary 字段提取
        summary = pattern_info.get('summary', {})
        if isinstance(summary, dict):
            solution_approaches = summary.get('solution_approaches', [])
            story_guides = summary.get('story', [])[:1]
        else:
            solution_approaches = []
            story_guides = []

        # 1. 从 solution_approaches 中提取稳定性方法
        stability_methods = []
        if solution_approaches:
            # 优先提取包含稳定性关键词的方法
            for approach in solution_approaches:
                if approach and any(kw in approach.lower() for kw in ['稳定', '鲁棒', '一致', '对抗', '正则', '混合', 'robust', 'stable']):
                    stability_methods.append(approach[:200])
                    if len(stability_methods) >= 2:
                        break
            # 如果没有匹配到，直接提取前2个
            if not stability_methods and solution_approaches:
                for approach in solution_approaches[:2]:
                    if approach:
                        stability_methods.append(approach[:200])

        # 2. 兼容旧结构: 从 skeleton_examples 中提取
        skeleton_examples = pattern_info.get('skeleton_examples', [])
        if skeleton_examples and not stability_methods:
            for ex in skeleton_examples[:3]:
                method_story = ex.get('method_story', '')
                if method_story and any(kw in method_story.lower() for kw in ['稳定', '鲁棒', '一致', '对抗', '正则', '混合']):
                    stability_methods.append(method_story[:150])
                    if len(stability_methods) >= 2:
                        break
            if not stability_methods and skeleton_examples:
                for ex in skeleton_examples[:2]:
                    method_story = ex.get('method_story', '')
                    if method_story:
                        stability_methods.append(method_story[:150])

        # 3. 从 common_tricks 提取技术点 (兼容旧结构)
        tech_tricks = []
        for trick in pattern_info.get('common_tricks', [])[:5]:
            if isinstance(trick, dict):
                trick_name = trick.get('trick_name', '')
            else:
                trick_name = str(trick)
            # 过滤通用 Trick
            is_generic = any(gt in trick_name for gt in self.GENERIC_TRICKS)
            if is_generic:
                continue
            tech_tricks.append(trick_name)
            if len(tech_tricks) >= 2:
                break

        # 4. 构建注入指令（直接注入方法论细节）
        if stability_methods:
            # 【核心改进】直接注入稳定性方法的具体描述
            for i, method in enumerate(stability_methods[:1], 1):  # 取最相关的一个
                injection_instructions.append(
                    f"【稳定性方法论】参考 {pattern_name} 的鲁棒性设计：{method}"
                )
                print(f"      注入稳定性方法论 {i}: {method[:80]}...")

        if story_guides:
            # 添加写作包装策略
            injection_instructions.append(
                f"【包装策略】{story_guides[0]}"
            )
            print(f"      注入包装策略: {story_guides[0][:80]}...")

        if tech_tricks:
            # 补充具体技术名称
            injection_instructions.append(
                f"【稳定性技术】融合 {pattern_name} 的成熟技术：{' + '.join(tech_tricks)}"
            )
            for trick in tech_tricks:
                print(f"      注入稳定性技术: {trick}")

        if not injection_instructions:
            injection_instructions.append(f"融合 {pattern_name} 的成熟方法，增强技术稳定性")

        return injection_instructions

    def _inject_cross_domain_tricks(self) -> List[str]:
        """跨域注入: 从domain_distance维度选择Pattern,引入不同视角"""
        print("\n🎯 策略: Cross-Domain Injection (跨域视角注入)")
        print("   目标: 从domain_distance维度选择Pattern,引入跨域视角优化叙事")

        # 优先从 domain_distance 维度中选择（领域距离适中的）
        if self.ranked_patterns and 'domain_distance' in self.ranked_patterns:
            domain_patterns = self.ranked_patterns['domain_distance']
            idx = self.dimension_indices['domain_distance']

            # 从当前索引开始查找未使用的 Pattern
            while idx < len(domain_patterns):
                pattern_id, pattern_info, metadata = domain_patterns[idx]
                self.dimension_indices['domain_distance'] = idx + 1  # 更新索引

                # 检查是否已失败
                if self._is_pattern_failed_for_issue(pattern_id, 'domain_distance'):
                    print(f"      ⏭️  跳过已失败的 {pattern_id}")
                    idx += 1
                    continue

                if pattern_id not in self.used_patterns:
                    self.used_patterns.add(pattern_id)
                    self.current_pattern_id = pattern_id
                    size = pattern_info.get('size', 0)

                    print(f"\n   ✅ 选择 Pattern: {pattern_id} (来自 domain_distance 维度, 排序位置={idx+1})")
                    print(f"      名称: {pattern_info.get('name', '')}")
                    print(f"      聚类大小: {size} 篇")

                    scores = metadata.get('scores', {})
                    if scores:
                        print(f"      领域距离: {scores.get('domain_distance', 0):.2f}")

                    return self._extract_pattern_insights(pattern_id, pattern_info, 'cross_domain')

                idx += 1

            print("   ⚠️  domain_distance 维度中的所有 Pattern 已用完")

        # Fallback: 从召回结果中选择未使用的Pattern
        print("   ⚠️  使用Fallback策略: 从召回结果中选择未使用Pattern")
        for pattern_id, pattern_info, score in self.recalled_patterns:
            if pattern_id not in self.used_patterns:
                self.used_patterns.add(pattern_id)
                self.current_pattern_id = pattern_id
                print(f"\n   ✅ 选择 Pattern: {pattern_id} (Fallback 模式)")
                print(f"      名称: {pattern_info.get('name', '')}")
                return self._extract_pattern_insights(pattern_id, pattern_info, 'cross_domain')

        print("   ⚠️  所有召回 Pattern 已用尽")
        return []

