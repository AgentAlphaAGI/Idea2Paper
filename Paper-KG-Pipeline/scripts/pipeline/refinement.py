from typing import List, Tuple, Dict

from .config import PipelineConfig


class RefinementEngine:
    """修正引擎: 根据 Critic 反馈进行 Pattern Injection"""

    # 通用/实验性 Trick 列表，这些 Trick 不足以提升技术新颖性
    GENERIC_TRICKS = [
        "消融实验", "多数据集验证", "对比实验", "Case Study", "案例分析",
        "可视化", "Attention 可视化", "参数敏感性分析", "鲁棒性测试",
        "现有方法局限性", "逻辑递进", "叙事结构", "性能提升", "实验验证"
    ]

    def __init__(self, recalled_patterns: List[Tuple[str, Dict, float]]):
        self.recalled_patterns = recalled_patterns
        self.used_patterns = set()  # 追踪已使用过的 Pattern，避免重复

    def refine(self, main_issue: str, suggestions: List[str]) -> List[str]:
        """根据问题类型注入 Trick

        Args:
            main_issue: 'novelty' | 'stability' | 'interpretability' | 'domain_mismatch'
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
        elif main_issue == 'interpretability':
            return self._inject_explanation_tricks()
        elif main_issue == 'domain_mismatch':
            return self._inject_domain_tricks()
        else:
            return []

    def _inject_tail_tricks(self) -> List[str]:
        """长尾注入: 选择冷门但有特色的 Trick - 注入核心方法论"""
        print("\n🎯 策略: Tail Injection (长尾注入 - 深度方法论融合)")
        print("   目标: 从 Rank 5-10 中选择 Cluster Size < 10 的冷门 Pattern，提取核心方法论")

        # 筛选候选 Pattern
        start, end = PipelineConfig.TAIL_INJECTION_RANK_RANGE
        candidates = []

        for i in range(start, min(end + 1, len(self.recalled_patterns))):
            pattern_id, pattern_info, score = self.recalled_patterns[i]
            # 避免重复使用已使用过的 Pattern
            if pattern_id in self.used_patterns:
                continue
            cluster_size = pattern_info.get('cluster_size', 999)

            if cluster_size < PipelineConfig.INNOVATIVE_CLUSTER_SIZE_THRESHOLD:
                candidates.append((pattern_id, pattern_info, cluster_size))

        if not candidates:
            print("   ⚠️  未找到符合条件的长尾 Pattern，尝试放宽条件...")
            # 放宽条件：在所有召回中找未使用的、聚类最小的
            candidates = [
                (pid, pinfo, pinfo.get('cluster_size', 999))
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

        pattern_id, pattern_info, cluster_size = selected_pattern
        # 记录已使用的 Pattern
        self.used_patterns.add(pattern_id)

        pattern_name = pattern_info.get('name', '')
        pattern_summary = pattern_info.get('summary', '')
        skeleton_examples = pattern_info.get('skeleton_examples', [])

        print(f"\n   ✅ 选择 Pattern: {pattern_id}")
        print(f"      名称: {pattern_name}")
        print(f"      聚类大小: {cluster_size} 篇（冷门）")
        print(f"      已使用 Pattern 数: {len(self.used_patterns)}")

        # 【关键改进】提取 Pattern 的核心方法论，而不是表层 trick
        method_insights = []

        # 1. 从 skeleton_examples 中提取核心方法步骤
        if skeleton_examples:
            for ex in skeleton_examples[:2]:  # 取前2个示例
                method_story = ex.get('method_story', '')
                if method_story:
                    # 提取关键短语（去除通用描述）
                    method_insights.append(method_story[:150])

        # 2. 从 top_tricks 中提取技术性 trick（过滤通用实验 trick）
        tech_tricks = []
        for trick in pattern_info.get('top_tricks', [])[:5]:
            trick_name = trick.get('name', '')
            # 过滤通用 Trick
            is_generic = any(gt in trick_name for gt in self.GENERIC_TRICKS)
            if is_generic:
                continue
            tech_tricks.append(trick_name)
            if len(tech_tricks) >= 2:
                break

        # 3. 构建注入描述（强调方法论融合）
        injection_instructions = []

        if method_insights:
            # 【核心改进】直接注入方法论的具体描述
            for i, insight in enumerate(method_insights[:1], 1):  # 取最相关的一个
                injection_instructions.append(
                    f"【方法论重构】参考 {pattern_name} 的核心技术路线：{insight}"
                )
                print(f"      注入方法论示例 {i}: {insight[:80]}...")

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
            cluster_size = pattern_info.get('cluster_size', 0)

            if cluster_size > PipelineConfig.HEAD_INJECTION_CLUSTER_THRESHOLD:
                candidates.append((pattern_id, pattern_info, cluster_size))

        if not candidates:
            # 如果没有符合条件的，选择 Cluster Size 最大的（且未使用过）
            candidates = [
                (pid, pinfo, pinfo.get('cluster_size', 0))
                for i, (pid, pinfo, _) in enumerate(self.recalled_patterns[:3])
                if pid not in self.used_patterns
            ]
            candidates.sort(key=lambda x: x[2], reverse=True)

        if not candidates:
            # 如果所有头部 Pattern 都用过了，从中间范围选择
            print("   ⚠️  头部 Pattern 已用完，尝试中间范围...")
            candidates = [
                (pid, pinfo, pinfo.get('cluster_size', 0))
                for i, (pid, pinfo, _) in enumerate(self.recalled_patterns[3:6])
                if pid not in self.used_patterns
            ]
            candidates.sort(key=lambda x: x[2], reverse=True)

        if not candidates:
            print("   ⚠️  未找到符合条件的头部 Pattern")
            return []

        selected_pattern = candidates[0]
        pattern_id, pattern_info, cluster_size = selected_pattern
        # 记录已使用的 Pattern
        self.used_patterns.add(pattern_id)

        pattern_name = pattern_info.get('name', '')
        skeleton_examples = pattern_info.get('skeleton_examples', [])

        print(f"\n   ✅ 选择 Pattern: {pattern_id}")
        print(f"      名称: {pattern_name}")
        print(f"      聚类大小: {cluster_size} 篇（成熟）")
        print(f"      已使用 Pattern 数: {len(self.used_patterns)}")

        # 【关键改进】提取稳定性相关的核心技术和方法论
        injection_instructions = []

        # 1. 从 top_tricks 中提取技术性 trick（过滤通用实验 trick）
        tech_tricks = []
        for trick in pattern_info.get('top_tricks', [])[:5]:
            trick_name = trick.get('name', '')
            # 过滤通用 Trick
            is_generic = any(gt in trick_name for gt in self.GENERIC_TRICKS)
            if is_generic:
                continue
            tech_tricks.append(trick_name)
            if len(tech_tricks) >= 2:
                break

        # 2. 从 skeleton_examples 中提取稳定性方法
        stability_methods = []
        if skeleton_examples:
            # 优先提取包含稳定性关键词的方法
            for ex in skeleton_examples[:3]:
                method_story = ex.get('method_story', '')
                if method_story and any(kw in method_story.lower() for kw in ['稳定', '鲁棒', '一致', '对抗', '正则', '混合']):
                    stability_methods.append(method_story[:150])
                    if len(stability_methods) >= 2:
                        break
            # 如果没有匹配到，直接提取前2个示例
            if not stability_methods and skeleton_examples:
                for ex in skeleton_examples[:2]:
                    method_story = ex.get('method_story', '')
                    if method_story:
                        stability_methods.append(method_story[:150])

        # 3. 构建注入指令（直接注入方法论细节）
        if stability_methods:
            # 【核心改进】直接注入稳定性方法的具体描述
            for i, method in enumerate(stability_methods[:1], 1):  # 取最相关的一个
                injection_instructions.append(
                    f"【稳定性方法论】参考 {pattern_name} 的鲁棒性设计：{method}"
                )
                print(f"      注入稳定性方法论 {i}: {method[:80]}...")

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

    def _inject_explanation_tricks(self) -> List[str]:
        """解释性注入: 增加可视化和分析"""
        print("\n🎯 策略: Explanation Injection (解释性注入)")
        print("   目标: 增加可视化和 Case Study 模块")

        tricks = [
            "增加 Attention 权重可视化分析",
            "设计代表性样本的 Case Study",
            "添加消融实验说明各组件贡献"
        ]

        for trick in tricks:
            print(f"      注入 Trick: {trick}")

        return tricks

    def _inject_domain_tricks(self) -> List[str]:
        """领域适配注入: 调整领域相关方法"""
        print("\n🎯 策略: Domain Adaptation Injection (领域适配注入)")
        print("   目标: 增加领域特定的预处理或特征工程")

        tricks = [
            "增加领域特定的数据预处理步骤",
            "设计领域相关的特征提取方法",
            "调整评估指标以适配目标领域"
        ]

        for trick in tricks:
            print(f"      注入 Trick: {trick}")

        return tricks

