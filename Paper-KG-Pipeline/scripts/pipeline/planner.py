"""
Story Planner - 基于 Pattern 生成写作框架

作用: 根据 Pattern 的 solution_approaches 和 story 生成详细的写作计划,
      指导 StoryGenerator 生成更符合 Pattern 风格的 Story
"""

from typing import Dict, List


class StoryPlanner:
    """Story 规划器: 基于 Pattern 生成写作框架"""

    def __init__(self, pattern_info: Dict):
        self.pattern_info = pattern_info
        self.pattern_name = pattern_info.get('name', '')
        self.pattern_size = pattern_info.get('size', 0)

        # 从 summary 提取核心信息
        summary = pattern_info.get('summary', {})
        if isinstance(summary, dict):
            self.representative_ideas = summary.get('representative_ideas', [])
            self.common_problems = summary.get('common_problems', [])
            self.solution_approaches = summary.get('solution_approaches', [])
            self.story_guides = summary.get('story', [])
        else:
            self.representative_ideas = []
            self.common_problems = []
            self.solution_approaches = []
            self.story_guides = []

    def generate_writing_framework(self, user_idea: str) -> Dict:
        """生成写作框架

        Args:
            user_idea: 用户的研究 Idea

        Returns:
            {
                'problem_framing_guide': '问题定位指南',
                'method_design_guide': '方法设计指南',
                'innovation_guide': '创新点包装指南',
                'story_strategy': '整体叙事策略'
            }
        """
        framework = {
            'problem_framing_guide': self._generate_problem_guide(),
            'method_design_guide': self._generate_method_guide(),
            'innovation_guide': self._generate_innovation_guide(),
            'story_strategy': self._generate_story_strategy()
        }

        return framework

    def _generate_problem_guide(self) -> str:
        """生成问题定位指南"""
        guide = f"【问题定位参考 - {self.pattern_name}】\n"

        if self.common_problems:
            guide += "\n该模式常解决以下类型的问题:\n"
            for i, problem in enumerate(self.common_problems[:2], 1):
                guide += f"{i}. {problem}\n"
            guide += "\n建议: 将你的问题与上述模式对应,找到切入点。"
        else:
            guide += "\n建议: 明确指出现有方法的局限性,引出你的研究动机。"

        return guide

    def _generate_method_guide(self) -> str:
        """生成方法设计指南"""
        guide = f"【方法设计参考 - {self.pattern_name}】\n"

        if self.solution_approaches:
            guide += "\n该模式的核心解决方案路径:\n"
            for i, approach in enumerate(self.solution_approaches, 1):
                guide += f"\n路径 {i}:\n{approach}\n"
            guide += "\n建议: 参考这些技术路线,构建你的方法框架。不要简单复制,要结合你的具体问题进行创新组合。"
        else:
            guide += "\n建议: 设计一个清晰的方法流程,包含3-5个关键步骤,每步都要有具体的技术实现细节。"

        return guide

    def _generate_innovation_guide(self) -> str:
        """生成创新点包装指南"""
        guide = f"【创新点包装参考 - {self.pattern_name}】\n"

        if self.representative_ideas:
            guide += "\n该模式的代表性研究想法:\n"
            for i, idea in enumerate(self.representative_ideas[:2], 1):
                guide += f"{i}. {idea}\n"
            guide += "\n建议: 你的创新点应该体现以下特征:\n"
            guide += "  - 与现有工作的本质区别 (不只是性能提升)\n"
            guide += "  - 技术组合的独特性 (为什么这样组合有效)\n"
            guide += "  - 方法的可泛化性 (能否应用到其他场景)"
        else:
            guide += "\n建议: 明确3个核心贡献点,避免泛泛而谈,要具体说明你的技术创新。"

        return guide

    def _generate_story_strategy(self) -> str:
        """生成整体叙事策略"""
        strategy = f"【叙事策略参考 - {self.pattern_name} ({self.pattern_size} 篇论文)】\n"

        if self.story_guides:
            strategy += "\n该模式的写作包装策略:\n"
            for i, guide in enumerate(self.story_guides, 1):
                strategy += f"{i}. {guide}\n"
            strategy += "\n建议: 用这种叙事方式包装你的工作,让评审看到你的独特视角。"
        else:
            strategy += "\n建议: 采用 '问题驱动 → 方法创新 → 效果验证' 的经典叙事结构。"

        return strategy

    def get_method_skeleton_template(self) -> str:
        """获取方法骨架模板 (供 Generator 参考)"""
        if not self.solution_approaches:
            return "步骤1: 定义问题和输入；步骤2: 设计核心算法；步骤3: 优化和验证"

        # 从 solution_approaches 提取关键步骤
        template_steps = []
        for approach in self.solution_approaches[:3]:
            # 提取前50个字符作为步骤概要
            step_summary = approach[:60].replace('\n', ' ').strip()
            template_steps.append(step_summary)

        return '；'.join(template_steps)

    def get_innovation_claims_template(self) -> List[str]:
        """获取创新点模板 (供 Generator 参考)"""
        if not self.representative_ideas:
            return [
                "提出了新的方法框架",
                "设计了高效的算法",
                "在多个数据集上验证了有效性"
            ]

        # 从 representative_ideas 提取创新点模式
        claims = []
        for idea in self.representative_ideas[:3]:
            # 提取前80个字符
            claim = idea[:80].replace('\n', ' ').strip()
            claims.append(claim)

        return claims

    def print_framework(self):
        """打印写作框架 (调试用)"""
        print("\n" + "=" * 80)
        print(f"📋 写作框架规划 - {self.pattern_name}")
        print("=" * 80)

        framework = self.generate_writing_framework("")

        print("\n" + framework['problem_framing_guide'])
        print("\n" + framework['method_design_guide'])
        print("\n" + framework['innovation_guide'])
        print("\n" + framework['story_strategy'])

        print("\n【方法骨架模板】")
        print(self.get_method_skeleton_template())

        print("\n【创新点模板】")
        for i, claim in enumerate(self.get_innovation_claims_template(), 1):
            print(f"{i}. {claim}")

        print("=" * 80)


def create_planner(pattern_info: Dict) -> StoryPlanner:
    """工厂函数: 创建 Story Planner"""
    return StoryPlanner(pattern_info)

