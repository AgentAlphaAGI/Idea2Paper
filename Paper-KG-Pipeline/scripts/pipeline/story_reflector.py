"""
Story Reflector: 在 Story 生成过程中加入反思融合机制

确保新 Pattern 的注入能够**有机融合**旧 idea 和新 pattern，
而不是生硬的技术叠加。通过多轮反思和验证来确保融合的逻辑连贯性和创新性。
"""

import json
from typing import Dict, Optional, List

from .utils import call_llm, parse_json_from_llm


class StoryReflector:
    """故事反思器：在融合过程中进行多层验证和迭代优化"""

    def __init__(self):
        pass

    def reflect_on_fusion(self,
                         old_story: Dict,
                         new_pattern: Dict,
                         fused_idea: Optional[Dict],
                         critic_feedback: Dict,
                         user_idea: str) -> Dict:
        """
        进行反思融合：确保新 pattern 的注入有机融合而非生硬拼接

        Args:
            old_story: 前一个版本的 Story
            new_pattern: 新注入的 Pattern 信息
            fused_idea: 融合后的创意（由 IdeaFusionEngine 生成）
            critic_feedback: 评审反馈
            user_idea: 用户原始想法

        Returns:
            {
                'fusion_quality_score': float (0-1),  # 融合质量评分
                'is_organic': bool,  # 是否有机融合（而不是生硬拼接）
                'fusion_insights': str,  # 融合洞察
                'suggested_title': str,  # 建议的标题改进
                'suggested_method_evolution': str,  # 建议的方法演进
                'coherence_analysis': str,  # 连贯性分析
                'ready_for_generation': bool,  # 是否准备好生成 Story
                'critic_warnings': List[str],  # Critic的关键警告
            }
        """

        print("\n" + "=" * 80)
        print("🔍 Phase 3.6: Story Reflection (故事反思融合)")
        print("=" * 80)

        # Step 0: 【新增】分析Critic反馈，提取关键警告
        print("\n⚠️  Step 0: 分析Critic负面反馈...")
        critic_warnings = self._extract_critic_warnings(critic_feedback)
        if critic_warnings:
            print(f"   发现 {len(critic_warnings)} 个关键警告:")
            for warning in critic_warnings:
                print(f"      • {warning[:100]}...")

        # Step 1: 分析融合点
        print("\n📊 Step 1: 分析融合点...")
        fusion_analysis = self._analyze_fusion_points(
            old_story, new_pattern, fused_idea, user_idea
        )

        # Step 2: 检查逻辑连贯性（结合Critic警告）
        print("\n🔗 Step 2: 检查逻辑连贯性...")
        coherence_check = self._check_coherence(
            old_story, new_pattern, fusion_analysis, critic_warnings
        )

        # Step 3: 评估融合质量（考虑Critic反馈）
        print("\n⭐ Step 3: 评估融合质量...")
        quality_score = self._evaluate_fusion_quality(
            fusion_analysis, coherence_check, fused_idea, critic_warnings
        )

        # Step 4: 生成融合建议（基于Critic反馈）
        print("\n💡 Step 4: 生成融合建议...")
        suggestions = self._generate_fusion_suggestions(
            fusion_analysis, coherence_check, quality_score, old_story, critic_warnings
        )

        result = {
            'fusion_quality_score': quality_score,
            'is_organic': quality_score >= 0.65,  # 质量分 >= 0.65 认为是有机融合
            'fusion_insights': fusion_analysis.get('insights', ''),
            'suggested_title': suggestions.get('title_evolution', ''),
            'suggested_method_evolution': suggestions.get('method_evolution', ''),
            'coherence_analysis': coherence_check.get('analysis', ''),
            'ready_for_generation': quality_score >= 0.65,
            'critic_warnings': critic_warnings,  # 保存Critic警告
            'full_analysis': {
                'fusion_analysis': fusion_analysis,
                'coherence_check': coherence_check,
                'suggestions': suggestions
            }
        }

        print(f"\n✅ 融合质量评分: {quality_score:.2f}/1.0")
        print(f"✅ 融合方式: {'有机融合' if result['is_organic'] else '需要优化'}")
        print(f"✅ 准备生成: {'是' if result['ready_for_generation'] else '否'}")

        if not result['ready_for_generation'] and critic_warnings:
            print(f"⚠️  警告: 融合可能存在问题，请注意Critic提出的{len(critic_warnings)}个关键问题")

        return result

    def _extract_critic_warnings(self, critic_feedback: Dict) -> List[str]:
        """
        从Critic反馈中提取关键警告（特别是关于"技术堆砌"、"常见套路"的负面评价）

        Returns:
            List of warning strings extracted from critic feedback
        """
        warnings = []

        # 关键词列表：这些词出现时说明Critic认为存在严重问题
        critical_keywords = [
            '堆砌', '堆叠', 'stacking', 'combination', 'A+B',
            '常见', '套路', 'common', 'typical', 'conventional',
            '缺乏创新', 'lack of novelty', 'insufficient innovation',
            '已有大量', 'widely explored', '频繁出现',
            '相似工作', 'similar work', 'existing methods',
            '简单组合', 'simple integration', 'straightforward'
        ]

        for review in critic_feedback.get('reviews', []):
            feedback_text = review.get('feedback', '')

            # 检查是否包含关键负面词汇
            has_critical_keyword = any(keyword in feedback_text.lower() for keyword in critical_keywords)

            # 如果评分低（<6）且包含关键词，或者评分极低（<5.5），提取为警告
            score = review.get('score', 10)
            if (score < 6.0 and has_critical_keyword) or score < 5.5:
                warning = f"[{review.get('role', 'Reviewer')}] {feedback_text[:200]}"
                warnings.append(warning)

        return warnings

    def _analyze_fusion_points(self,
                               old_story: Dict,
                               new_pattern: Dict,
                               fused_idea: Optional[Dict],
                               user_idea: str) -> Dict:
        """
        分析融合点：识别旧 idea 和新 pattern 如何融合

        Returns:
            {
                'old_core_concepts': List[str],  # 旧 story 的核心概念
                'pattern_core_concepts': List[str],  # Pattern 的核心概念
                'fusion_points': List[Dict],  # 融合点列表
                'insights': str  # 融合洞察
            }
        """

        prompt = f"""
你是一位资深的研究论文编者。分析以下信息中旧 Story、新 Pattern 和融合 Idea 如何相互关联。

【用户原始 Idea】
{user_idea}

【旧 Story 的核心】
标题: {old_story.get('title', '')}
摘要: {old_story.get('abstract', '')[:300]}...
问题框架: {old_story.get('problem_framing', '')[:200]}...
创新主张: {json.dumps(old_story.get('innovation_claims', []), ensure_ascii=False)[:300]}...

【新 Pattern 信息】
名称: {new_pattern.get('name', '')}
描述: {new_pattern.get('description', '')[:300]}...
关键方法: {json.dumps(new_pattern.get('skeleton_examples', []), ensure_ascii=False)[:300]}...

【融合后的创新 Idea】
{json.dumps(fused_idea, ensure_ascii=False) if fused_idea else 'N/A'}

请执行以下分析，并返回 JSON 格式的结果（不要返回其他文本）：

{{
  "old_core_concepts": ["概念1", "概念2", "..."],  // 从旧 Story 中提取 2-3 个核心概念
  "pattern_core_concepts": ["概念1", "概念2", "..."],  // 从新 Pattern 中提取 2-3 个核心概念
  "fusion_points": [
    {{
      "old_concept": "旧 Story 中的概念",
      "pattern_concept": "Pattern 中的概念",
      "fusion_opportunity": "如何融合的机会点",
      "implementation_path": "具体实现路径"
    }},
    // ... 更多融合点
  ],
  "insights": "对融合过程的总体洞察（100-200字）"
}}
"""

        try:
            response = call_llm(prompt, temperature=0.5, max_tokens=1000, timeout=120)
            result = parse_json_from_llm(response)
            if not result:
                result = self._default_fusion_analysis(old_story, new_pattern)
        except Exception as e:
            print(f"⚠️  融合点分析出错: {e}")
            result = self._default_fusion_analysis(old_story, new_pattern)

        print(f"   🔎 发现 {len(result.get('fusion_points', []))} 个融合点")
        for i, point in enumerate(result.get('fusion_points', [])[:3], 1):
            print(f"      {i}. {point.get('old_concept', '')} ←→ {point.get('pattern_concept', '')}")

        return result

    def _check_coherence(self,
                        old_story: Dict,
                        new_pattern: Dict,
                        fusion_analysis: Dict,
                        critic_warnings: List[str] = None) -> Dict:
        """
        检查融合后的逻辑连贯性：确保新技术能自然地融入现有框架

        【关键改进】结合Critic警告，如果Critic指出"技术堆砌"问题，降低连贯性评分

        Returns:
            {
                'coherence_score': float (0-1),  // 连贯性评分
                'potential_conflicts': List[str],  // 潜在冲突
                'strengths': List[str],  // 融合的优势
                'analysis': str  // 详细分析
            }
        """

        critic_warnings = critic_warnings or []

        fusion_points = fusion_analysis.get('fusion_points', [])
        old_method = old_story.get('method_skeleton', '')

        prompt = f"""
评估以下融合的逻辑连贯性。一个良好的融合应该能自然地融入现有框架，而不是生硬拼接。

【旧 Story 的方法】
{old_method[:400]}

【融合点信息】
{json.dumps(fusion_points[:3], ensure_ascii=False, indent=2)}

【新 Pattern 名称】
{new_pattern.get('name', '')}

请从以下角度分析连贯性，返回 JSON 格式的结果（不要返回其他文本）：

{{
  "coherence_score": 0.8,  // 0-1 之间的评分，表示新技术与现有方法的融合程度
  "potential_conflicts": ["冲突1", "冲突2"],  // 可能的逻辑冲突或不匹配点
  "strengths": ["优势1", "优势2"],  // 融合的强处
  "analysis": "详细分析（200-300字）：解释这个融合为什么在逻辑上是连贯的（或需要改进）"
}}
"""

        try:
            response = call_llm(prompt, temperature=0.5, max_tokens=800, timeout=120)
            result = parse_json_from_llm(response)
            if not result:
                result = self._default_coherence_check()
        except Exception as e:
            print(f"⚠️  连贯性检查出错: {e}")
            result = self._default_coherence_check()

        # 【关键】如果Critic警告中提到"堆砌"、"常见套路"等，大幅降低连贯性评分
        if critic_warnings:
            penalty = 0.0
            for warning in critic_warnings:
                warning_lower = warning.lower()
                if any(word in warning_lower for word in ['堆砌', 'stacking', 'a+b', '简单组合', 'simple combination']):
                    penalty += 0.15  # 每个严重警告降低0.15分
                elif any(word in warning_lower for word in ['常见', 'common', '套路', 'typical']):
                    penalty += 0.10  # 每个中等警告降低0.10分

            if penalty > 0:
                original_score = result.get('coherence_score', 0.7)
                result['coherence_score'] = max(0.3, original_score - penalty)
                result['analysis'] = result.get('analysis', '') + f"\n\n⚠️ Critic警告惩罚: -{penalty:.2f}分。存在技术堆砌或常见套路问题，需要深度重构而非简单组合。"
                if not result.get('potential_conflicts'):
                    result['potential_conflicts'] = []
                result['potential_conflicts'].append("Critic指出融合方式过于常见，缺乏真正的概念创新")
                print(f"   ⚠️  应用Critic警告惩罚: -{penalty:.2f}分 (原{original_score:.2f} → 现{result['coherence_score']:.2f})")

        coherence_score = result.get('coherence_score', 0.5)
        print(f"   🔗 连贯性评分: {coherence_score:.2f}/1.0")
        if result.get('potential_conflicts'):
            print(f"   ⚠️  潜在冲突: {', '.join(result['potential_conflicts'][:2])}")
        if result.get('strengths'):
            print(f"   ✅ 融合优势: {', '.join(result['strengths'][:2])}")

        return result

    def _evaluate_fusion_quality(self,
                                fusion_analysis: Dict,
                                coherence_check: Dict,
                                fused_idea: Optional[Dict],
                                critic_warnings: List[str] = None) -> float:
        """
        评估总体融合质量

        【关键改进】如果有Critic警告，进一步降低质量评分

        融合质量 = 0.4 * coherence_score + 0.4 * fusion_richness + 0.2 * fused_idea_bonus
        """

        critic_warnings = critic_warnings or []

        # 连贯性分数（权重 0.4）- 已经考虑了Critic警告
        coherence_score = coherence_check.get('coherence_score', 0.5)

        # 融合丰富度（权重 0.4）：融合点越多，融合越丰富
        fusion_points = fusion_analysis.get('fusion_points', [])
        fusion_richness = min(len(fusion_points) / 3.0, 1.0)  # 3 个以上融合点得分 1.0

        # 融合 Idea 奖励（权重 0.2）：如果有融合 Idea，加分
        fused_idea_bonus = 0.3 if fused_idea and fused_idea.get('fused_idea_description') else 0.0

        quality_score = (
            0.4 * coherence_score +
            0.4 * fusion_richness +
            0.2 * fused_idea_bonus
        )

        # 【额外惩罚】如果有多个Critic警告，再额外降低最终质量分
        if len(critic_warnings) >= 2:
            quality_score = max(0.3, quality_score - 0.10)  # 多个警告时额外-0.10
            print(f"   📉 多个Critic警告，额外降低质量分: -{0.10:.2f}")

        return min(quality_score, 1.0)

    def _generate_fusion_suggestions(self,
                                   fusion_analysis: Dict,
                                   coherence_check: Dict,
                                   quality_score: float,
                                   old_story: Dict,
                                   critic_warnings: List[str] = None) -> Dict:
        """
        生成融合建议：指导 Story 生成

        【关键改进】如果有Critic警告，建议更激进的重构策略

        Returns:
            {
                'title_evolution': str,  // 建议如何演进标题
                'method_evolution': str,  // 建议如何演进方法
                'narrative_strategy': str  // 叙事策略
            }
        """

        critic_warnings = critic_warnings or []

        # 【关键】如果有Critic警告，自动建议激进策略
        if critic_warnings:
            return {
                'title_evolution': '彻底重新定义问题视角，避免使用Pattern的常见术语',
                'method_evolution': '从问题假设层面重构方法，而不是在技术层面组合',
                'narrative_strategy': f'⚠️ Critic已警告: 避免技术堆砌！需要展示**为什么这个组合创造了新的研究视角**，而不是"A+B"。参考Critic具体反馈: {critic_warnings[0][:100]}...'
            }

        if quality_score < 0.5:
            # 融合质量不佳，建议采用保守策略
            return {
                'title_evolution': '在原标题基础上，添加新 Pattern 的关键词',
                'method_evolution': '保留原有方法框架，在关键步骤中融合新技术',
                'narrative_strategy': '递进式融合：先保留原有逻辑，再逐步引入新技术'
            }
        elif quality_score < 0.75:
            # 融合质量中等，建议采用平衡策略
            return {
                'title_evolution': '重新框架化标题，体现融合后的新观点',
                'method_evolution': '部分重构方法，将新技术作为方法的核心组件',
                'narrative_strategy': '并行融合：同时强调原有 Idea 和新技术的协同价值'
            }
        else:
            # 融合质量良好，建议采用激进策略
            return {
                'title_evolution': '以融合后的新观点重新命名，体现创新高度',
                'method_evolution': '根本性重构方法，将新技术与原 Idea 深度整合',
                'narrative_strategy': '创新融合：强调这是一个新的研究方向，而不是简单组合'
            }

    @staticmethod
    def _default_fusion_analysis(old_story: Dict, new_pattern: Dict) -> Dict:
        """默认融合分析"""
        return {
            'old_core_concepts': [
                old_story.get('title', '')[:30],
                'reasoning',
                'efficiency'
            ],
            'pattern_core_concepts': [
                new_pattern.get('name', '')[:30],
                'methodology',
                'innovation'
            ],
            'fusion_points': [
                {
                    'old_concept': '原有方法框架',
                    'pattern_concept': 'Pattern 的核心技术',
                    'fusion_opportunity': '在方法论层融合新技术',
                    'implementation_path': '将新技术作为方法的补充组件'
                }
            ],
            'insights': '这个融合点代表了对原有方法的技术升级和创新发展。'
        }

    @staticmethod
    def _default_coherence_check() -> Dict:
        """默认连贯性检查"""
        return {
            'coherence_score': 0.6,
            'potential_conflicts': [],
            'strengths': ['技术补充', '方法升级'],
            'analysis': '融合后的方法在逻辑上是可行的，需要在叙事中重点说明融合的必要性。'
        }

