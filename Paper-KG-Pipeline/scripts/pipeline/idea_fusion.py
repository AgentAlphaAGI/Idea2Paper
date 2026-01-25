"""
IdeaFusionEngine - 在概念层面融合两个研究 idea

核心设计思想：
1. 不只是抽取和拼接两个 idea 的技术点
2. 而是在问题空间、假设、创新点的层面发现它们的交集和融合点
3. 生成真正创新的新想法，而不是"A+B"的简单堆砌

工作流程：
  Phase 1: 分析 idea DNA
    - 提取 user_idea 的核心问题定义
    - 提取 user_idea 的核心假设 (为什么会导致这个问题)
    - 提取 user_idea 的核心创新点 (我如何解决)

  Phase 2: 分析 Pattern idea DNA
    - 对每个 Pattern 的方法论做同样分析

  Phase 3: 发现融合点
    - 问题空间的补集：Pattern 解决的问题与 user_idea 的关系
    - 假设空间的交集：两者共同假设但角度不同
    - 创新点的乘积：两个创新结合能否产生1+1>2的效果

  Phase 4: 生成融合 idea
    - 基于融合点，生成新的 problem framing
    - 生成新的 assumption
    - 生成新的 novelty claim
"""

from typing import Dict, Optional

from .utils import call_llm, parse_json_from_llm


class IdeaFusionEngine:
    """在概念层进行 idea 融合，而不是技术层的简单拼接"""

    def __init__(self):
        pass

    def fuse(self, user_idea: str, pattern_id: str, pattern_info: Dict,
             previous_story: Optional[Dict] = None) -> Dict:
        """
        融合 user_idea 和 pattern_info，生成创新的新想法

        Args:
            user_idea: 用户的原始 idea 描述
            pattern_id: Pattern 的 ID
            pattern_info: Pattern 的详细信息（包含方法论、技巧等）
            previous_story: 前一轮生成的 Story（用于提取更详细的 idea DNA）

        Returns:
            {
                'fused_idea': str,  # 融合后的新 idea 简述
                'problem_fusion': str,  # 融合后的问题定义
                'assumption_fusion': str,  # 融合后的假设
                'novelty_claim': str,  # 融合后的创新点
                'fusion_rationale': str,  # 为什么这个融合有意义
                'key_innovation_points': List[str],  # 关键创新点（3-5个）
            }
        """
        print("\n" + "=" * 80)
        print("💡 Phase: Idea Fusion (Conceptual Innovation Fusion)")
        print("=" * 80)

        # Step 1: Analyze user_idea DNA
        print("\n📍 Step 1: Analyzing User Idea DNA...")
        user_dna = self._analyze_idea_dna(user_idea, previous_story)
        print(f"   ✓ Problem Space: {user_dna['problem'][:80]}...")
        print(f"   ✓ Core Assumption: {user_dna['assumption'][:80]}...")
        print(f"   ✓ Innovation Claim: {user_dna['novelty_claim'][:80]}...")

        # Step 2: Analyze Pattern idea DNA
        print("\n📍 Step 2: Analyzing Pattern Idea DNA...")
        pattern_name = pattern_info.get('name', pattern_id)
        pattern_dna = self._extract_pattern_dna(pattern_id, pattern_info, pattern_name)
        print(f"   ✓ Pattern Problem: {pattern_dna['problem'][:80]}...")
        print(f"   ✓ Pattern Assumption: {pattern_dna['assumption'][:80]}...")
        print(f"   ✓ Pattern Innovation: {pattern_dna['novelty_claim'][:80]}...")

        # Step 3: Discover fusion points
        print("\n📍 Step 3: Discovering Fusion Points...")
        fusion_analysis = self._discover_fusion_points(
            user_idea, user_dna, pattern_name, pattern_dna, pattern_info
        )

        # Step 4: Generate fused idea
        print("\n📍 Step 4: Generating Fused Idea...")
        fused_result = self._generate_fused_idea(
            user_idea, user_dna, pattern_name, pattern_dna, fusion_analysis
        )

        return fused_result

    def _analyze_idea_dna(self, idea: str, previous_story: Optional[Dict] = None) -> Dict:
        """
        提取 idea 的核心 DNA：问题定义、假设、创新点
        """
        # 如果有前一轮 story，使用其中的结构化信息
        if previous_story:
            return {
                'problem': previous_story.get('problem_framing', idea),
                'assumption': previous_story.get('gap_pattern', ''),
                'novelty_claim': previous_story.get('solution', ''),
            }

        # 否则，调用 LLM 进行深层分析
        prompt = f"""
分析以下研究 idea，提取其核心 DNA 元素：

【研究 Idea】
{idea}

请以 JSON 格式返回以下内容（要简洁，每个100字以内）：
{{
  "problem": "这个 idea 要解决的核心问题是什么？",
  "assumption": "这个 idea 基于什么假设或观察？为什么这个问题存在？",
  "novelty_claim": "这个 idea 的核心创新主张是什么？如何与现有方法不同？"
}}

只返回 JSON，不需要其他说明。
"""
        response = call_llm(prompt, temperature=0.7, max_tokens=500, timeout=120)
        result = parse_json_from_llm(response)

        if result and all(k in result for k in ['problem', 'assumption', 'novelty_claim']):
            return result
        else:
            # Fallback
            return {
                'problem': idea,
                'assumption': '在已有方法基础上寻求改进',
                'novelty_claim': '提出新的视角或技术',
            }

    def _extract_pattern_dna(self, pattern_id: str, pattern_info: Dict,
                             pattern_name: str) -> Dict:
        """从 Pattern 信息中提取核心 DNA"""
        # 尝试从结构化信息中提取
        summary = pattern_info.get('summary', {})

        # 构建 Pattern 的描述文本
        approaches = []
        if isinstance(summary, dict):
            approaches = summary.get('solution_approaches', [])[:3]

        skeleton_examples = pattern_info.get('skeleton_examples', [])
        story_guides = pattern_info.get('story', [])

        # 组合成完整描述
        pattern_description = f"{pattern_name}\n"
        if approaches:
            pattern_description += "\n".join(approaches[:2])
        elif skeleton_examples:
            pattern_description += skeleton_examples[0].get('method_story', '')

        # 调用 LLM 分析
        prompt = f"""
分析以下研究范式/模式，提取其核心 DNA：

【研究范式】
名称: {pattern_name}
描述:
{pattern_description[:500]}

请以 JSON 格式返回以下内容（每个100字以内）：
{{
  "problem": "这个范式主要解决什么类型的问题？",
  "assumption": "这个范式基于什么核心假设或洞察？",
  "novelty_claim": "这个范式相比现有方法的核心创新是什么？"
}}

只返回 JSON，不需要其他说明。
"""
        response = call_llm(prompt, temperature=0.7, max_tokens=500, timeout=120)
        result = parse_json_from_llm(response)

        if result and all(k in result for k in ['problem', 'assumption', 'novelty_claim']):
            return result
        else:
            return {
                'problem': f'在{pattern_name}的问题域中寻求改进',
                'assumption': '通过组合已有技术实现突破',
                'novelty_claim': '引入新的架构或方法论',
            }

    def _discover_fusion_points(self, user_idea: str, user_dna: Dict,
                                pattern_name: str, pattern_dna: Dict,
                                pattern_info: Dict) -> Dict:
        """
        发现两个 idea 之间可以融合的创新点

        策略：
        1. 问题空间互补：Pattern 解决的问题是否能补充用户 idea 的局限性？
        2. 假设空间交集：两者是否基于相似假设但得出不同结论？
        3. 创新点乘积：两个创新结合是否能产生新的突破？
        """
        prompt = f"""
分析两个研究 idea 之间的融合潜力。

【User Idea】
问题: {user_dna['problem']}
假设: {user_dna['assumption']}
创新点: {user_dna['novelty_claim']}

【Pattern Idea】
名称: {pattern_name}
问题: {pattern_dna['problem']}
假设: {pattern_dna['assumption']}
创新点: {pattern_dna['novelty_claim']}

请分析以下几点（每点50-100字）：

1. 【问题空间互补】Pattern 如何能补充或扩展 User Idea 的问题定义？
2. 【假设空间交集】两者是否基于相似的问题观察但采用了不同的解决角度？
3. 【创新点乘积】结合两者的创新点，能否产生1+1>2的新想法？
4. 【关键融合机制】具体如何将 Pattern 的方法融入 User Idea 中，使其成为有机整体而非机械拼接？

以 JSON 格式返回：
{{
  "problem_complement": "...",
  "assumption_intersection": "...",
  "innovation_product": "...",
  "fusion_mechanism": "..."
}}
"""

        response = call_llm(prompt, temperature=0.8, max_tokens=800, timeout=180)
        result = parse_json_from_llm(response)

        if result:
            return result
        else:
            return {
                'problem_complement': '扩展问题的应用范围',
                'assumption_intersection': '共同面临的核心挑战',
                'innovation_product': '技术层的深度融合',
                'fusion_mechanism': '通过架构创新实现融合',
            }

    def _generate_fused_idea(self, user_idea: str, user_dna: Dict,
                             pattern_name: str, pattern_dna: Dict,
                             fusion_analysis: Dict) -> Dict:
        """
        基于融合分析生成真正创新的新想法
        """
        prompt = f"""
Based on the following analysis, generate a truly innovative fused idea (NOT a simple A+B combination).

【Original User Idea】
{user_idea}

【User Idea DNA】
- Problem: {user_dna['problem']}
- Assumption: {user_dna['assumption']}
- Innovation: {user_dna['novelty_claim']}

【Pattern: {pattern_name}】
- Problem: {pattern_dna['problem']}
- Assumption: {pattern_dna['assumption']}
- Innovation: {pattern_dna['novelty_claim']}

【Fusion Analysis】
- Problem Complement: {fusion_analysis['problem_complement']}
- Assumption Intersection: {fusion_analysis['assumption_intersection']}
- Innovation Product: {fusion_analysis['innovation_product']}
- Fusion Mechanism: {fusion_analysis['fusion_mechanism']}

---

【GOOD FUSION EXAMPLES - Learn from these】

Example 1: Image Captioning + Contrastive Learning
❌ Bad Fusion: "Use contrastive learning to improve image captioning by contrasting positive/negative image-caption pairs"
✅ Good Fusion: "Reframe image captioning as a contrastive reasoning task where the model learns to distinguish between visually similar scenes through their semantic differences in natural language descriptions, creating a unified vision-language alignment space rather than treating captioning as a one-way generation task."
Why Good: Redefines the TASK NATURE from generation to contrastive reasoning, creates conceptual unity.

Example 2: Small Model Compression + Knowledge Distillation
❌ Bad Fusion: "Apply knowledge distillation to compress small models using teacher-student training"
✅ Good Fusion: "Transform model compression from a post-training reduction process into a knowledge inheritance mechanism, where the student model doesn't just mimic outputs but inherits the teacher's reasoning structure through distillation-guided architecture search, co-evolving capacity and knowledge during compression."
Why Good: Changes compression from "reduction" to "inheritance", makes distillation and compression co-evolve.

Example 3: Graph Neural Networks + Attention Mechanism
❌ Bad Fusion: "Add attention mechanism to GNN to weight neighbor aggregation"
✅ Good Fusion: "Reframe graph neural networks as dynamic attention-driven topology learners, where attention doesn't just weight fixed edges but actively discovers latent relational structures, transforming GNNs from static structure encoders into adaptive structure-content co-learners."
Why Good: Elevates attention from a "weighting tool" to a "structure discovery mechanism", unifies topology and content learning.

---

Now generate YOUR fused idea following this quality standard.

Return JSON format:
{{
  "fused_idea_title": "A concise title (within 10 words)",
  "fused_idea_description": "Brief description of the fused idea (within 150 words)",
  "problem_framing": "Reframed problem definition (within 200 words)",
  "core_assumption": "Core assumption of the fused idea (within 150 words)",
  "novelty_claim": "Innovation claim (within 150 words, highlight what's NEW compared to original ideas)",
  "key_innovation_points": ["Innovation point 1", "Innovation point 2", "Innovation point 3"],
  "why_not_straightforward_combination": "Explain why this is NOT a simple A+B combination (within 100 words)"
}}

KEY REQUIREMENTS:
1. The new idea should NOT sound like stacking two ideas together
2. Should innovate in problem redefinition, assumption shift, or perspective transformation
3. Must clearly explain why this fusion creates NEW insights
4. Show how the two ideas CO-EVOLVE rather than CO-EXIST
5. Avoid phrases like "combine X with Y" or "integrate A and B" - instead use "reframe", "transform", "unify"
"""

        response = call_llm(prompt, temperature=0.9, max_tokens=1500, timeout=180)
        result = parse_json_from_llm(response)

        if result and 'fused_idea_description' in result:
            print(f"\n   ✅ Fusion Complete:")
            print(f"      Title: {result.get('fused_idea_title', 'N/A')}")
            print(f"      Novelty Claim: {result.get('why_not_straightforward_combination', '')[:100]}...")

            # 【验证输出】中文版关键信息，用于验证融合质量
            print(f"\n   📝 [验证] 融合标题: {result.get('fused_idea_title', 'N/A')}")
            print(f"   📝 [验证] 为何非堆砌: {result.get('why_not_straightforward_combination', '')[:150]}...")

            return result
        else:
            # Fallback: Generate basic fusion result
            return {
                'fused_idea_title': f'Integrating {pattern_name} with Original Approach',
                'fused_idea_description': f'Innovative fusion through {fusion_analysis["fusion_mechanism"]}',
                'problem_framing': f'{user_dna["problem"]}, and {fusion_analysis["problem_complement"]}',
                'core_assumption': fusion_analysis['assumption_intersection'],
                'novelty_claim': fusion_analysis['innovation_product'],
                'key_innovation_points': [
                    'Multidimensional problem redefinition',
                    'Innovative intersection of assumption space',
                    'Organic fusion of technical methods',
                ],
                'why_not_straightforward_combination': 'Through conceptual idea fusion rather than technical combination',
            }

