import json
import re
from typing import Dict, List, Optional

from .utils import call_llm, parse_json_from_llm


class StoryGenerator:
    """Story 生成器: 基于 Idea + Pattern 生成结构化 Story"""

    def __init__(self, user_idea: str):
        self.user_idea = user_idea

    def generate(self, pattern_id: str, pattern_info: Dict,
                 constraints: Optional[List[str]] = None,
                 injected_tricks: Optional[List[str]] = None,
                 previous_story: Optional[Dict] = None,
                 review_feedback: Optional[Dict] = None,
                 new_tricks_only: Optional[List[str]] = None,
                 fused_idea: Optional[Dict] = None,
                 reflection_guidance: Optional[Dict] = None) -> Dict:
        """生成 Story (支持初次生成和增量修正，支持 idea fusion 和 reflection 指导)"""

        # 模式判断：如果有上一轮 Story 和反馈，进入【增量修正模式】
        if previous_story and review_feedback:
            print(f"\n📝 修正 Story (基于上一轮反馈 + 新注入技巧)")

            # 【新增】打印关键指导信息（用于验证融合是否生效）
            if fused_idea:
                print(f"   💡 融合概念: {fused_idea.get('fused_idea_title', 'N/A')}")
                print(f"   📝 新颖性声明: {fused_idea.get('novelty_claim', 'N/A')[:80]}...")
            if reflection_guidance:
                print(f"   🎯 反思建议: 标题策略={bool(reflection_guidance.get('title_evolution'))}, 方法策略={bool(reflection_guidance.get('method_evolution'))}")

            prompt = self._build_refinement_prompt(
                previous_story, review_feedback, new_tricks_only, pattern_info, fused_idea, reflection_guidance
            )
        else:
            # 【初次生成模式】
            print(f"\n📝 生成 Story (基于 {pattern_id})")

            # 打印调试信息（完整内容，无截断）
            if injected_tricks:
                print(f"   🔧 已注入 {len(injected_tricks)} 个 Trick:")
                for trick in injected_tricks:
                    print(f"      - {trick}")
            else:
                print(f"   🔧 本轮无 Trick 注入（首次生成）")

            if constraints:
                print(f"   📌 应用 {len(constraints)} 个约束条件:")
                for constraint in constraints:
                    print(f"      - {constraint}")

            # 构建 Prompt
            prompt = self._build_generation_prompt(
                pattern_info, constraints, injected_tricks
            )

        # 调用 LLM 生成
        print("   ⏳ 调用 LLM 生成...")
        # 使用更长的超时时间（180 秒）以应对长 Prompt 和网络延迟
        response = call_llm(prompt, temperature=0.7, max_tokens=1500, timeout=180)

        # 解析输出
        story = self._parse_story_response(response)

        # 如果是修正模式，合并旧 Story 的未修改部分（保底策略）
        if previous_story:
            for key in ['title', 'abstract', 'problem_framing', 'gap_pattern', 'solution', 'method_skeleton', 'innovation_claims', 'experiments_plan']:
                if not story.get(key) or story.get(key) == "":
                    story[key] = previous_story.get(key)
                    print(f"   ⚠️  字段 '{key}' 为空，已从上一版本恢复")

            # 特殊处理 method_skeleton：如果是字典，尝试转换为字符串
            if isinstance(story.get('method_skeleton'), dict):
                method_dict = story['method_skeleton']
                story['method_skeleton'] = '；'.join(str(v) for v in method_dict.values() if v)
                print(f"   ⚠️  method_skeleton 是字典，已转换为字符串")

            # 特殊处理 innovation_claims：如果不是列表或内容异常，恢复
            if not isinstance(story.get('innovation_claims'), list) or \
               len(story.get('innovation_claims', [])) == 0 or \
               any(claim in ['novelty', 'specific_contributions', 'innovative_points']
                   for claim in story.get('innovation_claims', [])):
                story['innovation_claims'] = previous_story.get('innovation_claims', [])
                print(f"   ⚠️  innovation_claims 异常，已从上一版本恢复")

        # 打印生成的 Story
        self._print_story(story)

        return story

    def _build_reflection_fusion_guidance(self, fused_idea: Dict, reflection_result: Optional[Dict] = None) -> str:
        """构建反思融合指导 - 确保有机融合而非生硬拼接"""

        if not reflection_result:
            return ""

        guidance = "\n【💡 REFLECTION FUSION GUIDANCE (确保有机融合)】\n"
        guidance += f"Fusion Quality Score: {reflection_result.get('fusion_quality_score', 0):.2f}/1.0\n"
        guidance += f"Fusion Type: {'Organic Fusion' if reflection_result.get('is_organic') else 'Needs Optimization'}\n\n"

        if reflection_result.get('coherence_analysis'):
            guidance += f"【Coherence Analysis】\n{reflection_result['coherence_analysis']}\n\n"

        if reflection_result.get('suggested_title'):
            guidance += f"【Suggested Title Evolution】\n{reflection_result['suggested_title']}\n\n"

        if reflection_result.get('suggested_method_evolution'):
            guidance += f"【Suggested Method Evolution】\n{reflection_result['suggested_method_evolution']}\n\n"

        guidance += "【Key Fusion Requirement】\n"
        guidance += "DO NOT simply stack the new pattern on top of the old story.\n"
        guidance += "Instead, perform **conceptual-level fusion** where:\n"
        guidance += "1. The new pattern's core concepts are integrated with old story's assumptions\n"
        guidance += "2. The method_skeleton is **restructured** to make the new technique an organic component\n"
        guidance += "3. The innovation_claims highlight the **NEW PERSPECTIVE** created by this fusion\n"

        return guidance

    def _build_refinement_prompt(self, previous_story: Dict,
                               review_feedback: Dict,
                               new_tricks: List[str],
                               pattern_info: Dict,
                               fused_idea: Optional[Dict] = None,
                               reflection_guidance: Optional[Dict] = None) -> str:
        """构建增量修正 Prompt (Editor Mode) - 强调深度方法论融合 + 概念级创新融合 + Reflection指导"""

        # 提取评审意见摘要
        critique_summary = ""
        main_issue = ""
        for review in review_feedback.get('reviews', []):
            critique_summary += f"- {review['reviewer']} ({review['role']}): {review['score']}分. 反馈: {review['feedback'][:250]}...\n"
            if review['role'] == 'Novelty' and review['score'] < 7.0:
                main_issue = "novelty"
            elif review['role'] == 'Methodology' and review['score'] < 7.0 and not main_issue:
                main_issue = "stability"

        # 提取 Pattern 信息 (用于提供技术方案和包装策略)
        summary = pattern_info.get('summary', {})
        if isinstance(summary, dict):
            solution_approaches = summary.get('solution_approaches', [])[:3]
            story_guides = summary.get('story', [])[:2]
        else:
            solution_approaches = []
            story_guides = []

        # 构建 Pattern 参考信息
        pattern_reference = ""
        if solution_approaches or story_guides:
            pattern_reference = "\n【Pattern Reference】(Use these to enhance technical depth and narrative)\n"
            if solution_approaches:
                pattern_reference += "\n💡 Solution Approaches (add concrete technical steps if needed):\n"
                for i, sol in enumerate(solution_approaches, 1):
                    pattern_reference += f"  {i}. {sol}\n"
            if story_guides:
                pattern_reference += "\n📖 Story Packaging Strategy (use 'Reframe/Transform' pattern for claims):\n"
                for i, guide in enumerate(story_guides, 1):
                    pattern_reference += f"  {i}. {guide}\n"
            pattern_reference += "\nRemember: These are TOOLS to implement the User Idea, not the main focus.\n"

        # 【新增】融合后的 idea 指导（概念层创新）
        fused_idea_guidance = ""
        if fused_idea and fused_idea.get('fused_idea_description'):
            fused_idea_guidance = "\n【💡 CRITICAL: Conceptual Innovation from Idea Fusion】\n"
            fused_idea_guidance += f"Title: {fused_idea.get('fused_idea_title', '')}\n"
            fused_idea_guidance += f"Description: {fused_idea.get('fused_idea_description', '')}\n\n"
            fused_idea_guidance += f"New Problem Framing: {fused_idea.get('problem_framing', '')}\n\n"
            fused_idea_guidance += f"New Assumption: {fused_idea.get('core_assumption', '')}\n\n"
            fused_idea_guidance += f"New Novelty Claim: {fused_idea.get('novelty_claim', '')}\n\n"
            fused_idea_guidance += f"Why This is NOT Simple Combination:\n{fused_idea.get('why_not_straightforward_combination', '')}\n\n"
            fused_idea_guidance += "Key Innovation Points:\n"
            for i, point in enumerate(fused_idea.get('key_innovation_points', []), 1):
                fused_idea_guidance += f"  {i}. {point}\n"
            fused_idea_guidance += "\n⚠️ CRITICAL: This fused idea represents a conceptual-level innovation, not just a technical combination.\n"
            fused_idea_guidance += "Your story refinement should reflect THIS NEW CONCEPTUAL INNOVATION in problem_framing, gap_pattern, and innovation_claims.\n"
            fused_idea_guidance += "This is the KEY to achieving higher novelty scores - moving from \"combination of techniques\" to \"new perspective on the problem\".\n"

        # 提取新注入的技术（强调深度融合）
        tricks_instruction = ""
        if new_tricks:
            if "创新融合" in str(new_tricks) or "核心技术" in str(new_tricks) or "方法论" in str(new_tricks):
                # 针对创新融合和方法论注入的特殊指令
                tricks_instruction = "【核心任务：概念级创新融合】\n"
                tricks_instruction += "评审指出当前方法的创新性不足。这轮修正的关键是**概念层的思想创新**，而不仅仅是技术堆砌。\n"
                tricks_instruction += "参考以下融合后的创新想法，对整体论文的 problem framing、gap pattern 和 innovation claims 进行**根本性重构**：\n\n"
                for trick in new_tricks:
                    tricks_instruction += f"  🔧 {trick}\n"
                tricks_instruction += "\n【融合重构要求】\n"
                tricks_instruction += "1. **概念重新定义**：不要保持旧的问题定义，而是基于融合后的新观点完全重新框架化问题。\n"
                tricks_instruction += "2. **假设转变**：体现融合后的核心假设如何与原有方法的假设不同。\n"
                tricks_instruction += "3. **创新点升级**：在 innovation_claims 中明确指出这是\"概念级的创新\"而非\"技术级的组合\"。\n"
            else:
                tricks_instruction = "【本次修正核心任务】\n请将以下新技巧深度融合到 Method 和 Contribution 中，解决上述评审指出的问题：\n"
                for trick in new_tricks:
                    tricks_instruction += f"  👉 注入: {trick}\n"

        # 根据主要问题添加针对性指导
        specific_guidance = ""
        if main_issue == "novelty":
            specific_guidance = "\n【针对创新性问题的特别指导】\n"
            specific_guidance += "当前方法被评审认为\"创新性不足\"或\"技术组合常见\"。你需要：\n"
            specific_guidance += "1. 在 method_skeleton 中，突出新注入技术的**独特应用方式**，形成与众不同的技术路线。\n"
            specific_guidance += "2. 在 innovation_claims 中，明确指出你的技术组合与现有工作的**本质区别**。\n"
            specific_guidance += "3. 避免使用\"提升性能\"、\"增强效果\"等泛泛而谈的描述，要具体说明技术创新点。\n"
        elif main_issue == "stability":
            specific_guidance = "\n【针对稳定性问题的特别指导】\n"
            specific_guidance += "当前方法被评审认为\"技术细节不足\"或\"稳定性有待验证\"。你需要：\n"
            specific_guidance += "1. 在 method_skeleton 中，添加具体的稳定性保障机制（如正则化、混合策略、鲁棒性设计）。\n"
            specific_guidance += "2. 强调方法的可靠性和实用性，而不仅仅是理论创新。\n"

        # 【新增】Reflection 指导（来自融合质量评估）
        reflection_guidance_text = ""
        if reflection_guidance:
            reflection_guidance_text = "\n【🎯 CRITICAL: Reflection Guidance from Fusion Quality Assessment】\n"
            reflection_guidance_text += "The Story Reflector has analyzed the fusion quality and provided the following strategic guidance:\n\n"

            title_evolution = reflection_guidance.get('title_evolution', '')
            method_evolution = reflection_guidance.get('method_evolution', '')
            narrative_strategy = reflection_guidance.get('narrative_strategy', '')

            if title_evolution:
                reflection_guidance_text += f"📝 Title Evolution Strategy:\n   {title_evolution}\n\n"
            if method_evolution:
                reflection_guidance_text += f"🔧 Method Evolution Strategy:\n   {method_evolution}\n\n"
            if narrative_strategy:
                reflection_guidance_text += f"📖 Narrative Strategy:\n   {narrative_strategy}\n\n"

            reflection_guidance_text += "⚠️ IMPORTANT: These guidance points are based on analyzing the fusion between your current Story and the new Pattern.\n"
            reflection_guidance_text += "Follow these strategies to ensure the fusion creates genuine conceptual innovation, not just technical stacking.\n"

        # 提取用户Idea核心概念
        user_idea_reminder = f"\n【User's Original Idea - THE PROTAGONIST】\n\"{self.user_idea}\"\n\nCore Concepts to Preserve: [Identify 2-4 key concepts from the idea above]\n"

        prompt = f"""
You are a senior paper author at a top AI conference, skilled in deeply integrating new techniques into existing methods to form innovative technical combinations.

{user_idea_reminder}

⚠️ 【CRITICAL: User Idea Protection Rules During Refinement】
When refining, ALWAYS remember:
1. The user's core idea is "{self.user_idea}" - this is the PROTAGONIST of your story
2. Technical approaches (e.g., RL, neural networks) are IMPLEMENTATION MEANS, not the main characters
3. Title and abstract MUST always highlight the User Idea's core concepts
4. Even when injecting new techniques, describe them from the perspective of "how these techniques implement the User Idea"

【Current Story Version】
Title: {previous_story.get('title')}
Abstract: {previous_story.get('abstract')}
Problem Framing: {previous_story.get('problem_framing', 'N/A')}
Gap Pattern: {previous_story.get('gap_pattern', 'N/A')}
Solution: {previous_story.get('solution', 'N/A')}
Method: {previous_story.get('method_skeleton')}
Claims: {json.dumps(previous_story.get('innovation_claims', []), ensure_ascii=False)}

【Review Feedback】(Read carefully, preserve well-received parts, deeply revise criticized parts)
{critique_summary}

{fused_idea_guidance}

{reflection_guidance_text}

{tricks_instruction}
{specific_guidance}
{pattern_reference}

⚠️ 【HOW TO USE Fused Idea Guidance】
If you received 【Conceptual Innovation from Idea Fusion】 above, this is THE MOST IMPORTANT guidance:
- **Title & Abstract**: Must reflect the fused conceptual innovation, not just list techniques
- **Problem Framing**: Adopt the NEW problem perspective from the fused idea
- **Gap Pattern**: Explain why existing methods lack this conceptual unity
- **Innovation Claims**: Frame as "transforming/reframing X from Y to Z", NOT "combining A with B"
- **Method**: Show how techniques CO-EVOLVE to realize the fused concept, not just CO-EXIST

⚠️ 【HOW TO USE Pattern Information】
If the feedback mentions lack of technical depth or innovation, refer back to:
- **【Solution Approaches】from Pattern**: Use these to add concrete technical steps to method_skeleton (always frame as "means to realize [Core Concept]")
- **【Story Packaging Strategy】from Pattern**: Use the "Reframe/Transform" narrative pattern to strengthen problem_framing, gap_pattern, and innovation_claims

【Refinement Principles】
1. **Maintain User Idea Core**: No matter how you modify, ensure title, abstract, and claims all center on the user's core Idea.
2. **Preserve Excellence**: Keep dimensions that scored high or weren't criticized (e.g., problem definition, experiment plan) as-is.
3. **Deep Integration**: Organically embed newly injected techniques into method_skeleton's core logic as "means to implement User Idea".
4. **Restructure, Don't Stack**: Don't simply append new techniques to existing methods; **transform existing steps** to make new techniques organic components of the methodology.
5. **Concrete Description**: Avoid abstract descriptions; specifically explain how techniques implement, combine, and solve problems.

【Core Requirement】: Integrate multiple newly injected techniques into **a coherent methodological framework** that serves the user's core Idea.

【Output Requirements】
Please output the refined complete Story JSON (must strictly follow the format below, do not omit any fields):

Output Format (pure JSON, no other text):
{{
  "title": "...",
  "abstract": "...",
  "problem_framing": "...",
  "gap_pattern": "...",
  "solution": "...",
  "method_skeleton": "Step 1; Step 2; Step 3 (must be a string, separated by semicolons)",
  "innovation_claims": ["Claim 1", "Claim 2", "Claim 3"],
  "experiments_plan": "..."
}}

Notes:
- title MUST highlight User Idea's core concepts (e.g., "Self-Evolution of Agents via Reflection and Memory")
- abstract: Start with the User Idea vision, then describe how you realize it with technical solutions
- problem_framing: Use "Reframe [Core Concept] from X to Y" pattern to package the problem (100-150 words)
- gap_pattern: Show why current methods fail for the Core Concepts (100-150 words)
- solution: Describe the overall methodology with narrative packaging from 【Story Packaging Strategy】, emphasizing how solution approaches serve the Core Concepts (150-200 words, more descriptive)
- method_skeleton: Technical architecture with 3-5 concrete implementation steps from 【Solution Approaches】, separated by semicolons (focused on technical structure)
- innovation_claims must be a string array with 3 contribution points, **each using "Transform/Reframe" pattern**: "We transform [Core Concept] by [technical solution], achieving [benefit]"
  * Example: "Transform agent self-evolution from a passive learning process to an active reflection-driven paradigm by integrating memory mechanisms with iterative self-assessment, enabling autonomous improvement without human intervention"
  * Bad example: "Propose a novel state-space model for RL" (focuses on technique, not Core Concept)
- All fields must be filled, no empty values allowed

⚠️ LANGUAGE REQUIREMENT: Output ENTIRELY IN ENGLISH. No Chinese characters allowed.
"""
        return prompt


    def _build_generation_prompt(self, pattern_info: Dict,
                                  constraints: Optional[List[str]],
                                  injected_tricks: Optional[List[str]]) -> str:
        """构建生成 Prompt (适配新的 Pattern 结构)"""

        # 提取 Pattern 信息
        pattern_name = pattern_info.get('name', '')
        pattern_size = pattern_info.get('size', 0)

        # 新结构: 从 summary 字段提取
        summary = pattern_info.get('summary', {})
        if isinstance(summary, dict):
            representative_ideas = summary.get('representative_ideas', [])[:3]  # 取前3个
            common_problems = summary.get('common_problems', [])[:3]
            solution_approaches = summary.get('solution_approaches', [])[:3]
            story_guides = summary.get('story', [])[:2]
        else:
            # Fallback: 旧结构兼容
            representative_ideas = []
            common_problems = []
            solution_approaches = []
            story_guides = []

        # 兼容旧结构 (如果存在 skeleton_examples)
        skeleton_examples = pattern_info.get('skeleton_examples', [])[:2]

        # 构建代表性想法文本
        ideas_text = ""
        if representative_ideas:
            ideas_text = "\n【代表性研究想法】\n"
            for i, idea in enumerate(representative_ideas, 1):
                ideas_text += f"{i}. {idea}\n"

        # 构建常见问题文本
        problems_text = ""
        if common_problems:
            problems_text = "\n【该模式下常见的研究问题】\n"
            for i, problem in enumerate(common_problems, 1):
                problems_text += f"{i}. {problem}\n"

        # 构建解决方案文本 (核心参考)
        solutions_text = ""
        if solution_approaches:
            solutions_text = "\n【Solution Approaches】(Extract and adapt these technical solutions for your method)\n"
            solutions_text += "⚠️ CRITICAL: These solutions are the TECHNICAL MEANS to implement the User Idea.\n"
            solutions_text += "Use them to construct your method_skeleton, but always frame them as \"how they serve the user's core concepts\".\n\n"
            for i, solution in enumerate(solution_approaches, 1):
                solutions_text += f"{i}. {solution}\n"

        # 构建写作策略文本
        story_text = ""
        if story_guides:
            story_text = "\n【Story Packaging Strategy】(Learn how to PACKAGE the solutions into your narrative)\n"
            story_text += "⚠️ CRITICAL: These show how to REFRAME technical solutions as TRANSFORMATIVE INSIGHTS.\n"
            story_text += "Use these patterns to write problem_framing, gap_pattern, and claims, always connecting back to User Idea.\n\n"
            for i, guide in enumerate(story_guides, 1):
                story_text += f"{i}. {guide}\n"

        # 【核心改进】从 skeleton_examples 中提取完整的包装策略示例
        skeleton_text = ""
        if skeleton_examples:
            skeleton_text = "\n【参考论文的包装策略】(如何讲故事)\n"
            for i, sk in enumerate(skeleton_examples, 1):
                skeleton_text += f"\n【示例 {i}】{sk.get('title', '')}\n"

                # 提取 problem_framing (如何包装问题)
                if sk.get('problem_framing'):
                    skeleton_text += f" 问题包装策略: {sk.get('problem_framing', '')[:200]}...\n"

                # 提取 gap_pattern (如何批评现有方法)
                if sk.get('gap_pattern'):
                    skeleton_text += f" Gap呈现策略: {sk.get('gap_pattern', '')[:200]}...\n"

                # 提取 method_story (如何叙述方法)
                if sk.get('method_story'):
                    skeleton_text += f" 方法叙述策略: {sk.get('method_story', '')[:200]}...\n"

        # 构建约束文本
        constraints_text = ""
        if constraints:
            constraints_text = "\n【约束条件】\n"
            for constraint in constraints:
                constraints_text += f"  - {constraint}\n"

        # 构建注入 Trick 文本
        injection_text = ""
        if injected_tricks:
            injection_text = "\n【必须融合的技巧】\n"
            for trick in injected_tricks:
                injection_text += f"  - {trick}\n"
            injection_text += "\n注意: 必须将这些技巧自然地融合到方法中，不是简单拼接。\n"

        # 构建注入提示（针对 Novelty 问题强化重构引导）
        emphasis_text = ""
        if injected_tricks:
            if "novelty" in str(injected_tricks).lower() or len(injected_tricks) > 3:
                emphasis_text = "\n⚠️  【极重要：技术重构指令】\n"
                emphasis_text += "当前方案被评审指出“创新性不足”。你必须利用下列注入的技巧对核心方法进行**颠覆性重构**：\n"
                emphasis_text += "1. 不要只是在原有框架上修补，要将这些技巧作为方法论的第一优先级。\n"
                emphasis_text += "2. 在 method_skeleton 中，前两个步骤必须直接体现这些新技巧的应用。\n"
                emphasis_text += "3. 必须在 innovation_claims 中明确指出这些技巧如何解决了原有“平庸组合”的问题。\n"
            else:
                emphasis_text = "\n⚠️  【重要】请务必在方法中充分融合下列技巧，使其成为核心内容，而非简单堆砌：\n"

            for i, trick in enumerate(injected_tricks, 1):
                emphasis_text += f"   {i}. {trick}\n"

        prompt = f"""
You are a senior paper author at a top AI conference. Generate a structured paper story based on the user's idea and writing template.

【STEP 1: Extract Core Concepts from User Idea】
User Idea: "{self.user_idea}"

Before writing anything, identify the CORE ENTITIES in the user idea (e.g., "Agent", "Reflection", "Memory", "Self-Evolution").
These are the TRUE subjects of your paper. Write them down:
Core Concepts: [Extract 2-4 key concepts from the user idea above]

⚠️ 【CRITICAL: User Idea Protection Rules】
1. **The Core Concepts above are the PROTAGONISTS** of your story. They must appear in the title, abstract, problem_framing, gap_pattern, and all innovation claims.
2. **Pattern's techniques are TOOLS, NOT the hero**: The writing template below provides technical approaches (e.g., RL, neural models) as "means to implement the user's core concepts". They should NEVER become the main focus.
3. **Title & Abstract must highlight Core Concepts**: Technical terms can only appear as modifiers (e.g., "Self-Evolution of Agents via X" ✅, "X-based Framework" ❌).
4. **Method serves Core Concepts**: method_skeleton should describe "how to use Pattern's techniques to realize the user's core concepts", not list technique names.

【Writing Template】{pattern_name} (contains {pattern_size} papers)
{ideas_text}
{problems_text}
{solutions_text}
{story_text}
{skeleton_text}
{constraints_text}
{injection_text}
{emphasis_text}

【Task Requirements】
Generate structured content (JSON format) that tells a compelling story about the User Idea.

【STEP 2: How to Use the Writing Template】

**Part 1: Extract Technical Solutions from 【Solution Approaches】**
- These are concrete technical methods you will use to implement the User Idea
- Extract and adapt them into your method_skeleton
- BUT: Always describe them as "means to realize [Core Concept]", not as standalone techniques

**Part 2: Learn Packaging Strategy from 【Story Packaging Strategy】**
- These show the "Reframe X as Y" narrative pattern
- Use this pattern to write:
  * problem_framing: "Reframe [Core Concept] from [old view] to [new transformative view]"
  * gap_pattern: "Current methods for [Core Concept] fail because..."
  * innovation_claims: "Our approach transforms [Core Concept] by..."
- The pattern demonstrates how to position technical solutions as TRANSFORMATIVE INSIGHTS, not just methods

**Part 3: Reference 【参考论文的包装策略】 for storytelling examples**
- See how real papers package their problem, gap, and method narratives
- Learn the FLOW: problem → gap → solution → transformation

Key Guidelines:
- **Core Principle**: User Idea is the protagonist; Pattern's techniques are tools
- **Method Construction**: Use 【Solution Approaches】 to build concrete steps, framed as "implementing [Core Concept]"
- **Story Packaging**: Use 【Story Packaging Strategy】 to write problem_framing, gap_pattern, claims with "Reframe/Transform" narratives
- **Integration**: If 【必须融合的技巧】 is provided, integrate them as additional means to realize Core Concepts
- **Avoid Technique Focus**: Never let technical terms overshadow the Core Concepts

Output Fields (ALL IN ENGLISH):
1. title: Paper title (MUST highlight User Idea's core concepts; technique names only as modifiers)
2. abstract: Abstract (150-200 words, User Idea as main thread, describe both the VISION and the SOLUTION)
3. problem_framing: How to introduce the problem (100-150 words, use "Reframe/Transform" pattern, center on core concepts)
4. gap_pattern: How to critique existing methods (100-150 words, show why current approaches fail for core concepts, use "Reframe" pattern)
5. solution: Overall methodology description (150-200 words, narrative style using 【Story Packaging Strategy】, emphasize how solutions serve Core Concepts, more descriptive than method_skeleton)
6. method_skeleton: Technical architecture (3-5 implementation steps separated by semicolons, extract from 【Solution Approaches】, focused on concrete technical structure)
7. innovation_claims: 3 core contribution points (list format, use "Transform/Reframe" pattern from 【Story Packaging Strategy】, each claim MUST mention core concepts AND show transformation)
8. experiments_plan: Experiment design (50-80 words)

Output Format (pure JSON, no other text):
{{
  "title": "...",
  "abstract": "...",
  "problem_framing": "...",
  "gap_pattern": "...",
  "solution": "...",
  "method_skeleton": "Step 1; Step 2; Step 3...",
  "innovation_claims": ["Contribution 1", "Contribution 2", "Contribution 3"],
  "experiments_plan": "..."
}}

Critical Notes on Each Field:
- title: MUST highlight Core Concepts (e.g., "Self-Evolution of Agents via Reflection and Memory")
- abstract: Start with User Idea vision, then describe the technical realization
- problem_framing: Use "Reframe [Core Concept] from X to Y" pattern learned from 【Story Packaging Strategy】
- gap_pattern: Critique existing methods for Core Concepts, use "Reframe" pattern
- solution: **NEW FIELD** - Describe the overall solution methodology with narrative packaging (150-200 words, more descriptive, emphasizes HOW solution approaches serve Core Concepts)
- method_skeleton: Technical architecture with concrete implementation steps from 【Solution Approaches】 (focused on structure, 3-5 steps)
- innovation_claims: **CRITICAL** - Must use "Transform/Reframe" pattern:
  * Good example: "Transform agent self-evolution from passive RL training to an active reflection-memory paradigm by integrating episodic memory with self-assessment mechanisms, enabling autonomous improvement"
  * Bad example: "Propose a novel state-space model architecture" (technique-focused, not idea-focused)
  * Each claim MUST: mention Core Concept + describe transformation + specify technical means + show benefit

⚠️ LANGUAGE REQUIREMENT: Output ENTIRELY IN ENGLISH. No Chinese characters allowed.
"""
        return prompt

    def _parse_story_response(self, response: str) -> Dict:
        """解析 LLM 输出的 Story"""
        # 使用通用工具尝试解析
        story = parse_json_from_llm(response)

        if story:
            print(f"   ✅ JSON 解析成功")
            return story

        print(f"⚠️  无法找到 JSON 结构，尝试 Fallback 解析")
        return self._fallback_parse_story(response)

    def _fallback_parse_story(self, text: str) -> Dict:
        """Fallback: 使用正则提取 Story 字段 (更加健壮)"""
        story = self._default_story()

        # 辅助函数：提取字符串值 (处理复杂情况)
        def extract_str(key):
            # 更加健壮的正则：允许换行、特殊字符、嵌套引号
            # 匹配模式: "key": "value..." 其中 value 可以跨多行，直到遇到未转义的引号后跟逗号或}
            pattern = r'"' + re.escape(key) + r'"\s*:\s*"((?:[^"\\]|\\["\\/bfnrt]|\\u[0-9a-fA-F]{4})*)"'
            match = re.search(pattern, text, re.DOTALL)
            if match:
                val = match.group(1)
                # 处理转义字符
                val = val.replace('\\"', '"')
                val = val.replace('\\n', '\n')
                val = val.replace('\\r', '\r')
                val = val.replace('\\t', '\t')
                val = val.replace('\\\\', '\\')
                return val

            # 尝试另一种提取方式: 寻找 key 之后的首个引号，然后提取到最后一个合理的引号
            alt_pattern = r'"' + re.escape(key) + r'"\s*:\s*"([^"]*(?:\\.[^"]*)*)"'
            match = re.search(alt_pattern, text, re.DOTALL)
            if match:
                val = match.group(1)
                val = val.replace('\\"', '"')
                val = val.replace('\\n', '\n')
                return val

            return None

        # 辅助函数：提取列表
        def extract_list(key):
            pattern = r'"' + re.escape(key) + r'"\s*:\s*\[(.*?)\]'
            match = re.search(pattern, text, re.DOTALL)
            if match:
                content = match.group(1)
                items = []
                # 更加精确地提取列表项
                for m in re.finditer(r'"((?:[^"\\]|\\["\\/bfnrt]|\\u[0-9a-fA-F]{4})*)"', content):
                    item = m.group(1)
                    item = item.replace('\\"', '"')
                    item = item.replace('\\n', '\n')
                    items.append(item)
                return items if items else None
            return None

        # 打印调试信息
        print(f"   📋 使用 Fallback 解析，原始长度: {len(text)} 字符")

        # 尝试提取各字段
        val = extract_str('title')
        if val:
            story['title'] = val
            print(f"      ✓ 提取 title: {val[:60]}...")

        val = extract_str('abstract')
        if val:
            story['abstract'] = val
            print(f"      ✓ 提取 abstract: {val[:60]}...")

        val = extract_str('solution')
        if val:
            story['solution'] = val
            print(f"      ✓ 提取 solution: {val[:60]}...")

        val = extract_str('method_skeleton')
        if val:
            story['method_skeleton'] = val
            print(f"      ✓ 提取 method_skeleton: {val[:60]}...")

        val = extract_str('experiments_plan')
        if val:
            story['experiments_plan'] = val
            print(f"      ✓ 提取 experiments_plan: {val[:60]}...")

        # 新增: 提取包装策略字段
        val = extract_str('problem_framing')
        if val:
            story['problem_framing'] = val
            print(f"      ✓ 提取 problem_framing: {val[:60]}...")

        val = extract_str('gap_pattern')
        if val:
            story['gap_pattern'] = val
            print(f"      ✓ 提取 gap_pattern: {val[:60]}...")

        val = extract_list('innovation_claims')
        if val:
            story['innovation_claims'] = val
            print(f"      ✓ 提取 innovation_claims: {len(val)} 项")

        return story

    def _default_story(self) -> Dict:
        """默认 Story 结构 (全英文输出)"""
        return {
            'title': f"Innovative Approach for {self.user_idea[:50]}",
            'abstract': f"We propose a novel framework to address {self.user_idea}. Experiments demonstrate its effectiveness.",
            'problem_framing': f"How to appropriately introduce the core problem of {self.user_idea} with proper motivation and context.",
            'gap_pattern': "Existing methods have specific limitations and research gaps that need to be addressed.",
            'solution': f"We develop a comprehensive solution approach that addresses {self.user_idea} through systematic methodology and innovative techniques.",
            'method_skeleton': "Step 1: Build fundamental framework; Step 2: Design core algorithm; Step 3: Optimize performance.",
            'innovation_claims': [
                "Propose a novel methodological framework",
                "Design efficient algorithm",
                "Validate effectiveness on multiple datasets"
            ],
            'experiments_plan': "Compare with baseline methods on standard datasets and validate effectiveness of each component."
        }

    def _translate_story_to_chinese(self, story: Dict) -> Dict:
        """将英文Story翻译为中文（用于调试展示）"""
        prompt = f"""
Translate the following research paper story from English to Chinese. Keep the translation natural and accurate.

Title: {story.get('title', '')}
Abstract: {story.get('abstract', '')}
Problem Framing: {story.get('problem_framing', '')}
Gap Pattern: {story.get('gap_pattern', '')}
Solution: {story.get('solution', '')}
Method Skeleton: {story.get('method_skeleton', '')}
Innovation Claims:
{chr(10).join(f"- {claim}" for claim in story.get('innovation_claims', []))}
Experiments Plan: {story.get('experiments_plan', '')}

Output ONLY a JSON format (no other text):
{{
  "title": "...",
  "abstract": "...",
  "problem_framing": "...",
  "gap_pattern": "...",
  "solution": "...",
  "method_skeleton": "...",
  "innovation_claims": ["...", "...", "..."],
  "experiments_plan": "..."
}}
"""
        try:
            # 使用更长的超时时间（180 秒）
            response = call_llm(prompt, temperature=0.3, max_tokens=1000, timeout=180)
            cn_story = parse_json_from_llm(response)
            if cn_story and isinstance(cn_story, dict):
                return cn_story
        except Exception as e:
            print(f"   ⚠️  翻译失败: {e}")

        # Fallback: 返回原文
        return story

    def _print_story(self, story: Dict):
        """打印生成的 Story (英文版 + 中文调试版) - 显示完整内容"""
        print("\n   📄 生成的 Story (英文版):")
        print(f"   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"   Title: {story.get('title', '')}")
        print(f"   Abstract: {story.get('abstract', '')}")

        # 展示包装策略
        if story.get('problem_framing'):
            print(f"   Problem Framing: {story.get('problem_framing', '')}")
        if story.get('gap_pattern'):
            print(f"   Gap Pattern: {story.get('gap_pattern', '')}")

        # 展示 solution (新增)
        if story.get('solution'):
            print(f"   Solution: {story.get('solution', '')}")

        print(f"   Method: {story.get('method_skeleton', '')}")
        print(f"   Claims:")
        for i, claim in enumerate(story.get('innovation_claims', []), 1):
            print(f"     {i}. {claim}")
        print(f"   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        # 生成中文调试版
        print("\n   📄 中文调试版:")
        print(f"   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        cn_story = self._translate_story_to_chinese(story)
        print(f"   标题: {cn_story.get('title', '')}")
        print(f"   摘要: {cn_story.get('abstract', '')}")
        if cn_story.get('problem_framing'):
            print(f"   问题包装: {cn_story.get('problem_framing', '')}")
        if cn_story.get('gap_pattern'):
            print(f"   Gap包装: {cn_story.get('gap_pattern', '')}")
        if cn_story.get('solution'):
            print(f"   解决方案: {cn_story.get('solution', '')}")
        print(f"   方法架构: {cn_story.get('method_skeleton', '')}")
        print(f"   贡献:")
        for i, claim in enumerate(cn_story.get('innovation_claims', []), 1):
            print(f"     {i}. {claim}")
        print(f"   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

