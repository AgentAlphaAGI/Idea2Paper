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
                 new_tricks_only: Optional[List[str]] = None) -> Dict:
        """生成 Story (支持初次生成和增量修正)"""

        # 模式判断：如果有上一轮 Story 和反馈，进入【增量修正模式】
        if previous_story and review_feedback:
            print(f"\n📝 修正 Story (基于上一轮反馈 + 新注入技巧)")
            prompt = self._build_refinement_prompt(
                previous_story, review_feedback, new_tricks_only, pattern_info
            )
        else:
            # 【初次生成模式】
            print(f"\n📝 生成 Story (基于 {pattern_id})")

            # 打印调试信息
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
        response = call_llm(prompt, temperature=0.7, max_tokens=1500) # 稍微降低温度以保持稳定性

        # 解析输出
        story = self._parse_story_response(response)

        # 如果是修正模式，合并旧 Story 的未修改部分（保底策略）
        if previous_story:
            for key in ['title', 'abstract', 'problem_definition', 'method_skeleton', 'innovation_claims', 'experiments_plan']:
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

    def _build_refinement_prompt(self, previous_story: Dict,
                               review_feedback: Dict,
                               new_tricks: List[str],
                               pattern_info: Dict) -> str:
        """构建增量修正 Prompt (Editor Mode) - 强调深度方法论融合"""

        # 提取评审意见摘要
        critique_summary = ""
        main_issue = ""
        for review in review_feedback.get('reviews', []):
            critique_summary += f"- {review['reviewer']} ({review['role']}): {review['score']}分. 反馈: {review['feedback'][:250]}...\n"
            if review['role'] == 'Novelty' and review['score'] < 7.0:
                main_issue = "novelty"
            elif review['role'] == 'Methodology' and review['score'] < 7.0 and not main_issue:
                main_issue = "stability"

        # 提取新注入的技术（强调深度融合）
        tricks_instruction = ""
        if new_tricks:
            if "核心技术" in str(new_tricks) or "方法论" in str(new_tricks):
                # 针对方法论注入的特殊指令
                tricks_instruction = "【核心任务：方法论深度重构】\n"
                tricks_instruction += "评审指出当前方法存在问题，需要引入新的技术路线来解决。请参考以下注入的技术和方法论，对核心方法进行**深度改造**：\n\n"
                for trick in new_tricks:
                    tricks_instruction += f"  🔧 {trick}\n"
                tricks_instruction += "\n【重构要求】\n"
                tricks_instruction += "1. **方法论融合**：不要只是在 method_skeleton 末尾添加新步骤，而是要将新技术**深度嵌入**到现有方法的核心逻辑中。\n"
                tricks_instruction += "   - 例如：如果注入\"课程学习\"，应该是\"设计基于难度的课程学习调度器，让模型从易到难学习\"，而不是\"添加课程学习\"。\n"
                tricks_instruction += "   - 例如：如果注入\"对抗训练\"，应该是\"在优化目标中加入对抗扰动正则项，并采用混合训练策略\"，而不是\"使用对抗训练\"。\n"
                tricks_instruction += "2. **技术组合创新**：将注入的技术与现有方法结合，形成新的技术组合，产生 1+1>2 的效果。\n"
                tricks_instruction += "3. **贡献点更新**：在 innovation_claims 中明确指出新技术如何解决了评审指出的问题。\n"
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

        prompt = f"""
你是一位顶级 NLP 会议的资深论文作者，擅长将新技术深度融合到现有方法中，形成创新的技术组合。

【当前 Story 版本】
Title: {previous_story.get('title')}
Abstract: {previous_story.get('abstract')}
Problem: {previous_story.get('problem_definition')}
Method: {previous_story.get('method_skeleton')}
Claims: {json.dumps(previous_story.get('innovation_claims', []), ensure_ascii=False)}

【评审专家反馈】(请仔细阅读，保留好评部分，深度改造差评部分)
{critique_summary}

{tricks_instruction}
{specific_guidance}

【修正原则】
1. **保留精华**：评审中得分较高或未被批评的维度（如问题定义、实验计划等），请尽量保留原样。
2. **深度融合**：将新注入的技术**有机地嵌入**到 method_skeleton 的核心逻辑中，形成**统一的技术路线**，而不是逐个罗列技术。
3. **重构而非堆砌**：不要简单地在原有方法后追加新技术，而是要**改造现有步骤**，让新技术成为方法论的有机组成部分。
4. **具体描述**：避免抽象的描述，要具体说明技术如何实现、如何组合、解决什么问题。

【核心要求】：将多个新注入的技术**整合成一个连贯的方法论框架**，而不是分别描述每个技术

【输出要求】
请输出修正后的完整 Story JSON（必须严格遵循以下格式，不要省略任何字段）：

输出格式（纯JSON，不要包含其他文本）：
{{
  "title": "...",
  "abstract": "...",
  "problem_definition": "...",
  "method_skeleton": "步骤1；步骤2；步骤3（必须是字符串，用分号分隔各步骤）",
  "innovation_claims": ["贡献点1", "贡献点2", "贡献点3"],
  "experiments_plan": "..."
}}

注意：
- method_skeleton 必须是字符串类型，描述3-5个方法步骤，用分号分隔，**每个步骤要具体描述技术实现细节**
- innovation_claims 必须是字符串数组，包含3个具体的贡献点，**要突出技术组合的独特性**
- 所有字段都必须填写，不能为空
"""
        return prompt


    def _build_generation_prompt(self, pattern_info: Dict,
                                  constraints: Optional[List[str]],
                                  injected_tricks: Optional[List[str]]) -> str:
        """构建生成 Prompt"""

        # 提取 Pattern 信息
        pattern_name = pattern_info.get('name', '')
        pattern_summary = pattern_info.get('summary', '')
        skeleton_examples = pattern_info.get('skeleton_examples', [])[:2]  # 取前2个示例
        top_tricks = pattern_info.get('top_tricks', [])[:5]  # 取前5个高频技巧

        # 构建 Skeleton 示例文本
        skeleton_text = ""
        for i, sk in enumerate(skeleton_examples, 1):
            skeleton_text += f"\n示例 {i}:\n"
            skeleton_text += f"  标题: {sk.get('title', '')}\n"
            skeleton_text += f"  问题定位: {sk.get('problem_framing', '')[:100]}...\n"
            skeleton_text += f"  方法概述: {sk.get('method_story', '')[:100]}...\n"

        # 构建 Tricks 文本
        tricks_text = ""
        for trick in top_tricks:
            tricks_text += f"  - {trick.get('name', '')} (使用率 {trick.get('percentage', '')})\n"

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
你是一位顶级 NLP 会议的论文作者。请基于以下用户 Idea 和写作模板，生成一个结构化的论文 Story。

【用户 Idea】
{self.user_idea}

【写作模板】{pattern_name}
{pattern_summary}

【模板示例】
{skeleton_text}

【高频技巧】
{tricks_text}
{constraints_text}
{injection_text}
{emphasis_text}

【任务要求】
请生成以下结构化内容（JSON格式）。注意：如果提供了【必须融合的技巧】或【重要】部分，你生成的方法必须清晰体现这些要素，使其成为整个方案的核心组成部分。

1. title: 论文标题（简洁、专业、要体现关键创新点）
2. abstract: 摘要（150-200字，概括问题、方法、贡献）
3. problem_definition: 明确的问题定义（50-80字）
4. method_skeleton: 核心方法的步骤（3-5个步骤，每步用分号分隔，必须清晰体现已注入的技巧）
5. innovation_claims: 3个核心贡献点（列表格式，应包含已注入技巧带来的新创新）
6. experiments_plan: 实验设计（50-80字）

输出格式（纯JSON，不要包含其他文本）：
{{
  "title": "...",
  "abstract": "...",
  "problem_definition": "...",
  "method_skeleton": "...",
  "innovation_claims": ["...", "...", "..."],
  "experiments_plan": "..."
}}
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

        val = extract_str('problem_definition')
        if val:
            story['problem_definition'] = val
            print(f"      ✓ 提取 problem_definition: {val[:60]}...")

        val = extract_str('method_skeleton')
        if val:
            story['method_skeleton'] = val
            print(f"      ✓ 提取 method_skeleton: {val[:60]}...")

        val = extract_str('experiments_plan')
        if val:
            story['experiments_plan'] = val
            print(f"      ✓ 提取 experiments_plan: {val[:60]}...")

        val = extract_list('innovation_claims')
        if val:
            story['innovation_claims'] = val
            print(f"      ✓ 提取 innovation_claims: {len(val)} 项")

        return story

    def _default_story(self) -> Dict:
        """默认 Story 结构"""
        return {
            'title': f"基于 {self.user_idea[:20]} 的创新方法",
            'abstract': f"我们提出了一个新的框架来解决 {self.user_idea}。实验表明有效性。",
            'problem_definition': f"现有方法在 {self.user_idea} 上存在性能不足的问题。",
            'method_skeleton': "第一步：构建基础框架；第二步：设计核心算法；第三步：优化性能。",
            'innovation_claims': [
                "提出新的方法框架",
                "设计高效的算法",
                "在多个数据集上验证有效性"
            ],
            'experiments_plan': "在标准数据集上对比基线方法，验证各组件的有效性。"
        }

    def _print_story(self, story: Dict):
        """打印生成的 Story"""
        print("\n   📄 生成的 Story:")
        print(f"   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"   标题: {story.get('title', '')}")
        print(f"   摘要: {story.get('abstract', '')}")
        print(f"   问题: {story.get('problem_definition', '')}")
        print(f"   方法: {story.get('method_skeleton', '')}")
        print(f"   贡献:")
        for claim in story.get('innovation_claims', []):
            print(f"     - {claim}")
        print(f"   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

