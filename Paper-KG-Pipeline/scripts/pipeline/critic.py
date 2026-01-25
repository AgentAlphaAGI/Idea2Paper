import re
from typing import Dict, List, Tuple

from .config import PipelineConfig
from .utils import call_llm, parse_json_from_llm


class MultiAgentCritic:
    """多智能体评审团: 三个角色评审 Story"""

    def __init__(self):
        self.reviewers = [
            {'name': 'Reviewer A', 'role': 'Methodology', 'focus': '技术合理性'},
            {'name': 'Reviewer B', 'role': 'Novelty', 'focus': '创新性'},
            {'name': 'Reviewer C', 'role': 'Storyteller', 'focus': '叙事完整性'}
        ]

    def review(self, story: Dict) -> Dict:
        """评审 Story

        Returns:
            {
                'pass': bool,
                'avg_score': float,
                'reviews': [
                    {'reviewer': str, 'role': str, 'score': float, 'feedback': str},
                    ...
                ],
                'main_issue': str,  # 'novelty' | 'stability' | 'domain_distance'
                'suggestions': List[str]
            }
        """
        print("\n" + "=" * 80)
        print("🔍 Phase 3: Multi-Agent Critic (多智能体评审)")
        print("=" * 80)

        reviews = []
        scores = []

        for reviewer in self.reviewers:
            print(f"\n📝 {reviewer['name']} ({reviewer['role']}) 评审中...")

            review_result = self._single_review(story, reviewer)
            reviews.append(review_result)
            scores.append(review_result['score'])

            print(f"   评分: {review_result['score']:.1f}/10")
            print(f"   反馈: {review_result['feedback']}")

        # 计算平均分
        avg_score = sum(scores) / len(scores)
        passed = avg_score >= PipelineConfig.PASS_SCORE

        # 诊断主要问题
        main_issue, suggestions = self._diagnose_issue(reviews, scores)

        print("\n" + "-" * 80)
        print(f"📊 评审结果: 平均分 {avg_score:.2f}/10 - {'✅ PASS' if passed else '❌ FAIL'}")
        if not passed:
            print(f"🔧 主要问题: {main_issue}")
            print(f"💡 建议: {', '.join(suggestions)}")
        print("=" * 80)

        return {
            'pass': passed,
            'avg_score': avg_score,
            'reviews': reviews,
            'main_issue': main_issue,
            'suggestions': suggestions
        }

    def _single_review(self, story: Dict, reviewer: Dict) -> Dict:
        """单个评审员评审"""

        # 针对 Novelty 角色的特殊指令
        special_instructions = ""
        if reviewer['role'] == 'Novelty':
            special_instructions = """
【特别注意】
作为 Novelty 评审，你需要比较严格，不要被表面的“新颖”词汇迷惑。
1. **批判性评估组合**：仔细思考作者提出的技术是否在近两年的 NLP/CV 顶会中已经泛滥。如果是常见的“A+B”堆砌且缺乏深层理论创新，请给出低分（4-5分）。
2. **拒绝平庸**：如果 Story 只是将现有技术应用到新领域（如“用 BERT 做 X 任务”），而没有针对该领域的独特适配或理论贡献，这不叫创新。
3. **直言不讳**：如果发现是常见套路，请在反馈中明确指出“这种组合已经很常见”或“缺乏实质性创新”。
4. **高分门槛**：只有真正的范式创新、极具启发性的反直觉发现，或对现有方法的根本性改进，才能得到 8 分以上。
"""

        # 构建 Prompt
        prompt = f"""
你是顶级 NLP 会议（如 ACL/ICLR）的**严厉评审专家** {reviewer['name']}，专注于评估{reviewer['focus']}。
你的打分标准非常严格，满分 10 分。6 分以下为不及格（Reject），8 分以上为优秀（Accept）。
{special_instructions}
请评审以下论文 Story：

【标题】{story.get('title', '')}

【摘要】{story.get('abstract', '')}

【问题定义】{story.get('problem_definition', '')}

【方法概述】{story.get('method_skeleton', '')}

【贡献点】
{chr(10).join([f"  - {claim}" for claim in story.get('innovation_claims', [])])}

【实验计划】{story.get('experiments_plan', '')}

请从{reviewer['focus']}的角度进行评审。

【评审要求】
1. 请列出 3 个具体的评估维度。
2. **对每个维度进行打分（1-10分）**，并给出理由。
3. **最终总分（score）必须是各维度分数的综合评估，严禁出现细项分低但总分高的情况。**
4. 如果发现明显缺陷（如创新性不足、方法不合理），请给出低分（<6分）。

输出格式（JSON）：
{{
  "score": 6.5,
  "feedback": "1. 维度A (6.0分): 理由...\\n2. 维度B (7.0分): 理由...\\n\\n总结: ..."
}}
"""

        # 使用更长的超时时间（180 秒）以应对网络延迟
        response = call_llm(prompt, temperature=0.3, max_tokens=800, timeout=180)

        # 1. 尝试标准 JSON 解析
        result = parse_json_from_llm(response)
        if result:
            return {
                'reviewer': reviewer['name'],
                'role': reviewer['role'],
                'score': float(result.get('score', 5.0)),
                'feedback': result.get('feedback', '')
            }

        print(f"   ⚠️  JSON 解析失败，尝试 Fallback 解析")

        # 2. Fallback: 正则提取分数和反馈
        score = 5.0
        feedback = "评审意见解析失败，请查看原始输出"

        # 尝试匹配分数 "score": 7.5 或 score: 7.5
        score_match = re.search(r'(?:\"|\')?score(?:\"|\')?\s*:\s*([\d\.]+)', response)
        if score_match:
            try:
                score = float(score_match.group(1))
                print(f"      📊 从响应中提取分数: {score}")
            except:
                pass

        # 尝试提取 feedback 字段（更加健壮）
        # 方法1: 匹配 "feedback": "..."
        feedback_match = re.search(
            r'(?:\"|\')?feedback(?:\"|\')?\s*:\s*"((?:[^"\\]|\\["\\/bfnrt]|\\u[0-9a-fA-F]{4})*)"',
            response,
            re.DOTALL
        )
        if feedback_match:
            feedback = feedback_match.group(1)
            feedback = feedback.replace('\\"', '"')
            feedback = feedback.replace('\\n', '\n')
            print(f"      💬 从响应中提取 feedback（模式1）")
        else:
            # 方法2: 更宽松的匹配
            feedback_match = re.search(
                r'(?:\"|\')?feedback(?:\"|\')?\s*:\s*"([^"]*(?:\\.[^"]*)*)"',
                response,
                re.DOTALL
            )
            if feedback_match:
                feedback = feedback_match.group(1)
                feedback = feedback.replace('\\"', '"')
                feedback = feedback.replace('\\n', '\n')
                print(f"      💬 从响应中提取 feedback（模式2）")
            else:
                # 方法3: 如果还是失败，尝试找到所有冒号后的内容，取最长的
                content_matches = list(re.finditer(r':\s*"([^"]*(?:\\.[^"]*)*)"', response))
                if len(content_matches) >= 2:
                    # 假设 score 是第一个，feedback 是第二个
                    feedback = content_matches[1].group(1)
                    feedback = feedback.replace('\\"', '"')
                    feedback = feedback.replace('\\n', '\n')
                    print(f"      💬 从响应中提取 feedback（模式3-启发式）")
                else:
                    # 最后的尝试：使用原始响应的部分内容
                    print(f"      ⚠️  无法精确提取 feedback，使用原始响应摘录")

        return {
            'reviewer': reviewer['name'],
            'role': reviewer['role'],
            'score': score,
            'feedback': feedback
        }

    def _diagnose_issue(self, reviews: List[Dict], scores: List[float]) -> Tuple[str, List[str]]:
        """诊断主要问题

        Returns:
            (main_issue, suggestions)
        """
        # 找出分数最低的评审员
        min_idx = scores.index(min(scores))
        worst_review = reviews[min_idx]

        role = worst_review['role']

        # 打印诊断信息
        print(f"\n   📊 诊断信息:")
        print(f"      分数分布: {scores}")
        print(f"      最低分评审员: {worst_review['reviewer']} ({role}), 分数: {scores[min_idx]}")

        # 根据角色诊断问题,映射到Pattern分类维度
        if role == 'Novelty':
            return 'novelty', ['从novelty维度选择创新Pattern', '注入长尾Pattern提升新颖性']
        elif role == 'Methodology':
            return 'stability', ['从stability维度选择稳健Pattern', '注入成熟方法增强鲁棒性']
        elif role == 'Storyteller':
            return 'domain_distance', ['从domain_distance维度选择跨域Pattern', '引入不同视角优化叙事']
        else:
            # Fallback
            return 'novelty', ['从novelty维度选择创新Pattern']

