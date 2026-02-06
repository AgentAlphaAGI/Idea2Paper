import re
from typing import Dict, List, Tuple, Optional

from idea2paper.config import PipelineConfig
from idea2paper.infra.llm import call_llm, parse_json_from_llm
from idea2paper.review.review_index import ReviewIndex
from idea2paper.infra.run_context import get_logger


def _sigmoid(x: float) -> float:
    return 1 / (1 + pow(2.718281828459045, -x))


def _safe_mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


class MultiAgentCritic:
    """多智能体评审团: 三个角色评审 Story"""

    def __init__(self, review_index: Optional[ReviewIndex] = None):
        self.review_index = review_index
        self.reviewers = [
            {'name': 'Reviewer A', 'role': 'Methodology', 'focus': '技术合理性'},
            {'name': 'Reviewer B', 'role': 'Novelty', 'focus': '创新性'},
            {'name': 'Reviewer C', 'role': 'Storyteller', 'focus': '叙事完整性'}
        ]

    def _log_event(self, event_type: str, payload: Dict):
        logger = get_logger()
        if logger:
            logger.log_event(event_type, payload)

    def _suspect_truncation(self, text: str) -> bool:
        if not text:
            return True
        stripped = text.strip()
        if not stripped:
            return True
        if stripped.count('{') > stripped.count('}'):
            return True
        if stripped.count('[') > stripped.count(']'):
            return True
        if not (stripped.endswith('}') or stripped.endswith(']') or stripped.endswith('```')):
            return True
        return False

    def _validate_comparisons(self, result: Dict, anchors: List[Dict]) -> Tuple[bool, str, Dict]:
        if not isinstance(result, dict):
            return False, "schema_invalid", {}
        comparisons = result.get("comparisons")
        if not isinstance(comparisons, list):
            return False, "schema_invalid", {}

        anchor_map = {a.get("paper_id"): a for a in anchors if a.get("paper_id")}
        if not anchor_map:
            return False, "missing_anchors", {}

        seen = set()
        normalized = []
        for comp in comparisons:
            if not isinstance(comp, dict):
                continue
            pid = comp.get("paper_id")
            if pid not in anchor_map or pid in seen:
                continue
            judgement = comp.get("judgement")
            confidence = comp.get("confidence")
            rationale = comp.get("rationale")
            if judgement not in ("better", "tie", "worse"):
                return False, "schema_invalid", {}
            try:
                confidence = float(confidence)
            except Exception:
                return False, "schema_invalid", {}
            if confidence < 0.0 or confidence > 1.0:
                return False, "schema_invalid", {}
            if not isinstance(rationale, str):
                return False, "schema_invalid", {}
            score_text = f"{anchor_map[pid]['score10']:.1f}"
            if "score10" not in rationale.lower() or score_text not in rationale:
                return False, "missing_score10", {}
            normalized.append({
                "paper_id": pid,
                "judgement": judgement,
                "confidence": confidence,
                "rationale": rationale
            })
            seen.add(pid)

        if len(seen) != len(anchor_map):
            return False, "missing_anchors", {}

        ordered = []
        for a in anchors:
            pid = a.get("paper_id")
            for comp in normalized:
                if comp["paper_id"] == pid:
                    ordered.append(comp)
                    break

        main_gaps = result.get("main_gaps", [])
        if not isinstance(main_gaps, list):
            main_gaps = []

        return True, "", {"comparisons": ordered, "main_gaps": main_gaps}

    def _build_idea_brief_block(self) -> str:
        brief = getattr(self, "_idea_brief", None)
        if not brief:
            return ""
        constraints = ", ".join(brief.get("constraints", []) or [])
        keywords_en = ", ".join(brief.get("keywords_en", []) or [])
        block = "\nUser Requirements Brief (for alignment only):\n"
        if brief.get("problem_definition"):
            block += f"- Problem: {brief.get('problem_definition')}\n"
        if constraints:
            block += f"- Constraints: {constraints}\n"
        if keywords_en:
            block += f"- Keywords: {keywords_en}\n"
        return block

    def _build_anchor_prompt(self, story: Dict, reviewer: Dict, anchors: List[Dict]) -> str:
        problem_text = story.get('problem_framing') or story.get('problem_definition', '')
        method_text = story.get('method_skeleton', '')
        if isinstance(method_text, dict):
            method_text = ' '.join(str(v) for v in method_text.values() if v)

        anchor_lines = []
        for a in anchors:
            anchor_lines.append(
                f"- paper_id: {a['paper_id']} | title: {a.get('title','')} | score10: {a['score10']:.1f}"
            )
        anchor_text = "\n".join(anchor_lines)
        idea_brief_block = self._build_idea_brief_block()

        return f"""
You are a strict reviewer ({reviewer['role']}) for top-tier ML/NLP conferences.
You must NOT output a direct score. Only compare the Story against anchor papers with real review scores.

Anchors (score10 comes from real review statistics):
{anchor_text}
{idea_brief_block}

Story:
Title: {story.get('title','')}
Abstract: {story.get('abstract','')}
Problem: {problem_text}
Method: {method_text}
Claims: {', '.join(story.get('innovation_claims', []))}
Experiments: {story.get('experiments_plan','')}

Task:
For each anchor, decide whether the Story is better, tie, or worse on {reviewer['role']}, and provide confidence (0-1).
You must mention the anchor's score10 in the rationale using the format "score10: X.X".
Comparisons must include every anchor exactly once.
Each rationale must be ONE sentence (<=25 words).

Output JSON ONLY. No markdown, no extra text:
{{
  "comparisons": [
    {{"paper_id":"...", "judgement":"better|tie|worse", "confidence":0.0-1.0, "rationale":"...score10: X.X..."}}
  ],
  "main_gaps": ["gap1", "gap2", "gap3"]
}}
"""

    def _build_repair_prompt(self, previous_text: str, anchors: List[Dict], reviewer_role: str) -> str:
        anchor_lines = []
        for a in anchors:
            anchor_lines.append(f"- paper_id: {a['paper_id']} | score10: {a['score10']:.1f}")
        anchor_text = "\n".join(anchor_lines)
        truncated_prev = previous_text[:6000]
        return f"""
You must fix the previous output into STRICT valid JSON.
Role: {reviewer_role}

Anchors (must cover ALL exactly once):
{anchor_text}

Rules:
1) Output JSON ONLY (no markdown, no explanations).
2) "comparisons" length MUST equal number of anchors.
3) Each comparison's paper_id must be from the anchor list.
4) judgement must be one of: better|tie|worse.
5) confidence must be a number in [0,1].
6) rationale must be ONE sentence and MUST include "score10: X.X" for that anchor.

Previous output to repair:
{truncated_prev}

Return ONLY the corrected JSON:
{{
  "comparisons": [
    {{"paper_id":"...", "judgement":"better|tie|worse", "confidence":0.0-1.0, "rationale":"...score10: X.X..."}}
  ],
  "main_gaps": ["gap1", "gap2", "gap3"]
}}
"""

    def _build_reemit_prompt(self, story: Dict, reviewer: Dict, anchors: List[Dict]) -> str:
        return self._build_anchor_prompt(story, reviewer, anchors)

    def _get_comparisons_with_retries(
        self,
        story: Dict,
        reviewer: Dict,
        anchors: List[Dict],
        pattern_id: str
    ) -> Tuple[List[Dict], List[str]]:
        base_prompt = self._build_anchor_prompt(story, reviewer, anchors)
        response = call_llm(
            base_prompt,
            temperature=PipelineConfig.LLM_TEMPERATURE_CRITIC_MAIN,
            max_tokens=800,
            timeout=180,
        )
        result = parse_json_from_llm(response)
        ok, reason, normalized = (False, "parse_failed", {})
        if result:
            ok, reason, normalized = self._validate_comparisons(result, anchors)

        if ok:
            return normalized["comparisons"], normalized.get("main_gaps", [])

        self._log_event("critic_invalid_output", {
            "pattern_id": pattern_id,
            "role": reviewer.get("role"),
            "attempt": 0,
            "reason": reason,
            "response_len": len(response),
            "truncated_suspected": self._suspect_truncation(response)
        })

        retries = getattr(PipelineConfig, "CRITIC_JSON_RETRIES", 2)
        last_reason = reason
        last_response = response

        for attempt in range(1, retries + 1):
            truncated = self._suspect_truncation(last_response)
            strategy = "reemit" if truncated else "repair"
            if strategy == "repair":
                prompt = self._build_repair_prompt(last_response, anchors, reviewer.get("role", ""))
            else:
                prompt = self._build_reemit_prompt(story, reviewer, anchors)

            print(f"   ⏳ Critic JSON retry {attempt}/{retries} ({strategy})...")
            response = call_llm(
                prompt,
                temperature=PipelineConfig.LLM_TEMPERATURE_CRITIC_REPAIR,
                max_tokens=800,
                timeout=180,
            )
            result = parse_json_from_llm(response)
            ok, reason, normalized = (False, "parse_failed", {})
            if result:
                ok, reason, normalized = self._validate_comparisons(result, anchors)

            if ok:
                self._log_event("critic_parse_recovered", {
                    "pattern_id": pattern_id,
                    "role": reviewer.get("role"),
                    "attempt": attempt,
                    "strategy": strategy
                })
                return normalized["comparisons"], normalized.get("main_gaps", [])

            self._log_event("critic_invalid_output", {
                "pattern_id": pattern_id,
                "role": reviewer.get("role"),
                "attempt": attempt,
                "reason": reason,
                "response_len": len(response),
                "truncated_suspected": self._suspect_truncation(response),
                "strategy": strategy
            })
            last_reason = reason
            last_response = response

        if getattr(PipelineConfig, "CRITIC_STRICT_JSON", True):
            self._log_event("critic_invalid_output_fatal", {
                "pattern_id": pattern_id,
                "role": reviewer.get("role"),
                "attempts": retries,
                "reason": last_reason
            })
            raise RuntimeError(
                f"Critic JSON invalid after retries: role={reviewer.get('role')} "
                f"pattern_id={pattern_id}, reason={last_reason}"
            )

        self._log_event("critic_fallback_neutral", {
            "pattern_id": pattern_id,
            "role": reviewer.get("role"),
            "attempts": retries,
            "reason": last_reason
        })
        comparisons = [
            {"paper_id": a["paper_id"], "judgement": "tie", "confidence": 0.0, "rationale": "LLM output unavailable"}
            for a in anchors
        ]
        main_gaps = ["LLM output unavailable; fallback to neutral comparisons."]
        return comparisons, main_gaps

    def _compute_pass_decision(self, avg_score: float, role_scores: Dict[str, float], pattern_id: str) -> Tuple[bool, Dict]:
        mode = getattr(PipelineConfig, "PASS_MODE", "two_of_three_q75_and_avg_ge_q50")
        min_papers = getattr(PipelineConfig, "PASS_MIN_PATTERN_PAPERS", 20)
        fallback = getattr(PipelineConfig, "PASS_FALLBACK", "global")

        used_distribution = "fixed"
        pattern_stats_n = 0
        q50 = None
        q75 = None

        if self.review_index and pattern_id:
            stats = self.review_index.get_pattern_quantiles(pattern_id, [0.5, 0.75])
            pattern_stats_n = int(stats.get("n", 0) or 0)
            if pattern_stats_n >= min_papers and stats.get("q50") is not None and stats.get("q75") is not None:
                q50 = float(stats.get("q50"))
                q75 = float(stats.get("q75"))
                used_distribution = "pattern"
            else:
                if fallback == "global":
                    gstats = self.review_index.get_global_quantiles([0.5, 0.75])
                    if gstats.get("q50") is not None and gstats.get("q75") is not None:
                        q50 = float(gstats.get("q50"))
                        q75 = float(gstats.get("q75"))
                        used_distribution = "global"
                if used_distribution == "fixed":
                    q50 = None
                    q75 = None

        roles = ["Methodology", "Novelty", "Storyteller"]
        roles_ge_q75 = {r: False for r in roles}
        count_ge_q75 = 0
        avg_ge_q50 = None

        passed = False
        if mode == "two_of_three_q75_and_avg_ge_q50" and q50 is not None and q75 is not None:
            roles_ge_q75 = {r: float(role_scores.get(r, 0.0)) >= q75 for r in roles}
            count_ge_q75 = sum(1 for v in roles_ge_q75.values() if v)
            avg_ge_q50 = avg_score >= q50
            passed = (count_ge_q75 >= 2) and avg_ge_q50
        else:
            passed = avg_score >= PipelineConfig.PASS_SCORE

        pass_audit = {
            "mode": mode if mode else "fixed",
            "used_distribution": used_distribution,
            "pattern_paper_count": pattern_stats_n,
            "q50": q50,
            "q75": q75,
            "count_roles_ge_q75": count_ge_q75,
            "roles_ge_q75": roles_ge_q75,
            "avg_ge_q50": avg_ge_q50,
            "avg_score": avg_score
        }

        return passed, pass_audit
    def review(self, story: Dict, context: Optional[Dict] = None) -> Dict:
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
        print("🔍 Phase 3: Multi-Agent Critic (多智能体评审 - Anchored)")
        print("=" * 80)

        context = context or {}
        pattern_id = context.get("pattern_id", "")
        pattern_info = context.get("pattern_info", {}) or {}
        anchors = context.get("anchors", []) or []
        self._idea_brief = context.get("idea_brief")

        if not anchors and pattern_id and self.review_index:
            anchors = self.review_index.select_initial_anchors(
                pattern_id,
                pattern_info,
                max_initial=getattr(PipelineConfig, "ANCHOR_MAX_INITIAL", 7),
                quantiles=getattr(PipelineConfig, "ANCHOR_QUANTILES", [0.1, 0.25, 0.5, 0.75, 0.9]),
                max_exemplars=getattr(PipelineConfig, "ANCHOR_MAX_EXEMPLARS", 2),
            )

        if not anchors:
            # Fallback: deterministic neutral scoring
            reviews = []
            scores = []
            for reviewer in self.reviewers:
                reviews.append({
                    'reviewer': reviewer['name'],
                    'role': reviewer['role'],
                    'score': 5.0,
                    'feedback': 'No anchors available; defaulted to neutral score.'
                })
                scores.append(5.0)
            avg_score = _safe_mean(scores)
            main_issue, suggestions = self._diagnose_issue(reviews, scores)
            return {
                'pass': avg_score >= PipelineConfig.PASS_SCORE,
                'avg_score': avg_score,
                'reviews': reviews,
                'main_issue': main_issue,
                'suggestions': suggestions,
                'audit': {
                    'pattern_id': pattern_id,
                    'anchors': [],
                    'role_details': {}
                }
            }

        # Round 1 anchored review
        round1 = self._anchored_reviews(story, anchors, pattern_id)
        densify_needed = any(
            (r['loss'] > getattr(PipelineConfig, "DENSIFY_LOSS_THRESHOLD", 0.03)) or
            (r['monotonic_violations'] >= 1) or
            (r['avg_confidence'] < getattr(PipelineConfig, "DENSIFY_MIN_AVG_CONF", 0.45))
            for r in round1['role_details'].values()
        )
        densify_enabled = getattr(PipelineConfig, "ANCHOR_DENSIFY_ENABLE", True)

        anchors_rounds = [anchors]
        if densify_needed and not densify_enabled:
            self._log_event("critic_densify_skipped", {
                "pattern_id": pattern_id,
                "reason": "disabled",
            })
        if densify_enabled and densify_needed and pattern_id and self.review_index:
            scores_round1 = [r['score'] for r in round1['role_details'].values()]
            S_hint = _safe_mean(scores_round1) if scores_round1 else 5.0
            additional = self.review_index.select_adaptive_anchors(
                pattern_id,
                selected_ids=[a["paper_id"] for a in anchors],
                S_hint=S_hint,
                offsets=getattr(PipelineConfig, "DENSIFY_OFFSETS", [-0.5, 0.5, -0.25, 0.25]),
                max_total=getattr(PipelineConfig, "ANCHOR_MAX_TOTAL", 9)
            )
            if additional:
                anchors = anchors + additional
                anchors_rounds.append(additional)
                round2 = self._anchored_reviews(story, anchors, pattern_id)
            else:
                round2 = round1
        else:
            round2 = round1

        # Build final outputs
        reviews = round2['reviews']
        scores = round2['scores']
        avg_score = _safe_mean(scores)
        role_scores = {r['role']: r['score'] for r in reviews}
        passed, pass_audit = self._compute_pass_decision(avg_score, role_scores, pattern_id)
        main_issue, suggestions = self._diagnose_issue(reviews, scores)

        print("\n" + "-" * 80)
        print(f"📊 评审结果: 平均分 {avg_score:.2f}/10 - {'✅ PASS' if passed else '❌ FAIL'}")
        if not passed:
            print(f"🔧 主要问题: {main_issue}")
            print(f"💡 建议: {', '.join(suggestions)}")
        print("=" * 80)

        audit = {
            'pattern_id': pattern_id,
            'anchors': anchors,
            'role_details': round2['role_details'],
            'anchors_rounds': anchors_rounds,
            'pass': pass_audit
        }

        self._log_event("pass_threshold_computed", {
            "pattern_id": pattern_id,
            "used_distribution": pass_audit.get("used_distribution"),
            "q50": pass_audit.get("q50"),
            "q75": pass_audit.get("q75"),
            "count_roles_ge_q75": pass_audit.get("count_roles_ge_q75"),
            "avg_score": avg_score,
            "passed": passed
        })

        return {
            'pass': passed,
            'avg_score': avg_score,
            'reviews': reviews,
            'main_issue': main_issue,
            'suggestions': suggestions,
            'audit': audit
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

        # NOTE: _single_review is kept for backward compatibility only.
        # For anchored review, use _anchored_review().
        problem_text = story.get('problem_framing') or story.get('problem_definition', '')
        method_text = story.get('method_skeleton', '')
        if isinstance(method_text, dict):
            method_text = ' '.join(str(v) for v in method_text.values() if v)

        # 构建 Prompt
        prompt = f"""
你是顶级 NLP 会议（如 ACL/ICLR）的**严厉评审专家** {reviewer['name']}，专注于评估{reviewer['focus']}。
你的打分标准非常严格，满分 10 分。6 分以下为不及格（Reject），8 分以上为优秀（Accept）。
{special_instructions}
请评审以下论文 Story：

【标题】{story.get('title', '')}

【摘要】{story.get('abstract', '')}

【问题定义】{problem_text}

【方法概述】{method_text}

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
        response = call_llm(
            prompt,
            temperature=PipelineConfig.LLM_TEMPERATURE_CRITIC_ANCHORED,
            max_tokens=800,
            timeout=180,
        )

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

    def _anchored_reviews(self, story: Dict, anchors: List[Dict], pattern_id: str) -> Dict:
        reviews = []
        scores = []
        role_details = {}
        for reviewer in self.reviewers:
            print(f"\n📝 {reviewer['name']} ({reviewer['role']}) 评审中...")
            anchored = self._anchored_review(story, reviewer, anchors, pattern_id)
            score = anchored['score']
            reviews.append({
                'reviewer': reviewer['name'],
                'role': reviewer['role'],
                'score': score,
                'feedback': anchored['feedback']
            })
            scores.append(score)
            role_details[reviewer['role']] = anchored['detail']

            print(f"   评分: {score:.1f}/10")
            print(f"   反馈: {anchored['feedback']}")

        return {
            'reviews': reviews,
            'scores': scores,
            'role_details': role_details
        }

    def _anchored_review(self, story: Dict, reviewer: Dict, anchors: List[Dict], pattern_id: str) -> Dict:
        """Anchored review: compare against real papers, then deterministically fit score."""
        comparisons, main_gaps = self._get_comparisons_with_retries(
            story=story,
            reviewer=reviewer,
            anchors=anchors,
            pattern_id=pattern_id
        )

        score, detail = self._compute_score_from_comparisons(anchors, comparisons)
        feedback = f"Main gaps: {', '.join(main_gaps[:3])}. Anchored against {len(anchors)} papers."

        return {
            'score': score,
            'feedback': feedback,
            'detail': {
                'comparisons': comparisons,
                'main_gaps': main_gaps,
                'score': score,
                **detail
            }
        }

    def _compute_score_from_comparisons(self, anchors: List[Dict], comparisons: List[Dict]) -> Tuple[float, Dict]:
        # Map paper_id -> comparison
        comp_map = {c.get('paper_id'): c for c in comparisons if c.get('paper_id')}
        probs = []
        weights = []
        scores = []
        confs = []

        for a in anchors:
            pid = a["paper_id"]
            comp = comp_map.get(pid, {"judgement": "tie", "confidence": 0.0})
            judgement = comp.get("judgement", "tie")
            confidence = float(comp.get("confidence", 0.0) or 0.0)
            confidence = max(0.0, min(1.0, confidence))
            if judgement == "better":
                p = 0.5 + 0.45 * confidence
            elif judgement == "worse":
                p = 0.5 - 0.45 * confidence
            else:
                p = 0.5
            probs.append(p)
            weights.append(float(a.get("weight", 1.0)))
            scores.append(float(a.get("score10", 5.0)))
            confs.append(confidence)

        # compute monotonic violations
        sorted_pairs = sorted(zip(scores, probs), key=lambda x: x[0])
        monotonic_violations = 0
        prev_p = None
        for s, p in sorted_pairs:
            if prev_p is not None and p > prev_p + 0.05:
                monotonic_violations += 1
            prev_p = p

        k = getattr(PipelineConfig, "SIGMOID_K", 1.2)
        step = getattr(PipelineConfig, "GRID_STEP", 0.01)
        best_s = 5.0
        best_loss = None
        S = 1.0
        while S <= 10.0 + 1e-9:
            loss = 0.0
            for p, w, s in zip(probs, weights, scores):
                pred = _sigmoid(k * (S - s))
                loss += w * (pred - p) ** 2
            if best_loss is None or loss < best_loss:
                best_loss = loss
                best_s = S
            S += step

        detail = {
            "loss": best_loss if best_loss is not None else 0.0,
            "avg_confidence": _safe_mean(confs),
            "monotonic_violations": monotonic_violations
        }
        return best_s, detail

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
