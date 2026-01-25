from typing import Dict, List, Tuple, Optional

from .config import PipelineConfig
from .utils import call_llm, parse_json_from_llm


class PatternSelector:
    """Pattern 选择器: 选择多样化的 Pattern（支持 LLM 辅助分类和动态排序）"""

    def __init__(self, recalled_patterns: List[Tuple[str, Dict, float]], user_idea: str = ""):
        """
        Args:
            recalled_patterns: [(pattern_id, pattern_info, score), ...]
            user_idea: 用户的原始 Idea（用于 LLM 判断领域距离）
        """
        self.recalled_patterns = recalled_patterns
        self.user_idea = user_idea
        self.pattern_classifications = {}  # 存储 LLM 分类结果

    def select(self) -> Dict[str, List[Tuple[str, Dict, Dict]]]:
        """选择多个 Pattern 并按三个维度（稳健度、新颖度、跨域度）分别排序

        Returns:
            {
                'stability': [(pattern_id, pattern_info, metadata), ...],   # 按稳定性(stability_score)降序
                'novelty': [(pattern_id, pattern_info, metadata), ...],     # 按新颖性(novelty_score)降序
                'domain_distance': [(pattern_id, pattern_info, metadata), ...]  # 按领域距离(domain_distance)升序 (越小越好)
            }
        """
        print("\n" + "=" * 80)
        print("📋 Phase 1: Pattern Selection (多维度评分与排序)")
        print("=" * 80)

        # Step 1: 为所有 Pattern 计算三个维度的得分
        print("\n🤖 Step 1: 多维度评分 (所有 Patterns)...")
        self._score_patterns_multidimensional()

        # Step 2: 按三个维度分别排序
        print("\n📊 Step 2: 按维度排序...")
        ranked = self._rank_patterns_by_dimensions()

        # Step 3: 打印排序结果
        print("\n" + "=" * 80)
        print("✅ Pattern 多维度排序结果:")
        print("=" * 80)

        dimension_names = {
            'stability': '【稳健度排序】',
            'novelty': '【新颖度排序】',
            'domain_distance': '【跨域度排序 (低→高距离)】'
        }

        for dimension, patterns in ranked.items():
            if not patterns:
                continue

            print(f"\n{dimension_names.get(dimension, dimension)} 共 {len(patterns)} 个:")
            for i, (pid, pinfo, meta) in enumerate(patterns[:5], 1):  # 显示前5个
                print(f"  {i}. {pid}")
                print(f"     名称: {pinfo.get('name', 'N/A')}")
                print(f"     聚类: {pinfo.get('size', 0)} 篇")
                if 'scores' in meta:
                    scores = meta['scores']
                    print(f"     得分: 稳健={scores.get('stability_score', 0):.2f}, "
                          f"新颖={scores.get('novelty_score', 0):.2f}, "
                          f"域距={scores.get('domain_distance', 0):.2f}")

        print("\n" + "-" * 80)
        total = sum(len(patterns) for patterns in ranked.values())
        print(f"✅ 共评分 {total} 个 Pattern")
        print("=" * 80)

        return ranked

    def _score_patterns_multidimensional(self):
        """为所有 Pattern 计算三个维度的得分（稳健度、新颖度、跨域度）"""
        # 对 Top-20 进行 LLM 评分（平衡效果和成本）
        top_patterns = self.recalled_patterns[:20]

        for pattern_id, pattern_info, recall_score in top_patterns:
            # 构建 Pattern 摘要信息
            pattern_name = pattern_info.get('name', '')
            pattern_size = pattern_info.get('size', 0)

            summary = pattern_info.get('summary', {})
            if isinstance(summary, dict):
                representative_ideas = summary.get('representative_ideas', [])[:3]
                common_problems = summary.get('common_problems', [])[:2]
            else:
                representative_ideas = []
                common_problems = []

            # 调用 LLM 进行多维度评分
            scores = self._call_llm_for_multidim_scoring(
                pattern_id, pattern_name, pattern_size,
                representative_ideas, common_problems
            )

            if scores:
                self.pattern_classifications[pattern_id] = scores
                print(f"  ✓ {pattern_id}: 稳健={scores.get('stability_score', 0):.2f}, "
                      f"新颖={scores.get('novelty_score', 0):.2f}, "
                      f"域距={scores.get('domain_distance', 0):.2f}")

    def _generate_reference_examples(self, current_pattern_id: str) -> str:
        """生成参考示例来校准 LLM 评分，基于已评分的 Pattern"""
        # 如果已有评分，使用它们作为参考；否则生成人工示例
        if not self.pattern_classifications:
            # 没有已评分的Pattern，使用人工示例
            return """
Example 1 - LOW novelty, HIGH stability (Size 150):
  "Attention Is All You Need" application - highly replicated, but well-known approach
  → stability_score: 0.85, novelty_score: 0.15

Example 2 - HIGH novelty, MEDIUM stability (Size 25):
  "Reframing task as code generation problem" - novel angle, but niche community
  → stability_score: 0.35, novelty_score: 0.75

Example 3 - MEDIUM novelty, MEDIUM stability (Size 60):
  "Combining RAG with multi-hop reasoning" - interesting combination, growing adoption
  → stability_score: 0.60, novelty_score: 0.55
"""
        else:
            # 从已评分中提取几个代表性样本
            samples = []
            for pid, scores in list(self.pattern_classifications.items())[:3]:
                pinfo = next((p[1] for p in self.recalled_patterns if p[0] == pid), {})
                samples.append(f"  {pid} (Size {pinfo.get('size', '?')}): "
                             f"stability={scores.get('stability_score', 0):.2f}, "
                             f"novelty={scores.get('novelty_score', 0):.2f}")
            return "Recent scoring calibration:\n" + "\n".join(samples) if samples else ""

    def _call_llm_for_multidim_scoring(self, pattern_id: str, pattern_name: str,
                                       pattern_size: int, ideas: List[str],
                                       problems: List[str]) -> Optional[Dict]:
        """调用 LLM 为单个 Pattern 计算三个维度的得分（稳健度、新颖度、跨域度）"""

        ideas_text = "\n".join(f"- {idea}" for idea in ideas[:3])

        # 生成一些对比参考（从已评分的 Pattern 中抽取）
        reference_examples = self._generate_reference_examples(pattern_id)

        prompt = f"""
You are a **CRITICAL Multidimensional Pattern Scorer** for top-tier AI conferences (ICLR/NeurIPS).
Your task is to rigorously evaluate a research pattern across THREE independent dimensions.
⚠️  IMPORTANT: Avoid clustering scores in the middle range. Be discriminative!

【User's Research Idea】
"{self.user_idea}"

【Pattern Information】
Pattern ID: {pattern_id}
Name: {pattern_name}
Cluster Size: {pattern_size} papers
Representative Research Ideas:
{ideas_text if ideas_text else "N/A"}

【Reference Examples (for calibration)】
{reference_examples}

【Scoring Guidelines - Be CRITICAL and DISCRIMINATIVE】

**Stability Score (0.0-1.0)** - How proven, mature, and widely-adopted?
Consider: Has this approach been replicated across many papers? Are there established benchmarks?
- 0.1-0.25: Highly experimental, niche idea, Size < 15, no standard benchmarks, high uncertainty
- 0.3-0.45: Early-stage research, Size 15-40, some implementations but inconsistent results
- 0.5-0.65: Maturing approach, Size 40-70, multiple independent implementations, emerging consensus
- 0.7-0.85: Well-established, Size 70-120, standard benchmarks, widely replicated with consistent gains
- 0.9-1.0: Foundational/canonical approach, Size > 120, ubiquitous, considered solved or foundational
🔴 RED FLAG: Avoid giving middle scores (0.4-0.6) to everything. Distinguish clearly.

**Novelty Score (0.0-1.0)** - How original, counter-intuitive, and fresh is this?
Consider: Is this a new perspective? Does it challenge existing assumptions? Or incremental variation?
- 0.1-0.25: Well-trodden path, combinations of existing techniques, straightforward application
- 0.3-0.45: Some novelty in execution or application domain, but builds on established ideas
- 0.5-0.65: Interesting recombination or new angle on known problems, moderate originality
- 0.7-0.85: Novel methodology, surprising insight, challenges conventional wisdom, fresh angle
- 0.9-1.0: Paradigm shift, highly counter-intuitive, fundamentally new problem formulation
🔴 RED FLAG: If pattern_name suggests "reframing" or "transforming", likely 0.6+. If it's optimization/tuning, likely 0.2-0.4.

**Domain Distance (0.0-1.0)** - How different from user's core idea?
Consider semantic and methodological distance, not just application domain.
- 0.0-0.15: Directly addresses same problem, highly relevant methodology
- 0.2-0.35: Related domain/approach, applicable with minor adaptation
- 0.4-0.55: Different domain but transferable insights, moderate adaptation needed
- 0.6-0.8: Orthogonal domain, interesting cross-domain inspirations
- 0.85-1.0: Completely different field, minimal direct relevance
💡 TIP: Compare pattern semantics to user idea content for distance.

【CRITICAL INSTRUCTIONS】
1. DO NOT give all patterns middle-range scores (0.4-0.6). Spread the distribution.
2. DISTINGUISH between: optimization (low novelty), new methodology (medium), paradigm shift (high).
3. Large cluster size (>100) should NOT automatically mean high stability if methodology is flawed.
4. Small cluster size (<20) should NOT automatically mean low novelty; niche innovation exists.

【Output Format - JSON ONLY】
{{
  "stability_score": 0.75,
  "novelty_score": 0.55,
  "domain_distance": 0.25,
  "reasoning": "Example: Established approach (Size 82), novel reframing angle, same-domain application"
}}
"""

        try:
            # 使用更长的超时时间（180 秒）以应对网络较慢的情况
            response = call_llm(prompt, temperature=0.3, max_tokens=300, timeout=180)
            scores = parse_json_from_llm(response)
            if scores and all(k in scores for k in ['stability_score', 'novelty_score', 'domain_distance']):
                return scores
        except Exception as e:
            print(f"  ⚠️  LLM 评分失败 ({pattern_id}): {e}")

        # Fallback: 使用规则计算
        return self._fallback_multidim_scoring(pattern_size)

    def _rank_patterns_by_dimensions(self) -> Dict[str, List[Tuple[str, Dict, Dict]]]:
        """按三个维度（稳健度、新颖度、跨域度）分别排序所有 Pattern"""
        ranked = {
            'stability': [],       # 按 stability_score 降序
            'novelty': [],         # 按 novelty_score 降序
            'domain_distance': []  # 按 domain_distance 升序（越小越好）
        }

        for pattern_id, pattern_info, recall_score in self.recalled_patterns:
            # 获取 LLM 评分结果（如果有）
            scores = self.pattern_classifications.get(pattern_id, None)

            # 如果 LLM 评分成功，使用 LLM 的评分
            if scores:
                metadata = {
                    'recall_score': recall_score,
                    'scores': scores
                }
            else:
                # Fallback: 使用规则计算（兼容性）
                scores = self._fallback_multidim_scoring(pattern_info.get('size', 0))
                metadata = {
                    'recall_score': recall_score,
                    'scores': scores
                }

            # 将该 pattern 添加到所有三个维度的列表中
            ranked['stability'].append((pattern_id, pattern_info, metadata))
            ranked['novelty'].append((pattern_id, pattern_info, metadata))
            ranked['domain_distance'].append((pattern_id, pattern_info, metadata))

        # 按三个维度分别排序
        ranked['stability'].sort(
            key=lambda x: x[2].get('scores', {}).get('stability_score', 0),
            reverse=True
        )
        ranked['novelty'].sort(
            key=lambda x: x[2].get('scores', {}).get('novelty_score', 0),
            reverse=True
        )
        # domain_distance 越小越好（越接近用户想法），所以升序
        ranked['domain_distance'].sort(
            key=lambda x: x[2].get('scores', {}).get('domain_distance', 1.0),
            reverse=False
        )

        return ranked

    def _fallback_multidim_scoring(self, pattern_size: int) -> Dict:
        """Fallback 规则计算（当 LLM 评分失败时）

        基于 pattern_size 的多维度启发式评分
        """
        # 【设计思路】根据 size 估计三个维度的得分
        # Size 越大 → Stability 越高
        # Size 越小 → Novelty 越高（小众方向往往更新颖）
        # Domain Distance 需要更多信息才能判断，这里设为中等值

        if pattern_size > 100:
            # 大型成熟社区
            stability = 0.85
            novelty = 0.25
            domain_dist = 0.25
        elif pattern_size > 70:
            # 成熟社区
            stability = 0.80
            novelty = 0.30
            domain_dist = 0.30
        elif pattern_size > 40:
            # 中等社区
            stability = 0.65
            novelty = 0.45
            domain_dist = 0.40
        elif pattern_size > 20:
            # 小型但有一定基础
            stability = 0.50
            novelty = 0.60
            domain_dist = 0.35
        else:
            # 非常小的社区，高创新性
            stability = 0.30
            novelty = 0.80
            domain_dist = 0.50

        return {
            'stability_score': stability,
            'novelty_score': novelty,
            'domain_distance': domain_dist,
            'reasoning': f'Rule-based: size={pattern_size}'
        }

    # 保留旧方法以兼容旧代码（标记为 deprecated）
    def _select_conservative(self) -> Optional[Tuple[str, Dict]]:
        """【已弃用】选择稳健型: Score 最高"""
        if not self.recalled_patterns:
            return None

        # 已经按分数排序，选择第一个
        pattern_id, pattern_info, score = self.recalled_patterns[0]
        return (pattern_id, pattern_info)

    def _select_innovative(self, exclude: List[str]) -> Optional[Tuple[str, Dict]]:
        """选择创新型: Cluster Size 最小"""
        candidates = [
            (pid, pinfo, score)
            for pid, pinfo, score in self.recalled_patterns
            if pid not in exclude and
               pinfo.get('size', 999) < PipelineConfig.INNOVATIVE_CLUSTER_SIZE_THRESHOLD
        ]

        if not candidates:
            # 如果没有符合条件的，选择 Cluster Size 最小的
            candidates = [
                (pid, pinfo, score)
                for pid, pinfo, score in self.recalled_patterns
                if pid not in exclude
            ]
            candidates.sort(key=lambda x: x[1].get('size', 999))

        if candidates:
            return (candidates[0][0], candidates[0][1])
        return None

    def _select_cross_domain(self, exclude: List[str]) -> Optional[Tuple[str, Dict]]:
        """选择跨域型: 从剩余的中选择"""
        candidates = [
            (pid, pinfo, score)
            for pid, pinfo, score in self.recalled_patterns
            if pid not in exclude
        ]

        if candidates:
            # 选择得分第二高的（不同于 conservative）
            return (candidates[0][0], candidates[0][1])
        return None

