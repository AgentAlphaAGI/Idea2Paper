from typing import Dict, List, Tuple, Optional

from .config import PipelineConfig


class PatternSelector:
    """Pattern 选择器: 选择多样化的 Pattern"""

    def __init__(self, recalled_patterns: List[Tuple[str, Dict, float]]):
        """
        Args:
            recalled_patterns: [(pattern_id, pattern_info, score), ...]
        """
        self.recalled_patterns = recalled_patterns

    def select(self) -> Dict[str, Tuple[str, Dict]]:
        """选择 3 个不同策略的 Pattern

        Returns:
            {
                'conservative': (pattern_id, pattern_info),
                'innovative': (pattern_id, pattern_info),
                'cross_domain': (pattern_id, pattern_info)
            }
        """
        print("\n" + "=" * 80)
        print("📋 Phase 1: Pattern Selection (策略选择)")
        print("=" * 80)

        selected = {}

        # 1. Conservative (稳健型): 最高分
        conservative = self._select_conservative()
        if conservative:
            selected['conservative'] = conservative
            print(f"\n✅ [稳健型] {conservative[0]}")
            print(f"   名称: {conservative[1].get('name', 'N/A')}")
            print(f"   聚类大小: {conservative[1].get('cluster_size', 0)} 篇")
            print(f"   策略: Score 最高，最符合直觉")

        # 2. Innovative (创新型): Cluster Size 小
        innovative = self._select_innovative(exclude=[conservative[0]] if conservative else [])
        if innovative:
            selected['innovative'] = innovative
            print(f"\n✅ [创新型] {innovative[0]}")
            print(f"   名称: {innovative[1].get('name', 'N/A')}")
            print(f"   聚类大小: {innovative[1].get('cluster_size', 0)} 篇")
            print(f"   策略: Cluster Size < {PipelineConfig.INNOVATIVE_CLUSTER_SIZE_THRESHOLD}，容易产生新颖结合")

        # 3. Cross-Domain (跨域型): 来自路径2或路径3
        cross_domain = self._select_cross_domain(
            exclude=[conservative[0] if conservative else None,
                    innovative[0] if innovative else None]
        )
        if cross_domain:
            selected['cross_domain'] = cross_domain
            print(f"\n✅ [跨域型] {cross_domain[0]}")
            print(f"   名称: {cross_domain[1].get('name', 'N/A')}")
            print(f"   聚类大小: {cross_domain[1].get('cluster_size', 0)} 篇")
            print(f"   策略: 来自领域相关或Paper相似路径")

        print("\n" + "-" * 80)
        print(f"✅ 共选择 {len(selected)} 个 Pattern")
        print("=" * 80)

        return selected

    def _select_conservative(self) -> Optional[Tuple[str, Dict]]:
        """选择稳健型: Score 最高"""
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
               pinfo.get('cluster_size', 999) < PipelineConfig.INNOVATIVE_CLUSTER_SIZE_THRESHOLD
        ]

        if not candidates:
            # 如果没有符合条件的，选择 Cluster Size 最小的
            candidates = [
                (pid, pinfo, score)
                for pid, pinfo, score in self.recalled_patterns
                if pid not in exclude
            ]
            candidates.sort(key=lambda x: x[1].get('cluster_size', 999))

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

