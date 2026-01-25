from typing import List, Dict

from .config import PipelineConfig
from .utils import compute_jaccard_similarity


class RAGVerifier:
    """RAG 查重验证器"""

    def __init__(self, papers: List[Dict]):
        self.papers = papers

    def verify(self, story: Dict) -> Dict:
        """查重验证

        Returns:
            {
                'pass': bool,
                'collision_detected': bool,
                'similar_papers': List[Dict],
                'max_similarity': float
            }
        """
        print("\n" + "=" * 80)
        print("🔎 Phase 4: RAG Verification (查重验证)")
        print("=" * 80)

        # 简单的相似度计算（基于 Method Skeleton）
        method_skeleton = story.get('method_skeleton', '')

        # 处理 method_skeleton 可能是字典的情况
        if isinstance(method_skeleton, dict):
            # 如果是字典，提取所有值并拼接成字符串
            method_skeleton = ' '.join(str(v) for v in method_skeleton.values() if v)
            print(f"   ⚠️  method_skeleton 是字典类型，已转换为字符串")
        elif not isinstance(method_skeleton, str):
            # 如果不是字符串也不是字典，转换为字符串
            method_skeleton = str(method_skeleton)
            print(f"   ⚠️  method_skeleton 类型异常，已转换为字符串")

        similar_papers = []
        max_similarity = 0.0

        print(f"🔍 检索与当前 Story 相似的论文...")
        print(f"   查询: {method_skeleton[:80]}...")

        for paper in self.papers[:50]:  # 仅检查前 50 篇（演示用）
            paper_method = paper.get('skeleton', {}).get('method_story', '')
            if not paper_method:
                continue

            similarity = compute_jaccard_similarity(method_skeleton, paper_method)

            if similarity > 0.3:  # 过滤低相似度
                similar_papers.append({
                    'paper_id': paper.get('paper_id', ''),
                    'title': paper.get('title', ''),
                    'similarity': similarity,
                    'method': paper_method[:100]
                })
                max_similarity = max(max_similarity, similarity)

        # 排序
        similar_papers.sort(key=lambda x: x['similarity'], reverse=True)
        top_similar = similar_papers[:3]

        # 判断是否撞车
        collision_detected = max_similarity > PipelineConfig.COLLISION_THRESHOLD

        print(f"\n📊 查重结果:")
        print(f"   找到 {len(similar_papers)} 篇相似论文")
        print(f"   最高相似度: {max_similarity:.2f}")

        if top_similar:
            print(f"\n   Top-3 相似论文:")
            for i, paper in enumerate(top_similar, 1):
                print(f"   {i}. {paper['title']}")
                print(f"      相似度: {paper['similarity']:.2f}")
                print(f"      方法: {paper['method'][:60]}...")

        if collision_detected:
            print(f"\n   ⚠️  检测到撞车 (相似度 > {PipelineConfig.COLLISION_THRESHOLD})")
        else:
            print(f"\n   ✅ 未检测到撞车")

        print("=" * 80)

        return {
            'pass': not collision_detected,
            'collision_detected': collision_detected,
            'similar_papers': top_similar,
            'max_similarity': max_similarity
        }

    def generate_pivot_constraints(self, story: Dict, similar_papers: List[Dict]) -> List[str]:
        """生成 Pivot 约束"""
        print("\n🔄 生成 Pivot 约束...")

        constraints = []

        if similar_papers:
            most_similar = similar_papers[0]
            constraints.append(f"禁止使用与《{most_similar['title']}》相同的核心技术")
            constraints.append("将应用场景迁移到新领域（如法律、金融、医疗等）")
            constraints.append("增加额外的约束条件（如无监督、少样本等设定）")

        for constraint in constraints:
            print(f"   - {constraint}")

        return constraints

