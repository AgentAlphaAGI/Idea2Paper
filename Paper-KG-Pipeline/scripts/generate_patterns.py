"""
基于skeleton+tricks聚类生成patterns
输出：
1. patterns_structured.json - 结构化数据
2. patterns_guide.txt - 用户指导文档
3. patterns_statistics.json - 统计报告
"""

import glob
import json
import os
import time
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np
import requests
from sklearn.cluster import AgglomerativeClustering

# LLM配置 - 切换至 SiliconFlow (参考 01_RAG_minimal_DEMO)
LLM_CONFIG = {
    "api_url": os.environ.get("LLM_API_URL", "https://api.siliconflow.cn/v1/chat/completions"),
    "auth_token": os.environ.get("SILICONFLOW_API_KEY", "sk-your-api-key-here"),
    "model": os.environ.get("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
}

# Embedding配置
EMBED_CONFIG = {
    "api_url": os.environ.get("EMBED_API_URL", "https://api.siliconflow.cn/v1/embeddings"),
    "auth_token": os.environ.get("SILICONFLOW_API_KEY", "sk-your-api-key-here"),
    "model": os.environ.get("EMBED_MODEL", "Qwen/Qwen3-Embedding-4B")
}

print(f"🚀 正在启动 Pattern 生成 (使用 SiliconFlow)...")
print(f"   - LLM 模型: {LLM_CONFIG['model']}")
print(f"   - Embedding 模型: {EMBED_CONFIG['model']}")

if not LLM_CONFIG["auth_token"]:
    print("⚠️  警告: 未设置 LLM_AUTH_TOKEN 环境变量")
    print("   Pattern生成功能将不可用，但可以直接使用已生成的 patterns_structured.json")

# 聚类参数
CLUSTER_PARAMS = {
    "distance_threshold": 0.35,  # 距离阈值
    "min_cluster_size": 5,       # 最小cluster大小
    "skeleton_weight": 0.4,      # skeleton权重
    "tricks_weight": 0.6,        # tricks权重
}


def get_embedding(text: str, max_retries: int = 3) -> List[float]:
    """获取文本的embedding向量"""
    url = EMBED_CONFIG["api_url"]
    headers = {
        "Authorization": f"Bearer {EMBED_CONFIG['auth_token']}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": EMBED_CONFIG["model"],
        "input": text[:8000]
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            return result['data'][0]['embedding']
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  ❌ Embedding失败: {e}")
                return [0.0] * 4096
    
    return [0.0] * 4096


def call_llm(prompt: str, max_retries: int = 3) -> str:
    """调用LLM API"""
    url = LLM_CONFIG["api_url"]
    headers = {
        "Authorization": f"Bearer {LLM_CONFIG['auth_token']}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": LLM_CONFIG["model"],
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content'].strip()
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  ❌ LLM调用失败: {e}")
                return ""
    
    return ""


def load_all_papers(base_dir: str = None) -> List[Dict]:
    """加载所有论文数据"""
    if base_dir is None:
        # 默认使用Pipeline的data目录
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.join(os.path.dirname(script_dir), "data")
    
    all_papers = []
    
    # 遍历所有会议目录
    for conf_dir in glob.glob(os.path.join(base_dir, "*")):
        if not os.path.isdir(conf_dir):
            continue
        
        conf_name = os.path.basename(conf_dir)
        files = glob.glob(os.path.join(conf_dir, "*_paper_node.json"))
        
        if not files:
            continue
        
        print(f"📁 加载 {conf_name}: {len(files)} 篇论文")
        
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    paper = json.load(f)
                    paper['conference'] = conf_name
                    paper['file_path'] = file_path
                    
                    # 验证必要字段
                    if 'skeleton' in paper and 'tricks' in paper:
                        all_papers.append(paper)
            except Exception as e:
                print(f"  ⚠️  读取失败 {file_path}: {e}")
    
    return all_papers


def build_pattern_embeddings(papers: List[Dict]) -> Tuple[np.ndarray, List[Dict]]:
    """构建pattern的embedding表示（skeleton + tricks融合）"""
    print(f"\n🔢 构建pattern embeddings...")
    
    embeddings = []
    pattern_data = []
    
    for i, paper in enumerate(papers):
        if (i + 1) % 50 == 0:
            print(f"  进度: {i+1}/{len(papers)}")
        
        # 1. Skeleton文本
        skeleton = paper.get('skeleton', {})
        skeleton_text = " ".join([
            skeleton.get('problem_framing', ''),
            skeleton.get('gap_pattern', ''),
            skeleton.get('method_story', ''),
            skeleton.get('experiments_story', '')
        ])
        
        # 2. Tricks文本
        tricks = paper.get('tricks', [])
        tricks_text = " ".join([
            f"{t.get('name', '')}: {t.get('description', '')}"
            for t in tricks
        ])
        
        # 3. 分别计算embedding
        skeleton_emb = get_embedding(skeleton_text.strip())
        time.sleep(0.1)
        tricks_emb = get_embedding(tricks_text.strip())
        time.sleep(0.1)
        
        # 4. 加权融合
        skeleton_emb = np.array(skeleton_emb)
        tricks_emb = np.array(tricks_emb)
        pattern_emb = (CLUSTER_PARAMS['skeleton_weight'] * skeleton_emb + 
                      CLUSTER_PARAMS['tricks_weight'] * tricks_emb)
        
        embeddings.append(pattern_emb)
        pattern_data.append({
            'paper_id': paper.get('paper_id', ''),
            'title': paper.get('title', ''),
            'conference': paper.get('conference', ''),
            'skeleton': skeleton,
            'tricks': tricks,
            'skeleton_text': skeleton_text[:500],
            'tricks_text': tricks_text[:500]
        })
    
    return np.array(embeddings), pattern_data


def cluster_patterns(embeddings: np.ndarray) -> np.ndarray:
    """对patterns进行层次聚类（自适应距离阈值方式）"""
    print(f"\n🔄 开始聚类...")

    distance_threshold = CLUSTER_PARAMS['distance_threshold']
    print(f"  距离阈值: {distance_threshold}")

    clusterer = AgglomerativeClustering(
        n_clusters=None,                           # 改为动态
        distance_threshold=distance_threshold,     # 使用阈值
        metric='cosine',
        linkage='average'
    )

    labels = clusterer.fit_predict(embeddings)

    # 处理可能的 -1 标签（未聚类的论文）
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"  ✓ 生成 {n_clusters} 个 clusters")

    cluster_sizes = Counter(labels)
    print(f"  📊 Cluster 大小分布:")
    for cluster_id in sorted(cluster_sizes.keys()):
        if cluster_id != -1:
            print(f"    Cluster {cluster_id}: {cluster_sizes[cluster_id]} 篇")

    if -1 in cluster_sizes:
        print(f"    未聚类: {cluster_sizes[-1]} 篇")

    
    return labels


def analyze_cluster(cluster_papers: List[Dict], cluster_id: int) -> Dict:
    """分析单个cluster的特征"""
    print(f"\n  📊 分析 Cluster {cluster_id} ({len(cluster_papers)} 篇)...")
    
    # 1. 统计高频tricks
    trick_counter = Counter()
    trick_examples = defaultdict(list)
    
    for paper in cluster_papers:
        for trick in paper['tricks']:
            trick_name = trick.get('name', '')
            if not trick_name:
                continue
            
            trick_counter[trick_name] += 1
            trick_examples[trick_name].append({
                'paper_id': paper['paper_id'],
                'title': paper['title'],
                'description': trick.get('description', ''),
                'type': trick.get('type', ''),
                'purpose': trick.get('purpose', '')
            })
    
    # 2. 选择代表性skeleton例子（取最前面3个）
    skeleton_examples = []
    for paper in cluster_papers[:3]:
        skeleton_examples.append({
            'paper_id': paper['paper_id'],
            'title': paper['title'],
            'skeleton': paper['skeleton']
        })
    
    # 3. 计算coherence（类内平均相似度）
    # 这里简化处理，实际可以重新计算
    coherence = 0.75  # 占位值
    
    return {
        'cluster_id': cluster_id,
        'size': len(cluster_papers),
        'skeleton_examples': skeleton_examples,
        'trick_frequency': trick_counter.most_common(20),
        'trick_examples': trick_examples,
        'coherence': coherence,
        'all_papers': cluster_papers
    }


def generate_pattern_summary(cluster_analysis: Dict) -> str:
    """生成pattern总结（LLM）- 使用cluster完整信息"""
    
    all_papers = cluster_analysis['all_papers']
    cluster_size = cluster_analysis['size']
    
    # ============================================================
    # 1. 构建完整的 skeleton 信息（所有论文，完整四个维度）
    # ============================================================
    skeleton_info_list = []
    for i, paper in enumerate(all_papers[:8]):  # 最多取8篇避免token过长
        if not paper:
            continue
        skeleton = paper.get('skeleton') or {}
        title = paper.get('title') or 'Unknown Title'
        skeleton_info_list.append(f"""
论文{i+1}：《{str(title)[:60]}》
  - 问题定位：{skeleton.get('problem_framing', '')}
  - 研究缺口：{skeleton.get('gap_pattern', '')}
  - 方法叙述：{skeleton.get('method_story', '')}
  - 实验设计：{skeleton.get('experiments_story', '')}""")
    
    skeleton_full_text = "\n".join(skeleton_info_list)
    
    # ============================================================
    # 2. 构建完整的 tricks 信息（包含 name + description + purpose）
    # ============================================================
    tricks_info_list = []
    seen_tricks = set()  # 去重
    for paper in all_papers:
        if not paper:
            continue
        for trick in (paper.get('tricks') or []):
            if not trick:
                continue
            trick_name = trick.get('name', '')
            if trick_name and trick_name not in seen_tricks:
                seen_tricks.add(trick_name)
                tricks_info_list.append({
                    'name': trick_name,
                    'type': trick.get('type', ''),
                    'description': trick.get('description', ''),
                    'purpose': trick.get('purpose', ''),
                    'location': trick.get('location', '')
                })
    
    # 按频率统计，取前15个高频trick的完整信息
    trick_freq = cluster_analysis.get('trick_frequency', [])
    if not trick_freq:
        trick_freq = []
    
    top_trick_names = [item[0] for item in trick_freq[:15] if item and isinstance(item, (list, tuple))]

    tricks_full_list = []
    for trick_info in tricks_info_list:
        if trick_info['name'] in top_trick_names:
            tricks_full_list.append(
                f"- {trick_info['name']} [{trick_info['type']}]\n"
                f"    描述：{trick_info['description']}\n"
                f"    目的：{trick_info['purpose']}\n"
                f"    位置：{trick_info['location']}"
            )
    
    tricks_full_text = "\n".join(tricks_full_list[:10])  # 最多10个完整trick
    
    # ============================================================
    # 3. 统计信息（用于prompt参考）
    # ============================================================
    trick_stats = ", ".join([f"{name}({count}次)" for name, count in trick_freq[:10]])
    
    # ============================================================
    # 4. 构建完整的 prompt
    # ============================================================
    prompt = f"""
你是NLP研究专家。请基于以下cluster的完整信息，生成一个技术性总结。

【Cluster概览】
- 包含 {cluster_size} 篇论文
- 高频Tricks统计：{trick_stats}

【所有论文的Skeleton信息】
{skeleton_full_text}

【高频Tricks详细信息】
{tricks_full_text}

【任务要求】
请分析上述论文的共同特征，生成一个 150-200 字的技术性总结。

要求：
1. 找出这些论文在研究问题、方法设计、实验策略上的共性
2. 保留具体技术词（模型名、方法名、数据集名）
3. 突出这类论文的核心写作套路和技术特征
4. 避免空泛的描述，要有可操作的具体信息

【输出格式】（分三段）:
第1段（60字）：核心研究问题与技术路线 - 这类论文主要解决什么问题，用什么方法
第2段（60字）：关键技术组合与写作策略 - skeleton特点 + 常用tricks组合
第3段（60字）：适用场景与预期效果 - 什么任务/数据/目标适合这个套路

【示例对比】:
✅ 好: "针对跨语言理解任务中的数据稀缺问题，采用多语言预训练+零样本迁移的技术路线。
      skeleton通常以'低资源语言困境'开篇，通过'多语言对齐不足'指出gap，
      方法部分采用'对比学习+语言标签'组合。高频使用消融实验验证各组件贡献，
      配合多数据集验证增强泛化性。适用于低资源NLP任务，预期提升5-10%零样本性能。"

❌ 差: "这些论文采用问题导向的叙事结构，首先指出现有方法不足，然后提出创新方法..."

直接输出总结:
"""
    
    summary = call_llm(prompt)
    
    # 验证质量
    bad_patterns = ['叙事结构', '写作结构', '首先...接着', '这些论文采用']
    if len(summary) < 100 or any(bad in summary for bad in bad_patterns):
        print(f"    ⚠️  Summary质量不佳，尝试重新生成...")
        summary = call_llm(prompt)  # 再试一次
    
    return summary


def extract_pattern_name(summary: str) -> str:
    """从summary提取简短名称"""
    prompt = f"""
从以下pattern总结中提取一个简短的名称（不超过12个字）。

总结：{summary}

要求：
- 突出核心技术特征
- 简洁、专业
- 不要"XX研究"、"XX论文"等后缀

直接输出名称:
"""
    
    name = call_llm(prompt)
    return name.strip()


def generate_writing_guide_text(pattern_name: str, summary: str, skeleton_examples: List[Dict], 
                                common_tricks: List[Dict], cluster_size: int) -> str:
    """生成pattern的写作指导文本（给智能体用）"""
    
    guide_lines = []
    
    # 1. 模板聚焦
    guide_lines.extend([
        f"写作模板：{pattern_name}",
        "",
        "【模板聚焦】",
        summary,
        "",
    ])
    
    # 2. 骨架示例
    guide_lines.extend([
        "【代表性论文骨架示例】",
        f"该套路包含 {len(skeleton_examples)} 个代表性论文的骨架示例，可直观体现该模式的论文撰写框架：",
        ""
    ])
    
    for i, sk in enumerate(skeleton_examples):
        guide_lines.extend([
            f"示例 {i+1}：《{sk['title']}》",
            f"  • 问题定位：{compress_text(sk['problem_framing'], 150)}",
            f"  • 现有研究缺口：{compress_text(sk['gap_pattern'], 150)}",
            f"  • 核心方法：{compress_text(sk['method_story'], 150)}",
            f"  • 实验设计：{compress_text(sk['experiments_story'], 150)}",
            ""
        ])
    
    # 3. 高频技巧
    guide_lines.extend([
        "【高频研究技巧】",
        f"该模式下有以下 {len(common_tricks)} 个高频使用的研究技巧：",
        ""
    ])
    
    for i, trick in enumerate(common_tricks[:10]):
        example = trick['examples'][0] if trick['examples'] else {}
        guide_lines.extend([
            f"{i+1}. {trick['trick_name']}（使用频率 {trick['frequency']} 次，占比 {trick['percentage']}）",
            f"   类型：{example.get('type', '通用技巧')}",
            f"   应用：{compress_text(example.get('description', ''), 150)}",
            ""
        ])
    
    return "\n".join(guide_lines)


def assemble_pattern(cluster_analysis: Dict, summary: str) -> Dict:
    """组装最终的pattern结构"""
    
    pattern_name = extract_pattern_name(summary)
    
    # Skeleton例子
    skeleton_examples = [
        {
            'paper_id': sk['paper_id'],
            'title': sk['title'],
            'problem_framing': sk['skeleton'].get('problem_framing', ''),
            'gap_pattern': sk['skeleton'].get('gap_pattern', ''),
            'method_story': sk['skeleton'].get('method_story', ''),
            'experiments_story': sk['skeleton'].get('experiments_story', '')
        }
        for sk in cluster_analysis['skeleton_examples']
    ]
    
    # 高频Tricks
    common_tricks = [
        {
            'trick_name': name,
            'frequency': count,
            'percentage': f"{count/cluster_analysis['size']*100:.1f}%",
            'examples': cluster_analysis['trick_examples'][name][:3]
        }
        for name, count in cluster_analysis['trick_frequency'][:15]
    ]
    
    # 生成写作指导文本
    writing_guide = generate_writing_guide_text(
        pattern_name, summary, skeleton_examples, common_tricks, 
        cluster_analysis['size']
    )
    
    return {
        'pattern_id': cluster_analysis['cluster_id'],
        'pattern_name': pattern_name,
        'pattern_summary': summary,
        
        # 新增：完整的写作指导文本（给智能体用）
        'writing_guide': writing_guide,
        
        # Skeleton例子
        'skeleton_examples': skeleton_examples,
        
        # 高频Tricks
        'common_tricks': common_tricks,
        
        # 元数据
        'metadata': {
            'cluster_size': cluster_analysis['size'],
            'coherence_score': cluster_analysis['coherence'],
            'all_paper_ids': [p['paper_id'] for p in cluster_analysis['all_papers']]
        }
    }


def compress_text(text: str, max_len: int = 100) -> str:
    """压缩文本到指定长度"""
    if len(text) <= max_len:
        return text
    
    sentences = text.split('。')
    compressed = ""
    for sent in sentences:
        if len(compressed) + len(sent) + 1 <= max_len - 3:
            compressed += sent + "。"
        else:
            break
    
    return compressed if compressed else text[:max_len-3] + "..."


def generate_user_guide(patterns: List[Dict]) -> str:
    """生成用户指导文档"""
    print(f"\n📝 生成用户指导文档...")
    
    # 1. 整体介绍
    total_papers = sum(p['metadata']['cluster_size'] for p in patterns)
    
    guide_lines = [
        "="*80,
        "NLP 论文写作模式（Patterns）指南",
        "="*80,
        "",
        "【整体介绍】",
        "",
        f"本指南基于 {total_papers} 篇 NLP 顶会论文的深度分析，通过对论文骨架（skeleton）和",
        f"研究技巧（tricks）的聚类，抽象出 {len(patterns)} 个可复用的写作模式（patterns）。",
        "",
        "每个 pattern 包含：",
        "  • 模式总结：该类论文的核心技术路线和写作特点",
        "  • 骨架示例：2-3 篇代表性论文的完整结构框架",
        "  • 高频技巧：统计排序的常用研究技巧及使用频率",
        "  • 使用建议：针对性的写作和研究建议",
        "",
        "【如何使用本指南】",
        "",
        "1️⃣  定位你的研究类型",
        "   - 浏览各个 pattern 的【模板聚焦】部分",
        "   - 找到与你研究最相关的 1-2 个 patterns",
        "",
        "2️⃣  学习论文结构",
        "   - 参考【代表性论文骨架示例】",
        "   - 理解问题定位、缺口分析、方法叙述、实验设计的逻辑",
        "",
        "3️⃣  选择合适技巧",
        "   - 查看【高频研究技巧】列表",
        "   - 根据使用频率和适用场景，选择 3-5 个技巧应用到你的论文",
        "",
        "4️⃣  追溯具体论文",
        "   - 通过【相关论文】列表，找到具体论文深度学习",
        "",
        "【Pattern 列表】",
        ""
    ]
    
    # 2. Pattern目录
    for p in patterns:
        guide_lines.append(
            f"  Pattern #{p['pattern_id']:02d} - {p['pattern_name']} "
            f"({p['metadata']['cluster_size']}篇论文)"
        )
    
    guide_lines.extend(["", "="*80, ""])
    
    # 3. 每个Pattern的详细信息
    for pattern in patterns:
        guide_lines.extend([
            "="*80,
            f"写作模板 #{pattern['pattern_id']}：{pattern['pattern_name']}",
            "="*80,
            "",
            "【模板聚焦】",
            pattern['pattern_summary'],
            "",
            "-"*80,
            "【代表性论文骨架示例】",
            "-"*80,
            "",
            f"该套路包含 {len(pattern['skeleton_examples'])} 个代表性论文的骨架示例，可直观体现该模式的论文撰写框架：",
            ""
        ])
        
        # Skeleton例子
        for sk in pattern['skeleton_examples']:
            guide_lines.extend([
                f"📄 论文标题：《{sk['title']}》",
                "",
                f"   • 问题定位：{compress_text(sk['problem_framing'], 120)}",
                "",
                f"   • 现有研究缺口：{compress_text(sk['gap_pattern'], 120)}",
                "",
                f"   • 核心方法：{compress_text(sk['method_story'], 120)}",
                "",
                f"   • 实验设计：{compress_text(sk['experiments_story'], 120)}",
                ""
            ])
        
        # Tricks
        guide_lines.extend([
            "-"*80,
            "【高频研究技巧】",
            "-"*80,
            "",
            f"该模式下梳理出以下 {len(pattern['common_tricks'])} 个高频使用的研究技巧，含使用频率、占比及具体示例：",
            ""
        ])
        
        for i, trick in enumerate(pattern['common_tricks'][:10]):
            example = trick['examples'][0] if trick['examples'] else {}
            guide_lines.extend([
                f"{i+1}. {trick['trick_name']}",
                f"   - 使用频率：{trick['frequency']} 次（占比 {trick['percentage']}）",
                f"   - 技巧类型：{example.get('type', '通用技巧')}",
                f"   - 典型应用：{compress_text(example.get('description', ''), 150)}",
                ""
            ])
        
        # 相关论文
        paper_ids = pattern['metadata']['all_paper_ids']
        guide_lines.extend([
            "-"*80,
            f"【相关论文】（共 {len(paper_ids)} 篇）",
            "-"*80
        ])
        
        for i, paper_id in enumerate(paper_ids[:15]):
            guide_lines.append(f"  [{i+1}] {paper_id}")
        
        if len(paper_ids) > 15:
            guide_lines.append(f"  ... 及其他 {len(paper_ids) - 15} 篇")
        
        guide_lines.extend(["", "="*80, ""])
    
    return "\n".join(guide_lines)


def generate_statistics(patterns: List[Dict]) -> Dict:
    """生成统计报告"""
    print(f"\n📊 生成统计报告...")
    
    # 全局trick统计
    all_tricks = Counter()
    for pattern in patterns:
        for trick in pattern['common_tricks']:
            all_tricks[trick['trick_name']] += trick['frequency']
    
    # 聚类质量统计
    cluster_sizes = [p['metadata']['cluster_size'] for p in patterns]
    
    return {
        'total_patterns': len(patterns),
        'total_papers': sum(cluster_sizes),
        'average_cluster_size': float(np.mean(cluster_sizes)),
        'median_cluster_size': float(np.median(cluster_sizes)),
        'cluster_size_distribution': {
            'min': min(cluster_sizes),
            'max': max(cluster_sizes),
            'std': float(np.std(cluster_sizes))
        },
        'top_global_tricks': [
            {'name': name, 'total_count': count}
            for name, count in all_tricks.most_common(20)
        ],
        'pattern_size_distribution': {
            'small (<10)': len([s for s in cluster_sizes if s < 10]),
            'medium (10-20)': len([s for s in cluster_sizes if 10 <= s < 20]),
            'large (20-30)': len([s for s in cluster_sizes if 20 <= s < 30]),
            'xlarge (>=30)': len([s for s in cluster_sizes if s >= 30])
        }
    }


def main():
    """主流程"""
    print("="*80)
    print("基于 Skeleton + Tricks 聚类生成 Patterns")
    print("="*80)
    
    # 1. 加载论文
    print("\n【Step 1】加载论文数据")
    papers = load_all_papers()
    print(f"✅ 共加载 {len(papers)} 篇论文")
    
    # 2. 构建pattern embeddings
    print("\n【Step 2】构建pattern embeddings")
    embeddings, pattern_data = build_pattern_embeddings(papers)
    print(f"✅ 完成 {len(embeddings)} 个pattern的embedding")
    
    # 3. 聚类
    print("\n【Step 3】聚类")
    labels = cluster_patterns(embeddings)
    
    # 4. 分析每个cluster并生成pattern
    print("\n【Step 4】生成patterns")
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    patterns = []
    
    # 创建原始 cluster_id -> 新 pattern_id 的映射
    old_to_new_id = {}  # old_cluster_id -> new_pattern_id (1-based)
    new_pattern_id = 1

    # 创建 paper_id -> pattern_id 的映射
    paper_to_pattern = {}  # paper_id -> new_pattern_id

    for cluster_id in sorted(set(labels)):
        if cluster_id == -1:  # 跳过未聚类的论文
            print(f"  ⚠️  Cluster -1: {(labels == -1).sum()}篇 (未聚类，已跳过)")
            continue

        cluster_indices = [i for i in range(len(labels)) if labels[i] == cluster_id]

        if len(cluster_indices) < CLUSTER_PARAMS['min_cluster_size']:
            print(f"  ⚠️  Cluster {cluster_id}: {len(cluster_indices)}篇 (过小，跳过)")
            continue
        
        cluster_papers = [pattern_data[i] for i in cluster_indices]
        
        # 建立旧 ID 到新 ID 的映射
        old_to_new_id[cluster_id] = new_pattern_id
        
        # 记录这个 cluster 中所有 paper 的映射
        for idx in cluster_indices:
            paper_id = pattern_data[idx]['paper_id']
            paper_to_pattern[paper_id] = new_pattern_id

        # 分析cluster（使用新的 pattern_id）
        cluster_analysis = analyze_cluster(cluster_papers, new_pattern_id)

        # 生成summary
        summary = generate_pattern_summary(cluster_analysis)
        print(f"    Pattern {new_pattern_id}: {summary[:80]}...")
        
        # 组装pattern
        pattern = assemble_pattern(cluster_analysis, summary)
        patterns.append(pattern)
    
        new_pattern_id += 1

    print(f"\n✅ 共生成 {len(patterns)} 个patterns")
    
    # 5. 生成输出文件
    print("\n【Step 5】生成输出文件")
    
    # 获取输出目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(os.path.dirname(script_dir), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # 5.1 结构化JSON
    with open(os.path.join(output_dir, 'patterns_structured.json'), 'w', encoding='utf-8') as f:
        json.dump(patterns, f, ensure_ascii=False, indent=2)
    print("  ✅ patterns_structured.json")
    
    # 5.2 保存 Paper → Pattern 映射（供后续使用）
    with open(os.path.join(output_dir, 'paper_to_pattern.json'), 'w', encoding='utf-8') as f:
        json.dump(paper_to_pattern, f, ensure_ascii=False, indent=2)
    print(f"  ✅ paper_to_pattern.json ({len(paper_to_pattern)} 篇论文)")

    # 5.2 用户指导
    guide_text = generate_user_guide(patterns)
    with open(os.path.join(output_dir, 'patterns_guide.txt'), 'w', encoding='utf-8') as f:
        f.write(guide_text)
    print("  ✅ patterns_guide.txt")
    
    # 5.3 统计报告
    statistics = generate_statistics(patterns)
    with open(os.path.join(output_dir, 'patterns_statistics.json'), 'w', encoding='utf-8') as f:
        json.dump(statistics, f, ensure_ascii=False, indent=2)
    print("  ✅ patterns_statistics.json")
    
    print("\n" + "="*80)
    print("🎉 完成！")
    print("="*80)
    print(f"\n生成了 {len(patterns)} 个patterns，覆盖 {statistics['total_papers']} 篇论文")
    print(f"平均每个pattern包含 {statistics['average_cluster_size']:.1f} 篇论文")
    print(f"\n输出文件：")
    print(f"  1. patterns_structured.json  - 结构化数据（给程序用）")
    print(f"  2. patterns_guide.txt        - 用户指导文档（给人看）")
    print(f"  3. patterns_statistics.json  - 统计报告")


if __name__ == '__main__':
    main()
