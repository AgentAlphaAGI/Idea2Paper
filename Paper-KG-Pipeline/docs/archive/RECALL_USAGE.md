# 召回系统使用指南

## 快速开始

### 1. 准备工作

确保已经完成知识图谱构建：

```bash
# 构建节点
python scripts/build_entity.py

# 构建边（包含召回所需的边）
python scripts/build_edges.py
```

这会生成以下文件：
- `output/nodes_idea.json`
- `output/nodes_pattern.json`
- `output/nodes_domain.json`
- `output/nodes_paper.json`
- `output/edges.json`
- `output/knowledge_graph_v2.gpickle` ⭐ 召回系统需要

### 2. 运行测试

验证召回系统核心功能：

```bash
python scripts/test_recall.py
```

### 3. 运行完整Demo

体验三路召回的完整流程：

```bash
python scripts/recall_system.py
```

Demo包含4个测试用例：
1. Transformer文本分类
2. 注意力机制改进翻译
3. 对抗训练提升鲁棒性
4. 知识图谱增强语义理解

---

## 编程接口

### 基本使用

```python
from recall_system import RecallSystem

# 1. 初始化系统
system = RecallSystem()

# 2. 输入Idea，召回Pattern
user_idea = "使用Transformer模型进行文本分类任务，在多个数据集上验证效果"
results = system.recall(user_idea, verbose=True)

# 3. 处理结果
for rank, (pattern_id, pattern_info, score) in enumerate(results, 1):
    print(f"[{rank}] {pattern_info['name']}")
    print(f"    得分: {score:.4f}")
    print(f"    聚类大小: {pattern_info['cluster_size']} 篇论文")
    print(f"    摘要: {pattern_info['summary'][:100]}...")
```

### 结果格式

`system.recall()` 返回一个列表，每个元素是一个三元组：

```python
(pattern_id, pattern_info, score)
```

其中：
- `pattern_id` (str): Pattern的唯一ID，如 "pattern_5"
- `pattern_info` (dict): Pattern的完整信息
  - `name` (str): Pattern名称
  - `summary` (str): Pattern摘要
  - `writing_guide` (str): 写作指南（包含骨架示例和技巧）
  - `cluster_size` (int): 聚类包含的论文数量
  - `coherence_score` (float): 聚类的连贯性分数
  - `paper_ids` (list): 属于该Pattern的论文ID列表
  - `top_tricks` (list): 高频使用的研究技巧
- `score` (float): 综合召回得分（三路加权融合后）

### 配置参数

可以通过修改 `RecallConfig` 类来调整召回参数：

```python
from recall_system import RecallConfig

# 调整每路召回的数量
RecallConfig.PATH1_TOP_K_IDEAS = 15      # 默认10
RecallConfig.PATH2_TOP_K_DOMAINS = 8     # 默认5
RecallConfig.PATH3_TOP_K_PAPERS = 30     # 默认20

# 调整路径权重
RecallConfig.PATH1_WEIGHT = 0.5  # 默认0.4
RecallConfig.PATH2_WEIGHT = 0.25 # 默认0.3
RecallConfig.PATH3_WEIGHT = 0.25 # 默认0.3

# 调整最终返回数量
RecallConfig.FINAL_TOP_K = 15    # 默认10
```

---

## 使用场景

### 场景1: 论文写作辅助

**需求**: 用户有一个研究Idea，想知道如何撰写论文

```python
from recall_system import RecallSystem

system = RecallSystem()

user_idea = """
我想研究如何用大语言模型进行代码生成，通过few-shot prompting
在多个编程语言上进行实验，对比不同prompt设计的效果
"""

results = system.recall(user_idea)

# 获取推荐的写作套路
print("推荐的写作套路：")
for i, (pid, info, score) in enumerate(results[:3], 1):
    print(f"\n{i}. {info['name']}")
    print(f"   适用论文数: {info['cluster_size']}")
    print(f"   写作指南预览:")
    print(info['writing_guide'][:300] + "...")
```

### 场景2: 方法对比分析

**需求**: 分析不同Idea的写作套路差异

```python
ideas = [
    "用CNN进行图像分类",
    "用Transformer进行图像分类",
    "用Diffusion Model进行图像生成"
]

for idea in ideas:
    results = system.recall(idea, verbose=False)
    top_pattern = results[0][1] if results else None
    print(f"\nIdea: {idea}")
    print(f"  → 推荐: {top_pattern['name'] if top_pattern else 'N/A'}")
```

### 场景3: 领域热门套路挖掘

**需求**: 发现某个领域最常用的写作套路

```python
# 方法1: 通过领域关键词
domain_idea = "自然语言处理 NLP 文本理解"
results = system.recall(domain_idea)

# 路径2会返回该领域中效果好的Pattern
print("NLP领域热门套路:")
for pid, info, score in results[:5]:
    print(f"  - {info['name']} (聚类大小={info['cluster_size']})")

# 方法2: 直接查询图谱（高级用法）
import networkx as nx
for domain in system.domains:
    if "NLP" in domain['name'] or "自然语言" in domain['name']:
        domain_id = domain['domain_id']
        # 找到在该领域表现好的Pattern
        for pred in system.G.predecessors(domain_id):
            edge = system.G[pred][domain_id]
            if edge.get('relation') == 'works_well_in':
                pattern = system.pattern_id_to_pattern.get(pred)
                effectiveness = edge.get('effectiveness', 0)
                print(f"  {pattern['name']}: 效果增益={effectiveness:.3f}")
```

---

## 召回结果解读

### 得分的含义

召回得分综合考虑三个维度：

1. **相似度** (路径1, 权重0.4)
   - 用户Idea与历史Idea的语义相似度
   - **高分**: 与成功案例高度相似，适合参考
   - **低分**: 较新颖，需要创新

2. **领域效果** (路径2, 权重0.3)
   - Pattern在相关领域的历史效果
   - **高分**: 在该领域验证有效
   - **低分**: 在该领域较少使用或效果一般

3. **质量匹配** (路径3, 权重0.3)
   - 相似高质量Paper使用的Pattern
   - **高分**: 顶会论文常用
   - **低分**: 较少被高质量论文使用

### 各路得分占比

查看每个Pattern的得分来源：

```python
results = system.recall(user_idea, verbose=True)

# verbose=True时会打印类似：
# [Rank 1] pattern_5
#   最终得分: 0.2850
#   - 路径1 (相似Idea):   0.1800 (占比 63.2%)  ← 主要来源
#   - 路径2 (领域相关):   0.0600 (占比 21.1%)
#   - 路径3 (相似Paper):  0.0450 (占比 15.8%)
```

**解读**:
- **路径1占比高**: 历史上有很多相似Idea使用该Pattern，稳妥但可能不够新颖
- **路径2占比高**: 该Pattern在相关领域表现突出，领域通用性强
- **路径3占比高**: 高质量Paper偏好该Pattern，质量有保障

---

## 常见问题

### Q1: 为什么召回结果很少？

**可能原因**:
1. 图谱中没有足够相似的Idea/Paper
2. 用户Idea过于新颖，领域匹配不上
3. Pattern数量本身较少（如图谱只有30个Pattern）

**解决方案**:
- 调大 `RecallConfig.PATH1_TOP_K_IDEAS` 等参数
- 放宽相似度阈值（修改代码中的0.1阈值）
- 扩充图谱数据

### Q2: 如何提升召回准确性？

**方法**:
1. **升级相似度计算**: 从Jaccard改为BERT嵌入相似度
2. **优化领域识别**: 使用分类模型而非关键词匹配
3. **引入用户反馈**: 根据用户点击/使用情况调整权重

### Q3: 召回速度慢怎么办？

**优化方案**:
1. **预计算Idea嵌入**: 启动时一次性计算所有Idea的嵌入向量
2. **使用向量索引**: 用Faiss/Annoy等库加速相似度检索
3. **缓存结果**: 对常见查询进行缓存

### Q4: 如何调整三路权重？

根据实际需求调整：

```python
# 场景1: 优先历史经验（保守策略）
RecallConfig.PATH1_WEIGHT = 0.6  # 相似Idea
RecallConfig.PATH2_WEIGHT = 0.2  # 领域相关
RecallConfig.PATH3_WEIGHT = 0.2  # 相似Paper

# 场景2: 优先领域通用性（新颖Idea）
RecallConfig.PATH1_WEIGHT = 0.2
RecallConfig.PATH2_WEIGHT = 0.5  # ← 提高
RecallConfig.PATH3_WEIGHT = 0.3

# 场景3: 优先质量（追求高影响力）
RecallConfig.PATH1_WEIGHT = 0.2
RecallConfig.PATH2_WEIGHT = 0.2
RecallConfig.PATH3_WEIGHT = 0.6  # ← 提高
```

---

## 下一步

- 阅读 [召回系统设计文档](RECALL_SYSTEM.md) 了解算法细节
- 阅读 [边类型说明](EDGE_TYPES.md) 了解图谱结构
- 修改 `recall_system.py` 自定义召回策略

---

**更新时间**: 2026-01-08

