# 召回系统总结

## 📦 已完成的工作

### 1. 召回方案设计

设计了基于知识图谱的**三路召回策略**：

| 路径 | 召回逻辑 | 权重 | Top-K | 优势 |
|------|---------|------|-------|------|
| **路径1** | Idea → Idea → Pattern<br>(相似Idea召回) | 0.4 | 10个Idea | 直接利用历史成功经验 |
| **路径2** | Idea → Domain → Pattern<br>(领域相关性召回) | 0.3 | 5个Domain | 领域泛化能力强 |
| **路径3** | Idea → Paper → Pattern<br>(相似Paper召回) | 0.3 | 20个Paper | 兼顾质量和细粒度匹配 |

**融合策略**: 加权线性融合，最终返回 Top-10 Pattern

---

## 🔍 召回流程详解

### 路径1: 相似Idea召回

```
用户输入Idea
    ↓ [实时计算Jaccard相似度]
图谱中的Idea (Top-10)
    ↓ [通过source_paper_ids]
Paper (多个)
    ↓ [uses_pattern边，权重=quality]
Pattern (多个)
    ↓ [累加得分] score = similarity × quality
Pattern得分字典
```

**召回数量**: 预期 10-30 个Pattern

**权重定义**:
$$
\text{score}(p) = \sum_{i \in \text{Top-10 Ideas}} \text{sim}(user, i) \times \text{quality}(paper_i)
$$

---

### 路径2: 领域相关性召回

```
用户输入Idea
    ↓ [关键词匹配 或 通过相似Idea]
Domain (Top-5)
    ↓ [works_well_in边，权重=effectiveness, confidence]
    ↓ (反向查找predecessors)
Pattern (多个)
    ↓ [累加得分] score = domain_weight × effectiveness × confidence
Pattern得分字典
```

**召回数量**: 预期 20-40 个Pattern

**权重定义**:
$$
\text{score}(p) = \sum_{d \in \text{Top-5 Domains}} w_d \times \max(\text{eff}_{p,d}, 0.1) \times \text{conf}_{p,d}
$$

---

### 路径3: 相似Paper召回

```
用户输入Idea
    ↓ [实时计算相似度 + 质量过滤]
Paper (Top-20)
    ↓ [uses_pattern边，权重=quality]
Pattern (多个)
    ↓ [累加得分] score = similarity × paper_quality × pattern_quality
Pattern得分字典
```

**召回数量**: 预期 20-50 个Pattern

**权重定义**:
$$
\text{score}(p) = \sum_{paper \in \text{Top-20}} \text{sim}(user, paper) \times \text{qual}(paper) \times \text{qual}_p
$$

---

## 📊 参数配置

### 召回数量

| 参数 | 路径1 | 路径2 | 路径3 | 最终 |
|------|-------|-------|-------|------|
| **Top-K** | 10 Ideas | 5 Domains | 20 Papers | 10 Patterns |
| **每个保留** | 5 Patterns | 10 Patterns | 8 Patterns | - |

### 路径权重

- **路径1**: 0.4 (最高，直接有效)
- **路径2**: 0.3 (领域泛化)
- **路径3**: 0.3 (质量导向)

### 相似度阈值

- **Idea相似度**: 无硬阈值，取Top-K
- **Paper相似度**: ≥ 0.1

---

## 🎯 边游走概率 (排序)

### 如何确定边游走的概率？

每条路径中，**边的权重**直接影响Pattern的最终得分：

#### 路径1的权重传递

```
用户Idea
  → 相似度(0-1)
  → Idea节点
  → source_paper_ids (静态列表)
  → Paper节点
  → [uses_pattern] quality=0.8 ← 边权重
  → Pattern节点
```

**得分累加**: 每个Pattern的得分 = Σ(相似度 × Paper质量)

**排序依据**: 得分越高，Pattern越相关

#### 路径2的权重传递

```
用户Idea
  → Domain匹配度(0-1)
  → Domain节点
  → [works_well_in] effectiveness=0.15, confidence=0.8 ← 边权重
  → Pattern节点
```

**得分累加**: 每个Pattern的得分 = Σ(Domain权重 × effectiveness × confidence)

**排序依据**:
- `effectiveness` > 0: Pattern在该领域效果**优于基线**
- `confidence` → 1: 样本数越多，越可信

#### 路径3的权重传递

```
用户Idea
  → 相似度(0-1) × Paper质量(0-1)
  → Paper节点
  → [uses_pattern] quality=0.9 ← 边权重
  → Pattern节点
```

**得分累加**: 每个Pattern的得分 = Σ(相似度 × Paper质量 × Pattern质量)

**排序依据**: 同时考虑Paper与用户Idea的匹配度和Paper/Pattern的质量

---

## 💻 已实现的代码

### 核心文件

| 文件 | 功能 | 代码量 |
|------|------|--------|
| `scripts/recall_system.py` | 完整召回系统（含4个测试用例） | ~450行 |
| `scripts/simple_recall_demo.py` | 简化版Demo（单个测试） | ~250行 |
| `scripts/test_recall.py` | 核心功能测试脚本 | ~100行 |

### 文档文件

| 文件 | 内容 |
|------|------|
| `docs/RECALL_SYSTEM.md` | 召回系统设计文档（算法详解） |
| `docs/RECALL_USAGE.md` | 使用指南（API、场景、FAQ） |
| `docs/RECALL_SUMMARY.md` | 本文档（总结） |

---

## 🚀 使用方法

### 方法1: 完整召回系统

```bash
# 运行4个测试用例的完整Demo
python scripts/recall_system.py
```

### 方法2: 简化单次召回

```bash
# 自定义Idea进行召回
python scripts/simple_recall_demo.py "你的Idea描述"

# 示例
python scripts/simple_recall_demo.py "使用Transformer进行文本分类，在GLUE数据集上验证"
```

### 方法3: Python API

```python
from recall_system import RecallSystem

# 初始化
system = RecallSystem()

# 召回
user_idea = "提出新的注意力机制改进神经机器翻译"
results = system.recall(user_idea, verbose=True)

# 处理结果
for rank, (pattern_id, info, score) in enumerate(results, 1):
    print(f"{rank}. {info['name']} (得分={score:.4f})")
```

---

## 📈 召回效果

### 预期召回量

| 路径 | 召回Pattern数 | 去重后 |
|------|--------------|--------|
| 路径1 | 10-30 | - |
| 路径2 | 20-40 | - |
| 路径3 | 20-50 | - |
| **融合后** | - | **30-80** |
| **最终Top-K** | - | **10** |

### 评估维度

| 维度 | 路径1 | 路径2 | 路径3 | 综合 |
|------|-------|-------|-------|------|
| **覆盖率** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **准确性** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **多样性** | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |

---

## 🔧 可调参数

### 快速调整

```python
from recall_system import RecallConfig

# 增加召回量
RecallConfig.PATH1_TOP_K_IDEAS = 20     # 默认10
RecallConfig.PATH3_TOP_K_PAPERS = 50    # 默认20
RecallConfig.FINAL_TOP_K = 20           # 默认10

# 调整路径权重（追求质量）
RecallConfig.PATH1_WEIGHT = 0.3
RecallConfig.PATH2_WEIGHT = 0.2
RecallConfig.PATH3_WEIGHT = 0.5  # ← 提高Paper路径权重
```

### 适用场景

| 场景 | 推荐权重 | 理由 |
|------|---------|------|
| **保守写作** | PATH1=0.6, PATH2=0.2, PATH3=0.2 | 优先历史成功经验 |
| **新颖Idea** | PATH1=0.2, PATH2=0.5, PATH3=0.3 | 提高领域泛化能力 |
| **追求高质量** | PATH1=0.2, PATH2=0.2, PATH3=0.6 | 偏向顶会论文套路 |

---

## ✅ 设计亮点

### 1. 三路互补

- **路径1**: 历史经验，高准确性
- **路径2**: 领域泛化，高覆盖率
- **路径3**: 质量导向，高可信度

三路融合，兼顾**准确性、覆盖率、多样性**

### 2. 权重可解释

每个Pattern的得分都可以追溯来源：

```
Pattern_5: 得分 0.285
  - 路径1贡献: 0.180 (63.2%) ← 主要来自历史相似Idea
  - 路径2贡献: 0.060 (21.1%)
  - 路径3贡献: 0.045 (15.8%)
```

用户可以理解**为什么推荐这个Pattern**

### 3. 灵活可扩展

- **相似度算法**: 当前Jaccard，可升级为BERT嵌入
- **领域识别**: 当前关键词匹配，可升级为分类模型
- **权重策略**: 当前固定权重，可引入强化学习动态调整

### 4. 实时 + 预计算结合

- **实时计算**: 用户Idea与图谱的相似度（路径1、路径3）
- **预计算边**: Domain-Pattern效果（路径2）

兼顾**响应速度**和**召回质量**

---

## 🎨 Mock测试用例

系统内置了4个测试用例，覆盖不同类型的Idea：

### 测试1: Transformer文本分类

```
"使用Transformer模型进行文本分类任务，在多个数据集上验证效果"
```

**特点**: 常见任务，历史经验丰富
**预期**: 路径1得分高

### 测试2: 注意力机制改进翻译

```
"提出一种新的注意力机制改进神经机器翻译的对齐质量"
```

**特点**: 方法创新，领域明确（NLP/MT）
**预期**: 路径2得分高

### 测试3: 对抗训练提升鲁棒性

```
"通过对抗训练提升模型在对话系统中的鲁棒性"
```

**特点**: 安全性研究，质量要求高
**预期**: 路径3得分高

### 测试4: 知识图谱增强语义理解

```
"利用知识图谱增强预训练语言模型的语义理解能力"
```

**特点**: 跨领域融合（KG + PLM）
**预期**: 三路均衡

---

## 📚 相关文档

- [EDGE_TYPES.md](EDGE_TYPES.md) - 知识图谱边类型说明
- [RECALL_SYSTEM.md](RECALL_SYSTEM.md) - 召回系统设计文档
- [RECALL_USAGE.md](RECALL_USAGE.md) - 使用指南和API文档

---

## 🔮 未来优化方向

### 短期 (1-2周)

- [ ] 升级相似度计算: Jaccard → Sentence-BERT
- [ ] 增加领域分类器: 关键词匹配 → 神经网络分类
- [ ] 添加缓存机制: 缓存常见查询结果

### 中期 (1-2个月)

- [ ] 引入用户反馈: 点击/使用数据调整权重
- [ ] Pattern特征增强: 任务类型、技术栈标签
- [ ] 多样性约束: MMR算法避免Pattern重复

### 长期 (3-6个月)

- [ ] 端到端召回模型: 深度学习Idea → Pattern映射
- [ ] 强化学习优化: 基于用户满意度优化策略
- [ ] 知识图谱补全: 链接预测补充缺失边

---

**总结人**: CatPaw AI
**完成时间**: 2026-01-08
**代码总量**: ~800行 + 文档
**核心功能**: ✅ 三路召回 + ✅ 权重融合 + ✅ 可解释性

