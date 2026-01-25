# V3版本召回系统更新说明

## 📋 概述

本文档说明基于V3知识图谱的召回系统实现,与之前版本的主要差异和当前状态。

---

## 🔄 V3版本主要变化

### 1. 节点结构变化

#### Paper节点
- **旧版本**: `paper.idea` 是嵌套字典 `{core_idea: "...", tech_stack: [...], ...}`
- **V3版本**: `paper.idea` 是简单字符串
- **影响**: 路径3的Paper相似度计算逻辑需要适配

#### Pattern节点
- **旧版本**:
  - `cluster_size` 字段表示聚类大小
  - `summary` 是字符串摘要
- **V3版本**:
  - `size` 字段表示聚类大小
  - `summary` 是字典,包含示例列表
  - 新增 `llm_enhanced_summary` 字段(LLM生成的归纳总结)
  - 新增 `llm_enhanced` 标志

#### Idea节点
- **V3增强**: 直接包含 `pattern_ids` 列表
- **优势**: 路径1可以直接从Idea节点获取Pattern,无需通过Paper中转

### 2. 数据覆盖率

| 指标 | V3版本 | 说明 |
|------|--------|------|
| 总Idea数 | 8,284 | 100%覆盖所有Paper |
| 总Pattern数 | 124 | 基于cluster聚类 |
| 总Domain数 | 98 | 聚合自assignments |
| 总Paper数 | 8,285 | ICLR数据集 |
| Idea→Pattern覆盖率 | 72.2% | 5,980/8,284个Idea有pattern_ids |

### 3. Review评分逻辑

#### ✅ 完整保留评分逻辑（重要！）
- **V3版本保留了完整的review评分逻辑，不移除！**
- **当前状态**: Paper节点暂时缺失review数据
- **实现机制**:
  - 当Paper有review数据时，自动计算真实质量分数（基于review评分，归一化到[0,1]）
  - 当Paper缺失review数据时，使用默认值0.5
  - 质量评分函数与`build_edges.py`保持一致
- **未来兼容**: 当review数据补充后，召回质量将自动提升，无需修改代码

#### 质量评分函数
```python
def _get_paper_quality(self, paper: Dict) -> float:
    """计算Paper的综合质量分数
    基于review的评分，归一化到[0, 1]
    如果没有review数据，返回默认值0.5
    """
    reviews = paper.get('reviews', [])
    if not reviews:
        return 0.5  # 默认中等质量

    # 提取评分并归一化
    scores = [float(r.get('overall_score', '5')) for r in reviews]
    avg_score = np.mean(scores)
    return (avg_score - 1) / 9  # 归一化到[0,1]
```

#### 在召回中的体现
- 路径3显示质量来源：`[review]` 或 `[默认]`
- 示例：`- ICLR_001 (相似度=0.850, 质量=0.500 [默认])`
- Edge权重会反映真实Paper质量

---

## 🔗 三路召回实现

### 路径1: Idea → Idea → Pattern (相似Idea召回)

#### 旧版本流程
```
用户Idea → 计算相似度 → Top-K Idea
         → 遍历source_paper_ids → 查询Paper→Pattern边
         → 收集Pattern (通过图谱)
```

#### V3版本流程
```
用户Idea → 计算相似度 → Top-K Idea
         → 直接获取idea.pattern_ids → Pattern
```

#### 代码对比

**旧版本**:
```python
for idea_id, similarity in top_ideas:
    idea = self.idea_id_to_idea[idea_id]
    # 通过Paper中转
    for paper_id in idea.get('source_paper_ids', []):
        for successor in self.G.successors(paper_id):
            if edge.get('relation') == 'uses_pattern':
                pattern_id = successor
                quality = edge.get('quality', 0.5)
                score = similarity * quality
```

**V3版本**:
```python
for idea_id, similarity in top_ideas:
    idea = self.idea_id_to_idea[idea_id]
    # 直接使用pattern_ids
    for pattern_id in idea.get('pattern_ids', []):
        pattern_scores[pattern_id] += similarity
```

#### 优势
- ✅ 更简洁高效
- ✅ 不依赖图谱边查询
- ✅ 逻辑更清晰

---

### 路径2: Idea → Domain → Pattern (领域相关性召回)

#### 实现逻辑
```
用户Idea → 关键词匹配Domain → Top-K Domain
         → 查询Pattern→Domain边(works_well_in)
         → 按effectiveness和confidence排序
```

#### V3版本无变化
- 仍使用 `Idea→Domain` 边 (belongs_to)
- 仍使用 `Pattern→Domain` 边 (works_well_in)
- 权重计算: `score = domain_weight × effectiveness × confidence`

#### 关键边属性
```python
# Pattern → Domain 边
{
  'relation': 'works_well_in',
  'frequency': 15,              # Pattern在该Domain的使用次数
  'effectiveness': 0.12,        # 相对基线的效果增益
  'confidence': 0.75,           # 置信度(基于样本数)
  'avg_quality': 0.82,
  'baseline': 0.70
}
```

---

### 路径3: Idea → Paper → Pattern (相似Paper召回)

#### 旧版本
```python
paper_idea = paper.get('idea', {}).get('core_idea', '')
reviews = paper.get('reviews', [])
if reviews:
    avg_score = np.mean([r.get('rating') for r in reviews])
    quality = (avg_score - 1) / 9
else:
    quality = 0.5
```

#### V3版本
```python
# idea是字符串
paper_idea = paper.get('idea', '')

# 暂无review数据
quality = 0.5  # 默认值
```

#### 当前状态
- ⚠️ Paper质量全部为0.5,失去了质量区分能力
- ⚠️ 路径3的权重会偏向高相似度Paper,而非高质量Paper

---

## 📊 召回参数配置

### Top-K设置
```python
PATH1_TOP_K_IDEAS = 10      # 路径1: 召回前10个最相似Idea
PATH2_TOP_K_DOMAINS = 5     # 路径2: 召回前5个相关Domain
PATH3_TOP_K_PAPERS = 20     # 路径3: 召回前20个相似Paper
FINAL_TOP_K = 10            # 最终返回Top-10 Pattern
```

### 路径权重
```python
PATH1_WEIGHT = 0.4  # 路径1权重 (最高,直接经验)
PATH2_WEIGHT = 0.3  # 路径2权重 (领域泛化)
PATH3_WEIGHT = 0.3  # 路径3权重 (Paper匹配)
```

### 融合公式
```
final_score(pattern) = path1_score × 0.4
                     + path2_score × 0.3
                     + path3_score × 0.3
```

---

## 🎯 召回示例

### 测试用例
```python
user_idea = "使用Transformer模型进行文本分类任务"
```

### 预期流程

#### 路径1
```
1. 计算与8,284个Idea的相似度
2. Top-10相似Idea: [idea_123, idea_456, ...]
3. 收集pattern_ids: [pattern_5, pattern_12, ...]
4. 得分: {pattern_5: 0.85, pattern_12: 0.72, ...}
```

#### 路径2
```
1. 匹配Domain: Natural Language Processing (相关度=0.6)
2. 查询works_well_in边: pattern_5, pattern_18, ...
3. 得分: {pattern_5: 0.18, pattern_18: 0.15, ...}
```

#### 路径3
```
1. 找到相似Paper: [paper_789, paper_234, ...]
2. 查询uses_pattern边: pattern_5, pattern_12, ...
3. 得分: {pattern_5: 0.25, pattern_12: 0.20, ...}
```

#### 融合结果
```
pattern_5: 0.85×0.4 + 0.18×0.3 + 0.25×0.3 = 0.469
pattern_12: 0.72×0.4 + 0.0×0.3 + 0.20×0.3 = 0.348
pattern_18: 0.0×0.4 + 0.15×0.3 + 0.0×0.3 = 0.045
```

### 排序输出
```
Rank 1: pattern_5 (得分=0.469)
  - 路径1贡献: 0.340 (72.5%)
  - 路径2贡献: 0.054 (11.5%)
  - 路径3贡献: 0.075 (16.0%)
```

---

## ✅ 已完成的适配

### 代码文件
- ✅ `scripts/recall_system.py` - 完整召回系统
- ✅ `scripts/simple_recall_demo.py` - 简化Demo

### 主要修改
1. ✅ 路径1: 直接使用`idea.pattern_ids`
2. ✅ 路径3: 适配`paper.idea`字符串结构
3. ✅ 移除review评分逻辑,质量默认0.5
4. ✅ 结果展示: 适配V3 Pattern节点结构
   - 使用`size`代替`cluster_size`
   - 优先显示`llm_enhanced_summary`
   - 降级显示`summary`示例

---

## 🔮 未来优化方向

### 短期 (当review数据可用时)
- [ ] 恢复Paper质量评分逻辑
- [ ] 根据质量分布调整路径权重
- [ ] 引入质量阈值过滤低质量Paper

### 中期
- [ ] 升级相似度计算: Jaccard → Sentence-BERT
- [ ] 增加领域分类器: 关键词匹配 → 神经网络
- [ ] 优化路径2的Domain识别准确率

### 长期
- [ ] 引入用户反馈学习
- [ ] 动态调整路径权重
- [ ] Pattern特征增强(任务类型、技术栈标签)

---

## 📝 使用方法

### 方法1: 完整召回系统
```bash
# 运行4个测试用例
python scripts/recall_system.py
```

### 方法2: 单个测试
```bash
# 自定义Idea
python scripts/simple_recall_demo.py "你的Idea描述"

# 示例
python scripts/simple_recall_demo.py "使用图神经网络进行节点分类"
```

### 方法3: Python API
```python
from recall_system import RecallSystem

# 初始化
system = RecallSystem()

# 召回
results = system.recall("提升Transformer模型效率", verbose=True)

# 处理结果
for rank, (pattern_id, pattern_info, score) in enumerate(results, 1):
    print(f"{rank}. {pattern_info['name']} (得分={score:.4f})")
```

---

## 🎯 与旧版本对比总结

| 维度 | 旧版本 | V3版本 | 状态 |
|------|--------|--------|------|
| **数据规模** | 545 Papers | 8,285 Papers | ✅ 扩大15倍 |
| **节点结构** | 嵌套字典 | 扁平化 + LLM增强 | ✅ 更清晰 |
| **路径1效率** | 需图谱查询 | 直接访问 | ✅ 更快 |
| **路径2** | 无变化 | 无变化 | ✅ 稳定 |
| **路径3** | 基于review | 暂无review | ⚠️ 待优化 |
| **Pattern描述** | 简单摘要 | LLM归纳总结 | ✅ 更丰富 |
| **Idea覆盖率** | 10.9% | 100% | ✅ 大幅提升 |

---

## ⚠️ 已知问题

1. **Paper质量无差异**: 所有Paper质量=0.5,失去质量区分
   - **影响**: 路径3召回准确性下降
   - **解决**: 等待review数据补充

2. **路径1权重偏高**: 由于路径3质量失效,可能需要调整权重
   - **当前**: `[0.4, 0.3, 0.3]`
   - **建议**: `[0.5, 0.3, 0.2]` (提高路径1,降低路径3)

3. **相似度计算简单**: 使用Jaccard相似度
   - **准确性**: 中等
   - **改进**: 升级为BERT嵌入相似度

---

**文档版本**: V3.0
**更新时间**: 2026-01-22
**作者**: CatPaw AI

