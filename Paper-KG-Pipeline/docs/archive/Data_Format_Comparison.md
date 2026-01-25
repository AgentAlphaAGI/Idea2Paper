# 数据格式对比分析：旧版 vs ICLR新版

## 📊 数据源对比

### **旧版数据源（ACL/ARR/COLING）**

| 文件 | 作用 | 示例路径 |
|-----|------|---------|
| `*_paper_node.json` | 单篇论文的详细信息 | `ACL_2017/ACL_2017_12_paper_node.json` |
| `*_review.json` | 论文评审意见 | `ACL_2017/ACL_2017_12_reviews.json` |
| `patterns_structured.json` | 手动构建的Pattern聚类 | `output/patterns_structured.json` |
| `paper_to_pattern.json` | Paper到Pattern的映射 | `output/paper_to_pattern.json` |

### **ICLR新版数据源**

| 文件 | 作用 | 记录数 |
|-----|------|--------|
| `assignments.jsonl` | Paper到Pattern的分配关系 | 8,285 |
| `cluster_library_sorted.jsonl` | Pattern Cluster信息 | 124 |
| `iclr_patterns_full_cn_912.jsonl` | Pattern详细属性（中文） | 912 |

---

## 🔄 数据结构对比

### 1. **Paper节点**

#### 旧版（from `*_paper_node.json`）
```json
{
  "paper_id": "ACL_2017_12",
  "title": "...",
  "conference": "ACL_2017",
  "ideal": {
    "core_idea": "核心创新点描述",
    "tech_stack": ["技术1", "技术2"],
    "input_type": "输入类型",
    "output_type": "输出类型"
  },
  "domain": {
    "domains": ["NLP", "Machine Learning"],
    "research_object": "研究对象",
    "core_technique": "核心技术",
    "application": "应用场景"
  },
  "skeleton": {
    "problem_framing": "问题框架",
    "gap_pattern": "研究空白",
    "method_story": "方法叙事",
    "experiments_story": "实验叙事"
  },
  "tricks": [
    {
      "name": "技巧名称",
      "type": "技巧类型",
      "description": "描述",
      "purpose": "目的",
      "location": "位置"
    }
  ]
}
```

#### 新版（from `assignments.jsonl` + `iclr_patterns_full_cn_912.jsonl`）
```json
{
  "paper_id": "RUzSobdYy0V",
  "title": "Quantifying and Mitigating...",
  "global_pattern_id": "g0",
  "cluster_id": 9,
  "cluster_prob": 0.384,
  "domain": "Fairness & Accountability",
  "sub_domains": ["Label Noise", "Disparity Metrics", ...],
  "idea": "通过分析标签错误对群体差异指标的影响...",
  "pattern_details": {
    "base_problem": "在群体差异指标评估中...",
    "solution_pattern": "提出一种方法估计...",
    "story": "将标签错误问题从模型性能影响...",
    "application": "高风险决策系统的公平性审计..."
  },
  "pattern_id": "pattern_9",
  "idea_id": "idea_0"
}
```

**对比分析**:
| 字段 | 旧版 | 新版 | 差异 |
|-----|------|------|------|
| **idea描述** | `ideal.core_idea` | `idea` | ✅ 新版更简洁 |
| **领域信息** | `domain.domains[]` | `domain` + `sub_domains[]` | ✅ 新版分层更清晰 |
| **Pattern信息** | `skeleton` (4个字段) | `pattern_details` (4个字段) | ✅ 新版字段更语义化 |
| **技巧信息** | `tricks[]` | ❌ 缺失 | ⚠️ 新版无Tricks |
| **Pattern关联** | 通过外部映射 | `cluster_id` + `cluster_prob` | ✅ 新版直接包含 |

---

### 2. **Pattern节点**

#### 旧版（from `patterns_structured.json`）
```json
{
  "pattern_id": 1,
  "pattern_name": "...",
  "pattern_summary": "...",
  "writing_guide": "...",
  "skeleton_examples": [
    {
      "paper_id": "...",
      "title": "...",
      "problem_framing": "...",
      "gap_pattern": "...",
      "method_story": "...",
      "experiments_story": "..."
    }
  ],
  "common_tricks": [
    {
      "trick_name": "...",
      "frequency": 5,
      "percentage": "50%",
      "examples": [...]
    }
  ],
  "metadata": {
    "cluster_size": 10,
    "coherence_score": 0.8,
    "all_paper_ids": [...]
  }
}
```

#### 新版（from `cluster_library_sorted.jsonl`）
```json
{
  "pattern_id": "pattern_24",
  "cluster_id": 24,
  "name": "Reframing Graph Learning Scalability",
  "size": 331,
  "domain": "Machine Learning",
  "sub_domains": ["Graph Neural Networks", ...],
  "coherence": {
    "centroid_mean": 0.668,
    "centroid_p50": 0.691,
    "pairwise_sample_mean": 0.461,
    "pairwise_sample_p50": 0.469
  },
  "summary": {
    "representative_ideas": ["...", "...", "..."],
    "common_problems": ["...", "...", "..."],
    "solution_approaches": ["...", "...", "..."]
  },
  "exemplar_paper_ids": ["cZM4iZmxzR7", ...]
}
```

**对比分析**:
| 维度 | 旧版 | 新版 | 差异 |
|-----|------|------|------|
| **聚类质量** | `coherence_score` (单一值) | `coherence` (4个指标) | ✅ 新版更细粒度 |
| **写作指南** | `writing_guide` | ❌ 缺失 | ⚠️ 新版无writing_guide |
| **Skeleton样例** | `skeleton_examples[]` | ❌ 缺失 | ⚠️ 新版无skeleton |
| **Trick统计** | `common_tricks[]` | ❌ 缺失 | ⚠️ 新版无tricks |
| **代表性论文** | `skeleton_examples` (手动选取) | `exemplar_paper_ids` (自动选取) | ✅ 新版自动化 |
| **Pattern描述** | `pattern_summary` (人工总结) | `summary.representative_ideas` (从exemplars提取) | ✅ 新版数据驱动 |

---

### 3. **Idea节点**

#### 旧版（from `*_paper_node.json`的`ideal`字段）
```json
{
  "idea_id": "idea_0",
  "description": "核心创新点描述",
  "tech_stack": ["技术1", "技术2"],
  "input_type": "输入类型",
  "output_type": "输出类型",
  "source_paper_ids": ["paper_1", "paper_2"],
  "pattern_ids": ["pattern_1"]
}
```

#### 新版（from `iclr_patterns_full_cn_912.jsonl`）
```json
{
  "idea_id": "idea_0",
  "description": "通过分析标签错误对群体差异指标的影响，提升模型公平性评估的可靠性",
  "base_problem": "在群体差异指标评估中，标签错误对少数群体的影响被放大...",
  "solution_pattern": "提出一种方法估计单个训练输入标签的变化...",
  "story": "将标签错误问题从模型性能影响扩展到公平性评估的可靠性问题...",
  "application": "高风险决策系统的公平性审计、数据质量提升与偏差检测",
  "domain": "公平性与可信人工智能",
  "sub_domains": ["标签噪声", "公平性评估", "模型审计"],
  "source_paper_ids": ["RUzSobdYy0V"],
  "pattern_ids": ["pattern_9"]
}
```

**对比分析**:
| 维度 | 旧版 | 新版 | 差异 |
|-----|------|------|------|
| **技术栈** | `tech_stack[]` | ❌ 缺失 | ⚠️ 新版无tech_stack |
| **输入输出** | `input_type`, `output_type` | ❌ 缺失 | ⚠️ 新版无类型信息 |
| **问题描述** | ❌ 缺失 | `base_problem` | ✅ 新版更完整 |
| **解决方案** | ❌ 缺失 | `solution_pattern` | ✅ 新版更完整 |
| **故事叙述** | ❌ 缺失 | `story` | ✅ 新版更完整 |
| **应用场景** | ❌ 缺失 | `application` | ✅ 新版更完整 |

---

### 4. **Domain节点**

#### 旧版
```json
{
  "domain_id": "domain_0",
  "name": "Natural Language Processing",
  "paper_count": 150,
  "research_objects": ["文本", "语言模型"],
  "core_techniques": ["深度学习", "Transformer"],
  "applications": ["机器翻译", "文本生成"]
}
```

#### 新版
```json
{
  "domain_id": "domain_0",
  "name": "Fairness & Accountability",
  "paper_count": 69,
  "sub_domains": ["Label Noise", "Bias Mitigation", ...],
  "related_pattern_ids": ["pattern_9", "pattern_15", ...],
  "sample_paper_ids": ["RUzSobdYy0V", ...]
}
```

**对比分析**:
| 维度 | 旧版 | 新版 | 差异 |
|-----|------|------|------|
| **子领域** | ❌ 缺失 | `sub_domains[]` | ✅ 新版更细粒度 |
| **研究对象** | `research_objects[]` | ❌ 缺失 | ⚠️ 新版无research_objects |
| **核心技术** | `core_techniques[]` | ❌ 缺失 | ⚠️ 新版无core_techniques |
| **应用场景** | `applications[]` | ❌ 缺失 | ⚠️ 新版无applications |
| **Pattern关联** | ❌ 缺失 | `related_pattern_ids[]` | ✅ 新版直接关联 |

---

## 📈 数据规模对比

| 维度 | 旧版 | 新版 | 变化 |
|-----|------|------|------|
| **Paper数量** | ~数百篇 | 8,285篇 | ✅ 大幅增加 |
| **Pattern数量** | ~数十个 | 124个 | ✅ 规模增大 |
| **Domain数量** | ~数十个 | 98个 | ➡️ 相似 |
| **Idea数量** | ~数百个 | 904个 | ✅ 规模增大 |
| **数据源** | 3个会议 | 1个会议（ICLR） | ➡️ 单一但规模大 |

---

## ⚖️ 优劣势对比

### **旧版优势**
✅ **Skeleton信息完整**: `problem_framing`, `gap_pattern`, `method_story`, `experiments_story`
✅ **Trick统计丰富**: 包含频率、百分比、样例
✅ **Writing Guide**: 人工总结的写作指南
✅ **技术栈明确**: `tech_stack`, `input_type`, `output_type`
✅ **Review信息**: 包含评审意见

### **新版优势**
✅ **数据规模大**: 8,285篇论文 vs 数百篇
✅ **聚类质量可量化**: 4个coherence指标
✅ **Pattern自动化**: 基于聚类算法自动生成
✅ **中文化描述**: idea和pattern_details均为中文
✅ **关联明确**: 直接包含`cluster_id`和`cluster_prob`
✅ **Pattern描述完整**: `base_problem`, `solution_pattern`, `story`, `application`

### **新版劣势**
⚠️ **缺少Skeleton**: 无`problem_framing`, `gap_pattern`等字段
⚠️ **缺少Tricks**: 无技巧统计信息
⚠️ **缺少Writing Guide**: 无写作指南
⚠️ **缺少技术栈**: 无`tech_stack`, `input_type`, `output_type`
⚠️ **缺少Review**: 无评审意见（ICLR数据源限制）
⚠️ **Idea覆盖率低**: 仅10.9% (901/8,285)

---

## 🎯 融合方案建议

### **方案1: 保留两套数据源**
- 旧版用于**高质量Skeleton和Trick分析**
- 新版用于**大规模Pattern发现和统计分析**

### **方案2: 补充新版数据**
通过LLM为新版数据补充缺失字段:
1. ✅ 为所有Paper生成`skeleton`信息
2. ✅ 为所有Paper生成`tricks`信息
3. ✅ 为所有Idea补充`tech_stack`, `input_type`, `output_type`
4. ✅ 为所有Pattern生成`writing_guide`

### **方案3: 迁移旧版数据到新结构**
将旧版的Skeleton和Trick信息迁移到新版结构中:
```python
# 为新版Paper节点补充skeleton和tricks字段
paper_node['skeleton'] = {
    'problem_framing': '...',
    'gap_pattern': '...',
    'method_story': '...',
    'experiments_story': '...'
}
paper_node['tricks'] = [...]
```

---

## 📝 总结

### **数据源特点**
| 维度 | 旧版（ACL/ARR/COLING） | 新版（ICLR） |
|-----|----------------------|-------------|
| **规模** | 中等（数百篇） | 大规模（8,285篇） |
| **质量** | 高（人工标注） | 中等（自动聚类） |
| **完整性** | 高（包含Skeleton, Tricks, Review） | 中等（缺少部分字段） |
| **自动化** | 低（需要人工构建Pattern） | 高（自动聚类生成） |
| **可扩展性** | 低（需要手动标注） | 高（可批量处理） |

### **推荐策略**
1. **短期**: 使用新版数据（ICLR）进行大规模Pattern分析
2. **中期**: 通过LLM补充新版数据的缺失字段（Skeleton, Tricks）
3. **长期**: 建立统一的数据标注流程，融合两套数据源的优势

---

**生成时间**: 2026-01-22
**版本**: V1
**作者**: AI Agent (Catpaw)

