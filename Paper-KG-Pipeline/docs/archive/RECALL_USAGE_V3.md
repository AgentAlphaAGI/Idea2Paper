# V3召回系统使用指南

## 🚀 快速开始

### 方法1: 完整召回系统 (推荐)

```bash
cd /Users/gaoge/code/mycode/Idea2Paper/Paper-KG-Pipeline
python3 scripts/recall_system.py
```

包含4个预设测试用例,展示完整的三路召回流程。

### 方法2: 单个测试

```bash
python3 scripts/simple_recall_demo.py "Your idea description in English"
```

示例:
```bash
python3 scripts/simple_recall_demo.py "Improve transformer efficiency with knowledge distillation"
```

### 方法3: Python API

```python
from recall_system import RecallSystem

# 初始化
system = RecallSystem()

# 召回
results = system.recall(
    "Improve graph neural network scalability for large graphs",
    verbose=True
)

# 处理结果
for rank, (pattern_id, pattern_info, score) in enumerate(results, 1):
    print(f"{rank}. {pattern_info['name']}")
    print(f"   得分: {score:.4f}")
    print(f"   聚类大小: {pattern_info['size']} 篇论文")
```

---

## ⚠️ 重要提示

### 1. Paper质量评分

**V3版本保留了完整的review评分逻辑**:
- 当Paper有review数据时,系统会自动计算真实质量分数(基于review评分,归一化到[0,1])
- 当Paper暂无review数据时,使用默认值0.5
- 这确保当未来review数据补充完整后,召回质量将自动提升

**质量分数计算逻辑**:
```python
# 从review中提取overall_score字段
# 支持多种格式: "7", "7/10", "7.0"
# 计算平均分并归一化到[0,1]: (avg_score - 1) / 9
# 假设评分范围为1-10
```

**在召回结果中的体现**:
- 路径3会显示每个Paper的质量来源: `[review]` 或 `[默认]`
- Edge构建时会使用相同的质量评分逻辑
- Pattern-Paper边的权重会反映真实的Paper质量

### 2. 输入语言
**V3数据集的Idea描述为英文**,请使用英文输入:

✅ 正确:
```python
"Use graph neural networks for node classification"
"Improve transformer efficiency with pruning techniques"
```

❌ 错误:
```python
"使用图神经网络进行节点分类"  # 中文无法匹配
```

### 2. 路径权重调整（当review数据缺失时）
如果大部分Paper没有review数据（质量都是0.5）:
- 路径1和路径3的区分度下降
- 建议临时调整权重: PATH1=0.5, PATH2=0.3, PATH3=0.2
- 当review数据补充后,可恢复默认权重: PATH1=0.4, PATH2=0.3, PATH3=0.3

### 3. 输入语言
**V3数据集的Idea描述为英文**,请使用英文输入:

✅ 正确:
```python
"Use graph neural networks for node classification"
"Improve transformer efficiency with pruning techniques"
```

❌ 错误:
```python
"使用图神经网络进行节点分类"  # 中文无法匹配
```

### 4. 召回速度
- 路径1: 快速 (直接访问pattern_ids)
- 路径2: 中速 (需图谱查询)
- 路径3: 较慢 (需计算8285个Paper的相似度)

---

## 📊 输出示例

```
================================================================================
🎯 开始三路召回
================================================================================

【用户Idea】
Improve transformer model efficiency

🔍 [路径1] 相似Idea召回...
  找到 5234 个相似Idea，选择Top-10
  - idea_123 (相似度=0.652): 2 个Pattern
  - idea_456 (相似度=0.601): 1 个Pattern
  ...
  ✓ 召回 15 个Pattern

🌍 [路径2] 领域相关性召回...
  找到 8 个相关Domain，选择Top-5
  - domain_12 (Natural Language Processing, 相关度=0.450)
  - domain_24 (Machine Learning, 相关度=0.320)
  ...
  ✓ 召回 28 个Pattern

📄 [路径3] 相似Paper召回...
  找到 3421 个相似Paper，选择Top-20
  - paper_xyz (相似度=0.432, 质量=0.500 [默认])
  ...
  ✓ 召回 32 个Pattern

🔗 融合三路召回结果...

================================================================================
📊 召回结果 Top-10
================================================================================

【Rank 1】 pattern_24
  名称: Reframing Graph Learning Scalability
  最终得分: 0.4523
  - 路径1 (相似Idea):   0.2810 (62.1%)
  - 路径2 (领域相关):   0.0890 (19.7%)
  - 路径3 (相似Paper):  0.0823 (18.2%)
  聚类大小: 331 篇论文
  归纳总结: Papers in this cluster explore innovative approaches to enhance the...

【Rank 2】 pattern_67
  ...
```

---

## 🔧 参数调整

### 修改召回数量

编辑 `recall_system.py`:

```python
class RecallConfig:
    PATH1_TOP_K_IDEAS = 20      # 默认10
    PATH2_TOP_K_DOMAINS = 10    # 默认5
    PATH3_TOP_K_PAPERS = 30     # 默认20
    FINAL_TOP_K = 20            # 默认10
```

### 修改路径权重

考虑到Paper质量都是0.5,建议调整权重:

```python
class RecallConfig:
    PATH1_WEIGHT = 0.5  # 提高 (默认0.4)
    PATH2_WEIGHT = 0.3  # 保持 (默认0.3)
    PATH3_WEIGHT = 0.2  # 降低 (默认0.3)
```

---

## 🐛 故障排查

### 问题1: 召回结果为空
**原因**: 输入使用中文,但数据集是英文
**解决**: 使用英文输入

### 问题2: ModuleNotFoundError: numpy._core
**原因**: numpy版本不兼容
**解决**: 重新运行 `python3 scripts/build_edges.py` 生成兼容的图谱文件

### 问题3: 路径1召回Pattern数为0
**原因**: 72.2%的Idea有pattern_ids,28%没有
**说明**: 正常现象,这些Idea未被分配到Pattern cluster

---

## 📚 相关文档

- `docs/RECALL_V3_UPDATES.md` - V3版本详细变化说明
- `docs/EDGE_TYPES.md` - 边类型和召回策略文档
- `docs/KG_Rebuild_Analysis_V3.md` - 知识图谱构建文档

---

**更新时间**: 2026-01-22
**版本**: V3.0

