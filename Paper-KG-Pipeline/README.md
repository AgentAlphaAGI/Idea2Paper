# Paper-Knowledge-Graph-Pipeline 运行指南

这是一个基于论文知识图谱的 **Pattern 生成与 Idea 召回系统**，现已支持 **Idea2Story 自动生成** 功能。本指南帮助你快速上手最新版本（2026年1月更新）。

## 🆕 新功能: Idea2Story Pipeline

✅ **自动生成可发表的 Paper Story**
✅ **方法论深度融合**（精准注入技术路线描述，而非简单堆砌）
✅ **多智能体评审机制**（Methodology/Novelty/Storyteller）
✅ **智能修正策略**（Tail/Head Injection + 增量修正）
✅ **多源数据合并**（patterns_structured.json 完整骨架数据）
✅ **RAG 查重与 Pivot 规避**
✅ **详细的执行日志**（便于调试）

**快速开始**: `python scripts/idea2story_pipeline.py "你的研究想法"`

---

## 📋 快速开始

### 前置条件

```bash
# 检查 Python 版本（建议 3.8+）
python --version

# 安装依赖
pip install numpy scikit-learn requests networkx
```

### 环境变量配置

系统需要 **SiliconFlow API** 来生成 Embeddings 和调用 LLM。参考 `01_RAG_minimal_DEMO.ipynb` 进行配置：

```bash
# 方式1：设置环境变量（推荐）
export SILICONFLOW_API_KEY="sk-your-api-key-here"
export LLM_API_URL="https://api.siliconflow.cn/v1/chat/completions"
export EMBED_API_URL="https://api.siliconflow.cn/v1/embeddings"
export LLM_MODEL="Qwen/Qwen2.5-7B-Instruct"
export EMBED_MODEL="Qwen/Qwen3-Embedding-4B"

# 或方式2：在脚本中直接配置（见各脚本的配置部分）
```

---

## 🔧 完整工作流

### Step 1: 数据准备

确保数据目录结构如下：

```
data/
├── ACL_2017/
│   ├── paper_0_paper_node.json
│   ├── paper_1_paper_node.json
│   └── ...
├── COLING_2020/
│   ├── paper_0_paper_node.json
│   └── ...
└── ARR_2022/
    └── ...
```

每个 `*_paper_node.json` 文件应包含以下字段：
```json
{
  "paper_id": "...",
  "title": "...",
  "conference": "...",
  "skeleton": {
    "problem_framing": "...",
    "gap_pattern": "...",
    "method_story": "...",
    "experiments_story": "..."
  },
  "tricks": [
    {
      "name": "技巧名称",
      "type": "...",
      "description": "...",
      "purpose": "..."
    }
  ]
}
```

### Step 2: 生成 Pattern（可选，如已有 patterns_structured.json 可跳过）

```bash
cd scripts
python generate_patterns.py
```

**预期输出**：
```
================================================================================
基于 Skeleton + Tricks 聚类生成 Patterns
================================================================================

【Step 1】加载论文数据
  📁 加载 ACL_2017: 123 篇论文
  📁 加载 COLING_2020: 234 篇论文
  ✅ 共加载 545 篇论文

【Step 2】构建pattern embeddings
  ✓ 完成 545 个pattern的embedding

【Step 3】聚类
🔄 开始聚类...
  距离阈值: 0.35
  ✓ 生成 34 个 clusters
  📊 Cluster 大小分布:
    Cluster 0: 8 篇
    Cluster 1: 12 篇
    ...

【Step 4】生成patterns
  📊 分析 Cluster 0 (8 篇)...
    Pattern 1: 模型压缩与知识蒸馏...
  ...
  ✅ 共生成 34 个patterns

【Step 5】生成输出文件
  ✅ patterns_structured.json
  ✅ paper_to_pattern.json
  ✅ patterns_guide.txt
  ✅ patterns_statistics.json
```

### Step 3: 构建知识图谱

```bash
# 构建实体节点
python build_entity.py

# 构建边关系
python build_edges.py
```

**预期输出**：
```
✅ 已生成：
  - output/nodes_idea.json (545 个Idea节点)
  - output/nodes_pattern.json (34 个Pattern节点)
  - output/nodes_domain.json (257 个Domain节点)
  - output/nodes_paper.json (545 个Paper节点)
  - output/edges_*.json (各类边关系)
```

### Step 4: 运行召回系统演示

```bash
python simple_recall_demo.py
```

**交互式输入**：
```
请输入 Idea（或按 Enter 使用默认示例）:
使用蒸馏技术完成Transformer跨领域文本分类任务，并在多个数据集上验证效果
```

**预期输出**：
```
================================================================================
🎯 三路召回系统 Demo
================================================================================

【用户Idea】
使用蒸馏技术完成Transformer跨领域文本分类任务，并在多个数据集上验证效果

📂 加载数据...
  ✓ Idea: 545, Pattern: 34, Domain: 257, Paper: 545
  ✓ 图谱: 1381 节点, 4509 边

🔍 [路径1] 相似Idea召回...
  找到 523 个相似Idea，选择 Top-10
  ✓ 召回 5 个Pattern

🌍 [路径2] 领域相关性召回...
  找到 3 个相关Domain，选择 Top-5
  ✓ 召回 34 个Pattern

📄 [路径3] 相似Paper召回...
  找到 171 个相似Paper，选择 Top-20
  ✓ 召回 9 个Pattern

🔗 融合三路召回结果...

================================================================================
📊 召回结果 Top-10
================================================================================

【Rank 1】 pattern_11
  名称: 模型压缩与知识蒸馏
  最终得分: 0.1312
  - 路径1 (相似Idea):   0.1049 (占比 79.9%)
  - 路径2 (领域相关):   0.0030 (占比 2.3%)
  - 路径3 (相似Paper):  0.0233 (占比 17.8%)
  聚类大小: 5 篇论文
  摘要: ...

【Rank 2】 pattern_17
  名称: 结构图谱预测方法
  最终得分: 0.1249
  ...

================================================================================
✅ 召回完成!
================================================================================
```

---

## 📊 关键改进点（相比旧版本）

### Pattern 聚类优化

| 方面 | 旧版本 | 新版本 | 改进 |
| :--- | :--- | :--- | :--- |
| **聚类策略** | 固定 30 个簇 | 自适应距离阈值 | ✅ 更合理 |
| **Pattern 总数** | 30 个 | 34 个 | ✅ 选择更多 |
| **最大 Pattern** | 448 篇(82%) | 30 篇(8.6%) | ✅ 消除"万金油" |
| **最小 Pattern** | 2 篇(质量低) | 5 篇(有筛选) | ✅ 质量保证 |
| **中位数簇大小** | 2 篇 | 8.5 篇 | ✅ 代表性强 |
| **中文 Idea 召回** | 0 个结果 | 精准召回 | ✅ 修复 bug |

### 参数配置

```python
# 聚类参数（位于 scripts/generate_patterns.py）
CLUSTER_PARAMS = {
    "distance_threshold": 0.35,  # 自适应聚类的距离阈值
    "min_cluster_size": 5,       # 最小簇大小，低于此值的簇被过滤
    "skeleton_weight": 0.4,      # Skeleton 权重（骨架）
    "tricks_weight": 0.6,        # Tricks 权重（技巧）
}
```

**如何调整**：
- 降低 `distance_threshold`：生成更多、更精细的 Pattern（推荐 0.30-0.35）
- 增加 `min_cluster_size`：只保留更大的、代表性更强的簇（推荐 5-10）
- 调整 weights：改变 Skeleton 和 Tricks 在聚类中的重要性比例

---

## 🔍 理解召回系统

### 三路召回策略

**路径1：相似Idea召回** (Idea → Idea → Pattern)
```
用户输入Idea
  ↓ [向量相似度计算]
找到最相似的 K 个已有Idea
  ↓ [获取关联Pattern]
直接推荐这些Pattern
  ↓
得分占比通常 > 50%（直接相关）
```

**路径2：领域相关性召回** (Idea → Domain → Pattern)
```
用户Idea属于哪些Domain
  ↓ [领域权重计算]
该Domain内表现最好的Pattern
  ↓ [效果增益计算]
推荐"领域之星"Pattern
  ↓
得分占比通常 20-40%（领域相关但不直接）
```

**路径3：相似Paper召回** (Idea → Paper → Pattern)
```
找到与用户Idea相似的高质量论文
  ↓ [论文质量评分]
这些论文使用的Pattern
  ↓ [质量反向背书]
高质量论文推荐的Pattern可信度高
  ↓
得分占比通常 20-40%（质量保证）
```

### 结果融合与解读

```python
最终得分 = Path1_Score * 0.4 + Path2_Score * 0.3 + Path3_Score * 0.3
```

**解读技巧**：

| 得分占比模式 | 解释 | 建议 |
| :--- | :--- | :--- |
| **路径1 > 70%** | 用户Idea 与历史相似 | ✅ 稳妥选择 |
| **路径2 > 40%** | Pattern 在领域表现好 | ✅ 领域通用 |
| **路径3 > 40%** | 高质量Paper背书 | ✅ 质量保证 |
| **三路均衡** | 各角度都支持 | ⭐ 最佳选择 |

---

---

## 🚀 Step 5: 运行 Idea2Story Pipeline（新增！）

完成召回后，可以使用 Pipeline 自动生成可发表的 Paper Story。

```bash
# 使用默认 Idea
python scripts/idea2story_pipeline.py

# 自定义 Idea
python scripts/idea2story_pipeline.py "你的研究想法描述"
```

### Pipeline 核心特性

#### 🎯 方法论深度融合（最新改进）

**问题**: 早期版本的 Refine 只是在 Story 末尾追加 Trick 名称，导致"技术堆砌"问题。

**解决**:
1. **精准提取**: 从 `patterns_structured.json` 的 `skeleton_examples` 中提取完整的 `method_story`（方法论描述）
2. **针尖式注入**: 将具体的方法论逻辑（而非仅 Trick 名称）直接注入到 Prompt 中
3. **重构引导**: Prompt 提供"深度融合 vs 简单堆砌"的正反范例，强制 LLM 进行方法论重构

**示例对比**:

❌ **旧版本（技术堆砌）**:
```
方法步骤1；方法步骤2；添加课程学习；再添加对抗训练
```

✅ **新版本（深度融合）**:
```
方法步骤1；在训练过程中引入基于难度的课程学习调度器，
结合对抗扰动正则项，形成渐进式鲁棒训练框架；方法步骤3
```

#### 📊 多源数据合并（关键修复）

- 加载 `nodes_pattern.json`（基础 Pattern 信息）
- 合并 `patterns_structured.json`（完整的 `skeleton_examples` 和 `common_tricks`）
- 确保 Refinement 阶段能访问到完整的方法论描述数据

### Pipeline 流程

```
User Idea → [召回 Top-10 Patterns]
    ↓
[Phase 1] 策略选择（稳健型/创新型/跨域型）
    ↓
[Phase 2] 初始 Story 生成
    ↓
[Phase 3] 多智能体评审（Methodology/Novelty/Storyteller）
    ↓ Fail
[Phase 3.5] 智能修正（注入方法论描述 + 增量修正）
    ↓ Pass
[Phase 4] RAG 查重验证
    ↓ Collision
[Pivot 策略] 领域迁移 + 约束生成
    ↓
✅ Final Story
```

### 输出文件

- `output/final_story.json` - 最终生成的 Story
- `output/pipeline_result.json` - 完整的执行历史（包含评审记录、修正历史）

详细说明见 `docs/PIPELINE_IMPLEMENTATION.md` 和 `docs/QUICK_START_PIPELINE.md`

---

## 📚 详细文档

- **知识图谱体系**：见 `docs/RECALL_SYSTEM_EXPLAINED.md`
  - Pattern 聚类逻辑详解
  - 节点/边定义与规模
  - 三路召回策略工作机制
  - 实际案例分析（Recall_Case_1）
  - 聚类改进前后对比

- **Idea2Story Pipeline**：见 `docs/PIPELINE_IMPLEMENTATION.md`（新增）
  - Pipeline 设计思路
  - 各模块详细说明
  - 配置参数说明
  - 调试建议
  - 优化方向

- **直观演示**：见 `docs/recall_case_1`
  - 真实 Idea 的三路召回输出
  - Top-10 Pattern 排名
  - 每个 Pattern 的详细信息

---

## 🐛 常见问题

### Q1: Embedding API 报错 (401/403)

**原因**：SiliconFlow API Key 配置错误

**解决**：
```bash
# 检查API Key是否正确
echo $SILICONFLOW_API_KEY

# 或在脚本中直接配置
# scripts/generate_patterns.py 第 20-31 行
```

### Q2: 聚类耗时很长

**原因**：Embedding API 调用频繁，有频率限制

**优化**：
```python
# scripts/generate_patterns.py 第 179-180 行
time.sleep(0.1)  # 调大间隔，如 0.2 或 0.3
```

### Q3: 召回结果为 0

**原因**：
1. 中文 Idea 分词问题（已修复）
2. 知识图谱未正确构建
3. 相似度阈值过高

**调试**：
```python
# scripts/simple_recall_demo.py
# 添加调试输出
print(f"找到 {len(similar_ideas)} 个相似Idea")
print(f"Top-1 相似度: {similarities[0] if similarities else 'N/A'}")
```

### Q4: Pattern 数量与预期不符

**检查**：
```python
# 查看聚类统计
import json
with open('output/patterns_statistics.json') as f:
    stats = json.load(f)
    print(f"Pattern 数: {stats['total_patterns']}")
    print(f"覆盖论文数: {stats['total_papers']}")
    print(f"簇大小分布: {stats['cluster_size_distribution']}")
```

---

## 📈 性能参考

基于 545 篇论文的演示数据集：

| 操作 | 耗时 | 备注 |
| :--- | :--- | :--- |
| 加载论文 | ~1秒 | 从磁盘读取 |
| 构建Embeddings | ~5-10分钟 | 需要调用API，受网络限制 |
| 聚类 | ~1秒 | 本地计算 |
| 生成Pattern摘要 | ~3-5分钟 | 调用LLM生成 |
| 构建图谱 | ~2秒 | 本地计算 |
| 单次召回 | ~1秒 | 实时计算相似度 |



