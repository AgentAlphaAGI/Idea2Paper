# Idea2Story 生成与注入机制详解

> **作者**: Paper-KG-Pipeline Team
> **日期**: 2026-01-13
> **版本**: v2.0
> **关键改进**: 方法论深度融合 | 多维度注入策略 | 增量修正模式

---

## 📑 目录

1. [系统概览](#1-系统概览)
2. [Pattern 选择机制](#2-pattern-选择机制)
3. [Story 生成机制](#3-story-生成机制)
4. [注入机制详解](#4-注入机制详解)
5. [生成与注入协作流程](#5-生成与注入协作流程)
6. [技术演进对比](#6-技术演进对比)
7. [实战案例分析](#7-实战案例分析)
8. [调优建议](#8-调优建议)

---

## 1. 系统概览

### 1.1 核心设计理念

Idea2Story Pipeline 采用 **"Pattern-Guided Generation + Multi-Strategy Injection"** 的架构设计。

**核心流程**:
```
User Idea → Pattern Selection → Initial Generation → Critic → Refinement → Final Story
```

### 1.2 三大核心模块

| 模块 | 文件 | 职责 | 关键方法 |
|------|------|------|---------|
| **PatternSelector** | `pattern_selector.py` | 选择多样化 Pattern | `select()` |
| **StoryGenerator** | `story_generator.py` | 生成/修正 Story | `generate()` |
| **RefinementEngine** | `refinement.py` | 注入方法论 | `refine()` |

### 1.3 关键创新点

| 维度 | 旧版本 | 新版本 |
|------|--------|--------|
| **注入内容** | 技术名词 | 完整方法论描述（150字） |
| **注入方式** | 末尾追加 | 针尖式注入到核心逻辑 |
| **修正模式** | 全量重生成 | 增量修正（保留精华） |
| **数据来源** | `nodes_pattern.json` | 合并 `patterns_structured.json` |

---

## 2. Pattern 选择机制

### 2.1 三种选择策略

**目标**: 从召回的 Top-10 中选择 3 个多样化 Pattern。

```python
class PatternSelector:
    def select(self) -> Dict[str, Tuple[str, Dict]]:
        return {
            'conservative': (pattern_id, pattern_info),  # 稳健型
            'innovative': (pattern_id, pattern_info),    # 创新型
            'cross_domain': (pattern_id, pattern_info)   # 跨域型
        }
```

| 策略类型 | 选择逻辑 | 配置参数 |
|---------|---------|---------|
| **Conservative** | 召回得分最高 | Top-1 |
| **Innovative** | `cluster_size < 10` | `INNOVATIVE_CLUSTER_SIZE_THRESHOLD = 10` |
| **Cross-Domain** | 剩余中得分次高 | 排除前两者 |

### 2.2 选择流程示例

```
✅ [稳健型] pattern_11
   名称: 模型压缩与知识蒸馏
   聚类大小: 30 篇

✅ [创新型] pattern_23
   名称: 课程学习调度
   聚类大小: 5 篇

✅ [跨域型] pattern_17
   名称: 结构图谱预测方法
   聚类大小: 15 篇
```

---

## 3. Story 生成机制

### 3.1 双模式设计

```python
def generate(self, pattern_id, pattern_info,
             previous_story=None, review_feedback=None, ...):
    if previous_story and review_feedback:
        # 【增量修正模式】
        prompt = self._build_refinement_prompt(...)
    else:
        # 【初次生成模式】
        prompt = self._build_generation_prompt(...)
```

### 3.2 初次生成模式

#### Prompt 结构

```
【用户 Idea】
使用蒸馏技术完成Transformer跨领域文本分类任务

【写作模板】模型压缩与知识蒸馏
...

【模板示例】
示例 1:
  标题: ...
  方法概述: 我们设计了一个自适应蒸馏框架。首先，通过注意力机制对齐教师和学生模型的中间层特征...

【高频技巧】
  - 知识蒸馏 (85%)
  - 温度调节 (70%)

【任务要求】
生成 JSON 格式的 Story...
```

#### 输出结构

```json
{
  "title": "自适应蒸馏框架在跨域文本分类中的应用",
  "abstract": "...",
  "problem_definition": "...",
  "method_skeleton": "第一步：...；第二步：...；第三步：...",
  "innovation_claims": ["贡献1", "贡献2", "贡献3"],
  "experiments_plan": "..."
}
```

### 3.3 增量修正模式

#### 触发条件

```python
# 评审不通过时触发
if not critic_result['pass']:
    refinement_result = refinement.refine(main_issue='novelty')

    # 增量修正
    story = generator.generate(
        pattern_id,
        pattern_info,
        previous_story=current_story,      # 上一版本
        review_feedback=critic_result,      # 评审反馈
        new_tricks_only=injected_tricks     # 新注入内容
    )
```

#### 增量修正 Prompt 核心

```
【当前 Story 版本】
Title: ...
Method: ...

【评审专家反馈】
- Reviewer B (Novelty): 4.0分. "创新性不足，技术组合常见"

【核心任务：方法论深度重构】
🔧 【方法论重构】参考 课程学习调度 的核心技术路线：
   我们设计了一个基于样本难度的课程学习调度器。首先，通过预训练模型
   计算每个样本的预测置信度作为难度指标；然后，在训练早期仅使用简单
   样本，随训练进程逐步引入困难样本...

【重构要求】
1. **方法论融合**：将新技术深度嵌入到核心逻辑中
2. **技术组合创新**：形成 1+1>2 的效果
3. **贡献点更新**：明确指出新技术如何解决问题

❌ 差的修正: "方法步骤1；方法步骤2；添加课程学习"
✅ 好的修正: "在训练过程中引入基于难度的课程学习调度器，结合对抗
              扰动正则项，形成渐进式鲁棒训练框架"
```

#### 保底策略

```python
# 如果 LLM "忘记"某些字段，从上一版本恢复
if previous_story:
    for key in ['title', 'abstract', 'method_skeleton', ...]:
        if not story.get(key):
            story[key] = previous_story.get(key)
```

---

## 4. 注入机制详解

### 4.1 注入策略矩阵

```python
def refine(self, main_issue: str):
    if main_issue == 'novelty':
        return self._inject_tail_tricks()       # 长尾注入
    elif main_issue == 'stability':
        return self._inject_head_tricks()       # 头部注入
    elif main_issue == 'interpretability':
        return self._inject_explanation_tricks() # 解释性注入
    elif main_issue == 'domain_mismatch':
        return self._inject_domain_tricks()     # 领域适配注入
```

| 问题类型 | 注入策略 | 注入源 | 目标效果 |
|---------|---------|-------|---------|
| `novelty` | **Tail Injection** | Rank 5-10, Size < 10 | 引入冷门方法论 |
| `stability` | **Head Injection** | Rank 1-3, Size > 15 | 引入成熟技术 |
| `interpretability` | **Explanation** | 固定模板 | 补充可视化分析 |
| `domain_mismatch` | **Domain Adaptation** | 固定模板 | 领域特定调整 |

### 4.2 Tail Injection (长尾注入)

#### 适用场景
- **问题**: 创新性不足，技术组合常见
- **目标**: 从冷门 Pattern 提取独特方法论

#### 选择逻辑

```python
# 1. 筛选 Rank 5-10 中 cluster_size < 10 的 Pattern
candidates = [
    (pid, pinfo, pinfo.get('cluster_size'))
    for i, (pid, pinfo, _) in enumerate(recalled_patterns[4:10])
    if pinfo.get('cluster_size') < 10 and pid not in used_patterns
]

# 2. 选择 cluster_size 最小的（最冷门）
candidates.sort(key=lambda x: x[2])
selected = candidates[0]

# 3. 记录已使用
self.used_patterns.add(selected[0])
```

#### 方法论提取

**核心改进**: 提取完整的 `method_story`，而非仅 Trick 名称。

```python
# 从 skeleton_examples 提取方法论
method_insights = []
for ex in skeleton_examples[:2]:
    method_story = ex.get('method_story', '')
    if method_story:
        method_insights.append(method_story[:150])  # 截取150字

# 从 top_tricks 提取技术名称（过滤通用 Trick）
GENERIC_TRICKS = ["消融实验", "Case Study", "可视化", ...]
tech_tricks = [
    trick['name'] for trick in top_tricks[:5]
    if not any(gt in trick['name'] for gt in GENERIC_TRICKS)
][:2]
```

#### 注入指令

```python
injection_instructions = []

# 1. 注入完整方法论描述
if method_insights:
    injection_instructions.append(
        f"【方法论重构】参考 {pattern_name} 的核心技术路线：{method_insights[0]}"
    )

# 2. 补充技术名称
if tech_tricks:
    injection_instructions.append(
        f"【核心技术】融合 {pattern_name} 的关键技术点：{' + '.join(tech_tricks)}"
    )

return injection_instructions
```

#### 输出示例

```
🎯 策略: Tail Injection (长尾注入 - 深度方法论融合)

   ✅ 选择 Pattern: pattern_23
      名称: 课程学习调度
      聚类大小: 5 篇（冷门）
      注入方法论: 我们设计了一个基于样本难度的课程学习调度器...
      注入技术: 课程学习调度器 + 样本难度评估

返回:
[
  "【方法论重构】参考 课程学习调度 的核心技术路线：我们设计了一个基于样本难度的课程学习调度器。首先，通过预训练模型计算每个样本的预测置信度作为难度指标；然后，在训练早期仅使用简单样本，随训练进程逐步引入困难样本...",
  "【核心技术】融合 课程学习调度 的关键技术点：课程学习调度器 + 样本难度评估"
]
```

### 4.3 Head Injection (头部注入)

#### 适用场景
- **问题**: 技术细节不足，稳定性有待验证
- **目标**: 从成熟 Pattern 提取验证过的方法论

#### 选择逻辑

```python
# 筛选 Rank 1-3 中 cluster_size > 15 的 Pattern
candidates = [
    (pid, pinfo, pinfo.get('cluster_size'))
    for i, (pid, pinfo, _) in enumerate(recalled_patterns[:3])
    if pinfo.get('cluster_size') > 15 and pid not in used_patterns
]

# 选择 cluster_size 最大的（最成熟）
candidates.sort(key=lambda x: x[2], reverse=True)
```

#### 稳定性方法论提取

```python
# 优先提取包含稳定性关键词的 method_story
stability_keywords = ['稳定', '鲁棒', '一致', '对抗', '正则', '混合']

stability_methods = []
for ex in skeleton_examples[:3]:
    method_story = ex.get('method_story', '')
    if any(kw in method_story.lower() for kw in stability_keywords):
        stability_methods.append(method_story[:150])
        if len(stability_methods) >= 2:
            break
```

#### 注入指令

```python
injection_instructions = []

if stability_methods:
    injection_instructions.append(
        f"【稳定性方法论】参考 {pattern_name} 的鲁棒性设计：{stability_methods[0]}"
    )

if tech_tricks:
    injection_instructions.append(
        f"【稳定性技术】融合 {pattern_name} 的成熟技术：{' + '.join(tech_tricks)}"
    )
```

### 4.4 Explanation Injection & Domain Adaptation

#### Explanation Injection (固定模板)

```python
def _inject_explanation_tricks(self):
    return [
        "增加 Attention 权重可视化分析",
        "设计代表性样本的 Case Study",
        "添加消融实验说明各组件贡献"
    ]
```

#### Domain Adaptation (固定模板)

```python
def _inject_domain_tricks(self):
    return [
        "增加领域特定的数据预处理步骤",
        "设计领域相关的特征提取方法",
        "调整评估指标以适配目标领域"
    ]
```

---

## 5. 生成与注入协作流程

### 5.1 完整迭代流程图

```
Phase 1: Pattern Selection
  ├─ Conservative: pattern_11 (最高分)
  ├─ Innovative: pattern_23 (Size < 10)
  └─ Cross-Domain: pattern_17 (次高分)
         ↓
Phase 2: Initial Generation (第1轮)
  ├─ 使用: Conservative Pattern
  ├─ 输入: User Idea + Pattern 骨架
  └─ 输出: Story V1
         ↓
Phase 3: Multi-Agent Critic
  ├─ Methodology: 7.5/10 ✅
  ├─ Novelty: 4.0/10 ❌
  ├─ Storyteller: 6.5/10 ⚠️
  ├─ 平均分: 6.0/10 → 未通过
  └─ 诊断: main_issue = 'novelty'
         ↓
Phase 3.5: Refinement
  ├─ 触发: Tail Injection
  ├─ 选择: pattern_23 (Rank 6, Size 5)
  └─ 注入: 完整方法论描述
         ↓
Phase 2: Incremental Update (第2轮)
  ├─ 模式: Refinement Mode
  ├─ 输入: previous_story + review_feedback + new_tricks
  ├─ Prompt: 包含修正原则 + 正反范例
  └─ 输出: Story V2
         ↓
Phase 3: Critic (第2轮)
  ├─ Methodology: 8.0/10 ✅
  ├─ Novelty: 7.5/10 ✅
  ├─ Storyteller: 7.0/10 ✅
  └─ 平均分: 7.5/10 → 通过 ✅
         ↓
Phase 4: RAG Verification
  ├─ 最高相似度: 0.65
  └─ < 0.75 → 无撞车 ✅
         ↓
    ✅ Final Story
```

### 5.2 关键决策点

#### Pattern 切换策略

```python
# Round 1: Conservative (稳健型)
if iteration == 1:
    current_pattern = 'conservative'

# Round 2: Innovative (创新型) - 如果 Novelty 不足
elif iteration == 2 and last_issue == 'novelty':
    current_pattern = 'innovative'

# Round 3: Cross-Domain (跨域型)
elif iteration == 3:
    current_pattern = 'cross_domain'
```

#### 注入策略叠加

**重要**: 注入是**叠加**的，不会覆盖上一轮的修改。

```
Round 1: 初次生成（无注入）→ Story V1
Round 2: Tail Injection（注入 pattern_23）→ Story V2 = V1 + pattern_23
Round 3: 再次 Tail Injection（注入 pattern_29）→ Story V3 = V2 + pattern_29
```

**去重机制**: `RefinementEngine.used_patterns` 记录已使用的 Pattern。

---

## 6. 技术演进对比

### 6.1 注入内容演进

#### 旧版本: 技术堆砌

```python
# 只提取 Trick 名称
tricks = ["课程学习", "对抗训练", "温度调节"]
```

**生成的 Story**:
```
Method:
第一步：构建基础框架；
第二步：设计算法；
第三步：添加课程学习；      ← 堆砌
第四步：引入对抗训练；      ← 堆砌
```

#### 新版本: 方法论深度融合

```python
# 提取完整方法论描述
method_story = "我们设计了一个基于样本难度的课程学习调度器。首先，通过预训练模型计算每个样本的预测置信度作为难度指标；然后，在训练早期仅使用简单样本，随训练进程逐步引入困难样本..."

injection = f"【方法论重构】参考 {pattern_name} 的核心技术路线：{method_story}"
```

**生成的 Story**:
```
Method:
第一步：构建教师-学生双塔架构；
第二步：设计基于样本难度的课程学习调度器，通过预训练模型评估置信度，
       动态调整训练样本顺序，让模型从易到难学习；         ← 深度融合
第三步：将课程学习与温度调节联动，形成渐进式鲁棒框架； ← 技术组合创新
```

### 6.2 Prompt 设计演进

#### 旧版本: 简单罗列

```
【必须融合的技巧】
  - 课程学习
  - 对抗训练

请融合到方法中。
```

#### 新版本: 正反范例 + 强约束

```
【核心任务：方法论深度重构】
🔧 【方法论重构】参考 课程学习调度 的核心技术路线：
   我们设计了一个基于样本难度的...

【重构要求】
1. 深度嵌入到核心逻辑，而不是末尾添加
2. 形成技术组合创新

❌ 差的修正: "添加课程学习；再添加对抗训练"
✅ 好的修正: "引入基于难度的课程学习调度器，结合对抗扰动正则项"
```

### 6.3 数据源演进

#### 旧版本: 单一数据源

```python
# 只加载 nodes_pattern.json
patterns = load('nodes_pattern.json')
# 问题: skeleton_examples 为空
```

#### 新版本: 多源合并

```python
# 加载并合并两个文件
patterns_data = load('nodes_pattern.json')
patterns_structured = load('patterns_structured.json')

# 合并数据
for p in patterns_data:
    if p['pattern_id'] in structured_map:
        p['skeleton_examples'] = structured_map[p['pattern_id']]['skeleton_examples']
```

---

## 7. 实战案例分析

### 案例: Novelty 不足 → Tail Injection

#### 初始 Idea

```
使用蒸馏技术完成Transformer跨领域文本分类任务
```

#### 第1轮: 初次生成

**Pattern**: `pattern_11` (模型压缩与知识蒸馏)

**Story V1**:
```json
{
  "title": "自适应蒸馏框架在跨域文本分类中的应用",
  "method_skeleton": "第一步：构建教师-学生双塔架构；第二步：设计自适应温度调节器；第三步：引入域感知的蒸馏损失。",
  "innovation_claims": [
    "首次提出自适应温度调节机制",
    "设计域感知的特征对齐策略"
  ]
}
```

**评审结果**:
```
Methodology: 7.5/10 ✅
Novelty: 4.0/10 ❌ "创新性不足，温度调节和特征对齐都是常见技术"
Storyteller: 6.5/10 ⚠️

平均分: 6.0/10 → 未通过
主要问题: novelty
```

#### 第2轮: Tail Injection + 增量修正

**注入策略**: Tail Injection

**选择 Pattern**: `pattern_23` (课程学习调度, Size: 5)

**注入内容**:
```
【方法论重构】参考 课程学习调度 的核心技术路线：
我们设计了一个基于样本难度的课程学习调度器。首先，通过预训练模型计算每个样本的预测置信度作为难度指标；然后，在训练早期仅使用简单样本，随训练进程逐步引入困难样本，最终实现稳定的模型收敛。

【核心技术】融合 课程学习调度 的关键技术点：课程学习调度器 + 样本难度评估
```

**Story V2**:
```json
{
  "title": "基于课程学习的自适应跨域蒸馏框架",
  "method_skeleton": "第一步：构建教师-学生双塔架构；第二步：设计基于样本难度的课程学习调度器，通过预训练模型评估样本置信度，动态调整训练样本顺序，让模型从易到难学习跨域特征；第三步：将课程学习进度与自适应温度调节器联动，在训练早期使用高温度平滑软标签，随课程推进逐步降低温度，形成渐进式鲁棒训练框架。",
  "innovation_claims": [
    "首次将课程学习调度与跨域蒸馏深度融合，通过样本难度驱动的训练策略解决跨域知识学习不稳定问题",
    "设计课程感知的自适应温度调节机制，实现训练进度与软标签平滑度的动态联动"
  ]
}
```

**改进对比**:

| 维度 | Story V1 | Story V2 |
|------|----------|----------|
| **标题** | 自适应蒸馏框架 | 基于课程学习的自适应跨域蒸馏框架 |
| **方法步骤** | 3步（通用描述） | 3步（详细技术路线） |
| **技术融合** | 分散描述 | 课程学习 + 温度调节联动 |
| **创新点** | "首次提出温度调节" | "首次将课程学习与跨域蒸馏深度融合" |

**第2轮评审**:
```
Methodology: 8.0/10 ✅
Novelty: 7.5/10 ✅ "技术组合创新，课程学习与蒸馏的联动设计新颖"
Storyteller: 7.0/10 ✅

平均分: 7.5/10 → 通过 ✅
```

---

## 8. 调优建议

### 8.1 注入效果不佳

**问题**: 注入后 Story 仍然堆砌技术

**原因**:
1. Prompt 约束不够强
2. LLM 温度过高导致不稳定
3. 注入的方法论描述不够具体

**解决**:
```python
# 1. 降低 LLM 温度
response = call_llm(prompt, temperature=0.6, max_tokens=1500)

# 2. 增强 Prompt 约束
tricks_instruction = "【极重要：技术重构指令】\n"
tricks_instruction += "你必须利用下列技巧对核心方法进行**颠覆性重构**...\n"

# 3. 提取更具体的方法论
method_story = ex.get('method_story', '')[:200]  # 增加到200字
```

### 8.2 Pattern 资源耗尽

**问题**: 多次迭代后 `used_patterns` 包含了所有召回的 Pattern

**解决**:
```python
# 在 RefinementEngine 中实现降级策略
if not candidates:
    print("   ⚠️  所有召回 Pattern 已用尽，注入通用创新算子")
    return [
        "引入对比学习负采样优化策略",
        "设计多尺度特征融合机制",
        "添加自适应动态权重分配"
    ]
```

### 8.3 增量修正丢失字段

**问题**: 修正后某些字段变为空

**解决**: 已实现保底策略
```python
# 在 StoryGenerator.generate() 中
if previous_story:
    for key in ['title', 'abstract', ...]:
        if not story.get(key):
            story[key] = previous_story.get(key)
```

### 8.4 调整配置参数

**关键配置** (`pipeline/config.py`):

```python
# Pattern 选择
INNOVATIVE_CLUSTER_SIZE_THRESHOLD = 10  # 降低到 8 可选择更多创新 Pattern

# Critic 阈值
PASS_SCORE = 6.0  # 降低到 5.5 可更容易通过
MAX_REFINE_ITERATIONS = 3  # 增加到 5 允许更多修正

# Refinement 策略
TAIL_INJECTION_RANK_RANGE = (4, 9)  # 扩展到 (3, 12) 可选择更多长尾 Pattern
HEAD_INJECTION_CLUSTER_THRESHOLD = 15  # 降低到 12 可选择更多成熟 Pattern

# RAG 查重
COLLISION_THRESHOLD = 0.75  # 提高到 0.80 可降低撞车敏感度
```

### 8.5 并行生成优化

**当前**: 串行生成 3 个 Pattern 的 Story，选择最佳

**优化**: 并行生成（需要多线程/多进程）

```python
from concurrent.futures import ThreadPoolExecutor

def parallel_generation(patterns):
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(generator.generate, pid, pinfo)
            for pid, pinfo in patterns.items()
        ]
        stories = [f.result() for f in futures]

    # 评审后选择最佳
    best_story = max(stories, key=lambda s: critic.review(s)['avg_score'])
    return best_story
```

---

## 📚 相关文档

- `PIPELINE_IMPLEMENTATION.md` - Pipeline 实现说明
- `PIPELINE_API_REFERENCE.md` - API 参考文档
- `QUICK_START_PIPELINE.md` - 快速上手指南

---

**最后更新**: 2026-01-13
**核心改进**: 方法论深度融合 | 多维度注入策略 | 增量修正模式

