# Idea2Story Pipeline 快速上手指南

本指南帮助你在 5 分钟内运行 Idea2Story Pipeline（2026年1月最新版本）。

---

## 🆕 最新版本亮点

✨ **方法论深度融合**: 不再简单堆砌技术名词，而是从召回的 Pattern 中提取完整的方法论描述，深度融入 Story 逻辑
✨ **增量修正模式**: 保留上一轮生成的精华部分，仅针对性改造评审反馈中的问题
✨ **多源数据合并**: 自动合并 `nodes_pattern.json` 和 `patterns_structured.json`，确保访问完整的方法论数据
✨ **强约束 Prompt**: 提供正反范例，引导 LLM 实现"统一框架"而非"技术罗列"

---

## 📋 前置条件

### 1. 完成第一步（知识图谱构建）

确保已经运行了以下命令，生成了必要的数据文件：

```bash
python scripts/generate_patterns.py
python scripts/build_entity.py
python scripts/build_edges.py
```

验证 `output/` 目录下有以下文件：
- `patterns_structured.json` （关键！包含完整的 skeleton_examples）
- `nodes_pattern.json`
- `nodes_paper.json`
- `nodes_idea.json`
- `nodes_domain.json`
- `knowledge_graph_v2.gpickle`

### 2. 配置 LLM API（推荐）

```bash
# 配置 SiliconFlow API Key
export SILICONFLOW_API_KEY="sk-your-api-key-here"
export LLM_API_URL="https://api.siliconflow.cn/v1/chat/completions"
export LLM_MODEL="Qwen/Qwen2.5-7B-Instruct"
```

**如果没有 API Key**：系统会使用模拟输出，但不会生成真实的 Story 内容。

---

## 🚀 快速运行

### 方法 1: 使用默认 Idea

```bash
cd /Users/gaoge/code/Idea2Paper/Paper-KG-Pipeline
python scripts/idea2story_pipeline.py
```

默认 Idea: "使用蒸馏技术完成Transformer跨领域文本分类任务，并在多个数据集上验证效果"

### 方法 2: 自定义 Idea

```bash
python scripts/idea2story_pipeline.py "你的研究想法描述"
```

示例:
```bash
python scripts/idea2story_pipeline.py "使用对比学习改进小样本文本分类，并在医疗领域数据集上验证"
```

---

## 📊 查看输出

### 控制台输出

Pipeline 会打印详细的执行过程:

```
================================================================================
🚀 Idea2Story Pipeline 启动
================================================================================

【用户 Idea】
使用蒸馏技术完成Transformer跨领域文本分类任务...

================================================================================
📋 Phase 1: Pattern Selection (策略选择)
================================================================================

✅ [稳健型] pattern_11
   名称: 模型压缩与知识蒸馏
   聚类大小: 30 篇
   策略: Score 最高，最符合直觉

...（完整流程）
```

### 输出文件

**1. `output/final_story.json`** - 最终生成的 Story

```json
{
  "title": "自适应蒸馏框架在跨域文本分类中的应用",
  "abstract": "我们提出了一个新的自适应蒸馏框架...",
  "problem_definition": "现有的知识蒸馏方法在跨域场景下性能下降显著...",
  "method_skeleton": "第一步：构建自适应权重调节机制；第二步：...",
  "innovation_claims": [
    "首次提出自适应权重机制应对域迁移下的知识蒸馏不稳定性",
    "设计了基于难度的课程学习调度，提升小模型学习效率",
    "在5个数据集上验证效果，相比基线提升8-12%"
  ],
  "experiments_plan": "在ACL、COLING基准数据集上对比测试..."
}
```

**2. `output/pipeline_result.json`** - 完整执行历史

```json
{
  "user_idea": "...",
  "success": true,
  "iterations": 2,
  "selected_patterns": {
    "conservative": "pattern_11",
    "innovative": "pattern_23",
    "cross_domain": "pattern_17"
  },
  "review_summary": {
    "total_reviews": 2,
    "final_score": 7.5
  },
  "refinement_summary": {
    "total_refinements": 1,
    "issues_addressed": ["novelty"]
  },
  "verification_summary": {
    "collision_detected": false,
    "max_similarity": 0.62
  }
}
```

---

## 🎯 预期执行流程

### 正常情况（无需修正）

```
Phase 1: 选择 3 个 Pattern
    ↓
Phase 2: 生成初始 Story
    ↓
Phase 3: 多智能体评审 → 平均分 7.5/10 → ✅ PASS
    ↓
Phase 4: RAG 查重 → 最高相似度 0.62 → ✅ PASS
    ↓
✅ Pipeline 完成（1 次迭代）
```

### 需要修正（Novelty 不足）

```
Phase 1: 选择 3 个 Pattern
    ↓
Phase 2: 生成初始 Story
    ↓
Phase 3: 评审 → Novelty 得分 4.0/10 → ❌ FAIL
    ↓
Phase 3.5: Tail Injection（从冷门 Pattern 提取完整方法论描述）
    ↓
Phase 2: 增量修正（保留原 Story 精华，深度融合新方法论）
    ↓
Phase 3: 评审 → 平均分 7.0/10 → ✅ PASS
    ↓
Phase 4: 查重 → ✅ PASS
    ↓
✅ Pipeline 完成（2 次迭代）
```

**关键区别（新版本）**:
- 不再注入"课程学习"、"对抗训练"等技术名词
- 而是注入完整的 `method_story`（如："我们设计了一个基于样本难度的课程学习调度器。首先，通过预训练模型计算每个样本的预测置信度作为难度指标；然后，在训练早期仅使用简单样本..."）
- LLM 被引导进行"方法论重构"而非"末尾追加"

### 检测到撞车（需要 Pivot）

```
Phase 1-3: 正常流程
    ↓
Phase 4: 查重 → 相似度 0.82 → ❌ COLLISION
    ↓
Pivot: 生成约束 + 切换到创新型 Pattern
    ↓
Phase 2: 重新生成 Story（带约束）
    ↓
Phase 4: 重新查重 → ✅ PASS
    ↓
✅ Pipeline 完成（含 Pivot）
```

---

## ⚙️ 常用配置调整

如果效果不理想，可以修改 `scripts/idea2story_pipeline.py` 中的配置:

### 1. 降低评审通过门槛

```python
class PipelineConfig:
    PASS_SCORE = 5.0  # 原来是 6.0
```

### 2. 增加迭代次数

```python
class PipelineConfig:
    MAX_REFINE_ITERATIONS = 5  # 原来是 3
```

### 3. 调整查重敏感度

```python
class PipelineConfig:
    COLLISION_THRESHOLD = 0.85  # 原来是 0.75，值越大越宽松
```

### 4. 修改 Pattern 选择标准

```python
class PipelineConfig:
    INNOVATIVE_CLUSTER_SIZE_THRESHOLD = 15  # 原来是 10，值越大选择面越大
```

---

## 💡 方法论深度融合详解（重要！）

这是最新版本最核心的改进，解决了"技术堆砌"问题。

### 问题演示

**旧版本的输出**（技术堆砌）:
```
Method:
第一步：构建基础蒸馏框架；
第二步：设计温度调节机制；
第三步：添加课程学习；
第四步：引入对抗训练；
第五步：使用多种子验证
```
**问题**: 技术名词罗列，缺乏逻辑关联

### 新版本的改进

**1. 精准提取方法论描述**

从 `patterns_structured.json` 的 `skeleton_examples` 中提取 `method_story`：
```json
{
  "method_story": "我们设计了一个基于样本难度的课程学习调度器。首先，通过预训练模型计算每个样本的预测置信度作为难度指标；然后，在训练早期仅使用简单样本，随训练进程逐步引入困难样本..."
}
```

**2. 针尖式注入到 Prompt**

不再只说"请融合课程学习"，而是：
```
【新颖性方法论】参考 pattern_23 的课程学习方案：
我们设计了一个基于样本难度的课程学习调度器。首先，通过预训练模型计算每个样本的预测置信度作为难度指标；然后，在训练早期仅使用简单样本，随训练进程逐步引入困难样本...

【核心要求】：将上述方法论整合成一个连贯的技术框架
```

**3. 强约束 Prompt 提供正反范例**

```
❌ 差的修正（技术堆砌）:
   "方法步骤1；方法步骤2；添加课程学习；再添加对抗训练"

✅ 好的修正（深度融合）:
   "方法步骤1；在训练过程中引入基于难度的课程学习调度器，
    结合对抗扰动正则项，形成渐进式鲁棒训练框架；方法步骤3"
```

**新版本的输出**（深度融合）:
```
Method:
第一步：构建自适应蒸馏框架，引入温度调节机制；
第二步：在训练过程中集成基于样本难度的课程学习调度器，
       通过预训练模型评估样本置信度，动态调整训练难度曲线；
第三步：融合对抗扰动正则项到目标函数，增强跨域稳定性；
第四步：采用多种子验证机制，确保结果可复现性
```

**关键区别**:
- ✅ 技术组合成统一框架
- ✅ 具体说明实现方式
- ✅ 逻辑连贯、层次清晰

---

## 🔍 常见问题

### Q1: 为什么一直评审不通过？

**原因**:
- LLM 输出不稳定
- PASS_SCORE 设置过高
- 初始 Pattern 选择不合适

**解决**:
1. 降低 `PASS_SCORE` 到 5.0
2. 增加 `MAX_REFINE_ITERATIONS` 到 5
3. 检查 LLM API 是否配置正确

### Q2: 为什么查重总是检测到撞车？

**原因**:
- `COLLISION_THRESHOLD` 设置过低
- 测试数据集较小，相似度容易偏高

**解决**:
1. 提高 `COLLISION_THRESHOLD` 到 0.85
2. 使用更多样化的测试数据

### Q3: 如何跳过某个 Phase？

在 `Idea2StoryPipeline.run()` 中注释掉对应的阶段:

```python
# 跳过 RAG 查重
# verification_result = self.verifier.verify(current_story)
verification_result = {'pass': True, 'collision_detected': False, 'similar_papers': [], 'max_similarity': 0.0}
```

### Q4: 如何自定义评审角色？

修改 `MultiAgentCritic.__init__()`:

```python
self.reviewers = [
    {'name': 'Reviewer A', 'role': 'Methodology', 'focus': '技术合理性'},
    {'name': 'Reviewer B', 'role': 'Novelty', 'focus': '创新性'},
    {'name': 'Reviewer C', 'role': 'Storyteller', 'focus': '叙事完整性'},
    {'name': 'Reviewer D', 'role': 'Experiment', 'focus': '实验设计'},  # 新增
]
```

### Q5: 如何使用自己的 LLM？

修改 `call_llm()` 函数，适配你的 API 接口:

```python
def call_llm(prompt: str, temperature: float = 0.7, max_tokens: int = 2000) -> str:
    # 替换为你的 LLM API 调用逻辑
    response = your_llm_api.call(prompt=prompt, ...)
    return response
```

---

## 📈 性能优化建议

### 1. 并行生成多个 Story

修改 `Idea2StoryPipeline.run()`，同时生成 3 个 Pattern 的 Story:

```python
# Phase 2: 并行生成
stories = []
for pattern_type, (pattern_id, pattern_info) in selected_patterns.items():
    story = self.story_generator.generate(pattern_id, pattern_info)
    stories.append((pattern_type, story))

# 评审后选择最佳
best_story = max(stories, key=lambda x: self.critic.review(x[1])['avg_score'])
```

### 2. 缓存 LLM 输出

添加缓存机制避免重复调用:

```python
import hashlib
import json

cache = {}

def call_llm_cached(prompt: str, **kwargs) -> str:
    key = hashlib.md5(prompt.encode()).hexdigest()
    if key in cache:
        return cache[key]

    result = call_llm(prompt, **kwargs)
    cache[key] = result
    return result
```

### 3. 增量 Refinement（已实现）

**当前版本已支持增量修正**:
- `StoryGenerator.generate()` 支持 `previous_story` 和 `review_feedback` 参数
- 修正时保留上一轮的精华部分，仅针对性改造评审反馈中的问题
- Prompt 中包含"保留精华、深度改造差评部分"的明确指令

**相关代码**: `scripts/idea2story_pipeline.py` 行 203-277（`StoryGenerator.generate()`）

---

## 🎓 进阶使用

### 1. 批量处理多个 Idea

```python
ideas = [
    "Idea 1 描述",
    "Idea 2 描述",
    "Idea 3 描述",
]

for i, idea in enumerate(ideas):
    print(f"\n处理 Idea {i+1}/{len(ideas)}")
    pipeline = Idea2StoryPipeline(idea, recalled_patterns, papers)
    result = pipeline.run()

    # 保存结果
    with open(f"output/story_{i+1}.json", 'w') as f:
        json.dump(result['final_story'], f, ensure_ascii=False, indent=2)
```

### 2. 人机协同模式

在关键节点加入人工审核:

```python
# 在 Phase 3 后加入人工审核
if not critic_result['pass']:
    print("\n⚠️  评审未通过，是否继续修正？(y/n)")
    choice = input().strip().lower()
    if choice != 'y':
        print("用户选择终止")
        break
```

### 3. 导出为 Markdown

```python
def export_story_to_markdown(story: Dict, filename: str):
    md_content = f"""# {story['title']}

## Abstract
{story['abstract']}

## Problem Definition
{story['problem_definition']}

## Method
{story['method_skeleton']}

## Innovation Claims
{chr(10).join([f"- {claim}" for claim in story['innovation_claims']])}

## Experiments
{story['experiments_plan']}
"""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(md_content)
```

---

## 📚 相关文档

- `docs/PIPELINE_IMPLEMENTATION.md` - 实现细节和设计思路
- `docs/IDEA_TO_STORY_PIPELINE.md` - 原始设计方案
- `scripts/test_pipeline.py` - 单元测试脚本

---

**最后更新**: 2026-01-12

**重要改进**:
- 方法论深度融合（从技术堆砌到深度重构）
- 多源数据合并（访问完整的 skeleton_examples）
- 增量修正模式（保留精华、针对性改造）
- 强约束 Prompt（提供正反范例引导 LLM）

