# Reflection → Story 重新生成逻辑修复

## 问题描述

用户发现在 `Phase 3.6: Story Reflection (故事反思融合)` 之后，系统**缺少根据 Reflection 建议重新生成 Story 的步骤**，直接跳到了下一个 Pattern 的 Critic 评审。

### 原始流程（有缺陷）

```
1. Refinement → 生成初步融合的 Story
2. Reflection → 评估融合质量 + 给出改进建议
3. ❌ 直接进入 Critic 评审（未使用 Reflection 建议）
```

### 期望流程

```
1. Refinement → 生成初步融合的 Story
2. Reflection → 评估融合质量 + 给出改进建议
3. ✅ 根据 Reflection 建议重新生成 Story
4. Critic 评审优化后的 Story
```

---

## 修复内容

### 1. `manager.py` - 添加基于 Reflection 的 Story 终稿生成

**位置**：`Phase 3.6: Story Post-Generation Reflection` 后

**【关键流程修正 - 2025-01-25】**

新颖性 Pattern 注入的正确流程应该是：
```
Critic评审 → 选取新颖Pattern → Idea Fusion → Story Gen初稿 → Reflection反思 → Story Gen终稿 → Critic评审
```

**核心修正**：**无论融合质量如何，Reflection 后都应该生成 Story 终稿**

**修复逻辑**：

```python
# 【关键修正】无论融合质量如何，都应该根据Reflection建议生成Story终稿
# 这是新颖性Pattern注入的核心步骤：初稿 → Reflection → 终稿
print(f"\n🔄 Step 2: 根据Reflection建议生成Story终稿...")

# 提取Reflection建议
fusion_suggestions = reflection_result.get('fusion_suggestions', {})

# 将Reflection建议注入到Story生成的约束中
enhanced_constraints = dict(constraints)
enhanced_constraints['reflection_guidance'] = fusion_suggestions

# 重新生成Story（终稿），传入Reflection建议
new_story = self.story_generator.generate(
    pattern_id, pattern_info, enhanced_constraints, injected_tricks,
    previous_story=new_story,  # 基于初稿进行改进
    review_feedback=critic_result,
    fused_idea=fused_idea,
    reflection_guidance=fusion_suggestions  # 传入Reflection建议
)

print(f"   ✅ Story终稿已根据Reflection建议生成")

# 【关键判断】如果融合质量极低（< 0.5），在新颖性模式下可以选择跳过Critic直接尝试下一个Pattern
# 但这应该是可选的优化策略，不应阻止终稿生成
if fusion_quality < 0.5 and novelty_mode_active and current_pattern_id:
    print(f"\n   ⚠️  融合质量极低 (< 0.5)，可能不适合此Pattern")
    print(f"   💡 提示: 将继续Critic评审，但如果失败可快速切换到下一个Pattern")
```

**关键改变**：
- **之前**：只有当 `fusion_quality >= 0.65` 时才生成终稿
- **现在**：**总是生成终稿**，融合质量分数仅用于诊断
- **原因**：终稿生成是流程的必要步骤，不应被融合质量分数阻断

**融合质量的正确用途**：
- `>= 0.65`：融合良好，打印优势信息
- `< 0.65`：融合不佳，打印警告和诊断信息
- `< 0.5`：融合极差，提示可能需要切换 Pattern（但仍完成终稿生成 + Critic 评审）

---

### 2. `story_generator.py` - 支持 `reflection_guidance` 参数

#### 2.1 修改函数签名

```python
def generate(self, pattern_id: str, pattern_info: Dict,
             constraints: Optional[List[str]] = None,
             injected_tricks: Optional[List[str]] = None,
             previous_story: Optional[Dict] = None,
             review_feedback: Optional[Dict] = None,
             new_tricks_only: Optional[List[str]] = None,
             fused_idea: Optional[Dict] = None,
             reflection_guidance: Optional[Dict] = None) -> Dict:  # 新增参数
    """生成 Story (支持初次生成和增量修正，支持 idea fusion 和 reflection 指导)"""
```

#### 2.2 修改 `_build_refinement_prompt`

**函数签名**：

```python
def _build_refinement_prompt(self, previous_story: Dict,
                           review_feedback: Dict,
                           new_tricks: List[str],
                           pattern_info: Dict,
                           fused_idea: Optional[Dict] = None,
                           reflection_guidance: Optional[Dict] = None) -> str:  # 新增参数
```

**Prompt 增强**：

在 `{fused_idea_guidance}` 和 `{tricks_instruction}` 之间插入：

```python
# 【新增】Reflection 指导（来自融合质量评估）
reflection_guidance_text = ""
if reflection_guidance:
    reflection_guidance_text = "\n【🎯 CRITICAL: Reflection Guidance from Fusion Quality Assessment】\n"
    reflection_guidance_text += "The Story Reflector has analyzed the fusion quality and provided the following strategic guidance:\n\n"

    title_evolution = reflection_guidance.get('title_evolution', '')
    method_evolution = reflection_guidance.get('method_evolution', '')
    narrative_strategy = reflection_guidance.get('narrative_strategy', '')

    if title_evolution:
        reflection_guidance_text += f"📝 Title Evolution Strategy:\n   {title_evolution}\n\n"
    if method_evolution:
        reflection_guidance_text += f"🔧 Method Evolution Strategy:\n   {method_evolution}\n\n"
    if narrative_strategy:
        reflection_guidance_text += f"📖 Narrative Strategy:\n   {narrative_strategy}\n\n"

    reflection_guidance_text += "⚠️ IMPORTANT: These guidance points are based on analyzing the fusion between your current Story and the new Pattern.\n"
    reflection_guidance_text += "Follow these strategies to ensure the fusion creates genuine conceptual innovation, not just technical stacking.\n"
```

**Prompt 结构**（新的顺序）：

```
【Review Feedback】 (Critic 评审反馈)
    ↓
【Fused Idea Guidance】 (融合后的概念创新)
    ↓
【Reflection Guidance】 (反思建议 - 新增！)
    ↓
【Tricks Instruction】 (注入的技术)
    ↓
【Pattern Reference】 (Pattern 参考信息)
```

---

## Reflection 建议的内容结构

`story_reflector.py` 生成的 `fusion_suggestions` 包含：

```python
{
    'title_evolution': '彻底重新定义问题视角，避免使用Pattern的常见术语',
    'method_evolution': '从问题假设层面重构方法，而不是在技术层面组合',
    'narrative_strategy': '⚠️ Critic已警告: 避免技术堆砌！需要展示**为什么这个组合创造了新的研究视角**，而不是"A+B"。'
}
```

**当有 Critic 警告时**（通过 `_extract_critic_warnings` 识别）：
- 自动建议激进的重构策略
- 强调避免技术堆砌
- 引用 Critic 的具体反馈

---

## 工作流程图

### 完整的 Refinement → Reflection → Regeneration 流程

```
┌─────────────────────────────────────────────────────┐
│  Phase 3.5: Refinement (创新融合修正)                  │
│  ├─ 选择新颖性 Pattern                                │
│  ├─ Idea Fusion: 融合 Old Story + New Pattern        │
│  └─ 生成初步融合的 Story                              │
└────────────────┬────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────┐
│  Phase 3.6: Story Post-Generation Reflection         │
│  ├─ 提取 Critic 警告 (_extract_critic_warnings)       │
│  ├─ 分析融合点 (_analyze_fusion_points)               │
│  ├─ 检查逻辑连贯性 (_check_coherence)                 │
│  │   └─ 如果有 Critic 警告 → 降低连贯性分数           │
│  ├─ 评估融合质量 (_assess_fusion_quality)             │
│  │   └─ 如果有多个警告 → 额外降低质量分              │
│  └─ 生成融合建议 (_generate_fusion_suggestions)       │
│      └─ 如果有 Critic 警告 → 建议激进重构策略         │
└────────────────┬────────────────────────────────────┘
                 │
                 ▼
         fusion_quality >= 0.65?
                 │
         ┌───────┴───────┐
         NO              YES
         │               │
         ▼               ▼
    标记失败        【新增】根据 Reflection 建议
    尝试下一个       重新生成 Story
    Pattern              │
                         ▼
                 调用 story_generator.generate()
                 传入 reflection_guidance
                         │
                         ▼
                 在 Prompt 中注入:
                 - Title Evolution Strategy
                 - Method Evolution Strategy
                 - Narrative Strategy
                         │
                         ▼
                 ✅ 生成优化后的 Story
                         │
                         ▼
               进入 Critic 评审阶段
```

---

## 关键改进点

### 1. **闭环反馈机制**

- **之前**：Reflection 评估融合质量，但不影响 Story 生成
- **现在**：Reflection 的建议直接指导 Story 重新生成

### 2. **Critic 警告传递**

- **之前**：Critic 的负面评价只存储在 `review_feedback` 中
- **现在**：通过 `_extract_critic_warnings` 提取关键警告 → 降低融合质量分 → 生成激进重构建议 → 传递给 Story Generator

### 3. **分数惩罚机制**

- 如果 Critic 警告中包含"堆砌"、"stacking"、"A+B" → 连贯性 `-0.15`
- 如果包含"常见"、"typical"、"套路" → 连贯性 `-0.10`
- 如果有 ≥2 个警告 → 最终质量分额外 `-0.10`

### 4. **Prompt 结构优化**

将 Reflection 建议放在 Prompt 的显著位置（在 Fused Idea 之后，Tricks Instruction 之前），确保 LLM 优先考虑融合策略而非简单技术堆砌。

---

## 测试验证

### 预期效果

1. **日志中应该出现**：

```
🔄 根据Reflection建议重新生成Story...
   ✅ Story已根据Reflection建议重新生成
```

2. **新生成的 Story 应该体现**：
   - 标题避免 Pattern 的常见术语
   - Method 从问题假设层面重构（而非技术堆砌）
   - Claims 强调"为什么这个组合创造了新的研究视角"

3. **Critic 评审分数提升**：
   - 特别是 Novelty 维度应该有显著提升

---

## 相关文件

- `scripts/pipeline/manager.py` (L258-295)
- `scripts/pipeline/story_generator.py` (L14-21, L114-236)
- `scripts/pipeline/story_reflector.py` (已完成 Critic 警告提取和惩罚逻辑)

---

## 额外修复：增强 Idea Fusion 指导的可见性和强调

### 问题发现

在验证 Reflection 修复时，发现了另一个问题：
- **Idea Fusion 生成的 `fused_idea` 虽然被传递给 Story Generator，但在 Prompt 中缺少明确的使用指导**
- 日志中没有打印融合概念的详细信息，无法验证是否生效

### 修复内容

#### 1. 添加调试输出（`story_generator.py` L26-32）

在增量修正模式下，添加融合概念和反思建议的打印：

```python
# 【新增】打印关键指导信息（用于验证融合是否生效）
if fused_idea:
    print(f"   💡 融合概念: {fused_idea.get('fused_idea_title', 'N/A')}")
    print(f"   📝 新颖性声明: {fused_idea.get('novelty_claim', 'N/A')[:80]}...")
if reflection_guidance:
    print(f"   🎯 反思建议: 标题策略={bool(reflection_guidance.get('title_evolution'))}, 方法策略={bool(reflection_guidance.get('method_evolution'))}")
```

**预期日志输出**：

```
📝 修正 Story (基于上一轮反馈 + 新注入技巧)
   💡 融合概念: Dynamic Multilingual Reasoning through Context-Filtered Knowledge Inheritance
   📝 新颖性声明: This fusion does not merely stack multilingual reasoning and retrieval...
   🎯 反思建议: 标题策略=True, 方法策略=True
   ⏳ 调用 LLM 生成...
```

#### 2. 添加明确的使用指导（`story_generator.py` L268-275）

在 Prompt 中增加 **`【HOW TO USE Fused Idea Guidance】`** 部分：

```
⚠️ 【HOW TO USE Fused Idea Guidance】
If you received 【Conceptual Innovation from Idea Fusion】 above, this is THE MOST IMPORTANT guidance:
- **Title & Abstract**: Must reflect the fused conceptual innovation, not just list techniques
- **Problem Framing**: Adopt the NEW problem perspective from the fused idea
- **Gap Pattern**: Explain why existing methods lack this conceptual unity
- **Innovation Claims**: Frame as "transforming/reframing X from Y to Z", NOT "combining A with B"
- **Method**: Show how techniques CO-EVOLVE to realize the fused concept, not just CO-EXIST
```

**放置位置**：在 `{pattern_reference}` 之后，`【HOW TO USE Pattern Information】` 之前，确保 LLM 优先看到融合概念的使用指导。

### 为什么需要这个修复？

1. **可验证性**：之前无法从日志中确认融合概念是否真的传递给了 LLM
2. **指导明确性**：虽然 `fused_idea_guidance` 中有说明，但分散在各个字段中，不如集中的使用指导清晰
3. **优先级强调**：通过 "THE MOST IMPORTANT guidance" 和显眼的位置，确保 LLM 优先考虑融合概念而非简单技术堆砌

### Prompt 最终结构

```
【Review Feedback】 (Critic 评审反馈)
    ↓
【💡 CRITICAL: Conceptual Innovation from Idea Fusion】
  - Title: XXX
  - Description: XXX
  - New Problem Framing: XXX
  - New Assumption: XXX
  - Why NOT Simple Combination: XXX
  - Key Innovation Points: [1, 2, 3]
    ↓
【🎯 CRITICAL: Reflection Guidance】
  - Title Evolution Strategy: XXX
  - Method Evolution Strategy: XXX
  - Narrative Strategy: XXX
    ↓
【核心任务：概念级创新融合】(Tricks Instruction)
    ↓
【Pattern Reference】
    ↓
⚠️ 【HOW TO USE Fused Idea Guidance】← 新增！强调融合概念的使用方式
    ↓
⚠️ 【HOW TO USE Pattern Information】
    ↓
【Refinement Principles】
```

---

## 后续优化建议

1. **动态调整重新生成次数**：如果第一次重新生成后 Critic 仍然不满意，可以再次尝试（目前只重新生成一次）
2. **保存 Reflection 历史**：记录每次 Reflection 的建议和效果，用于后续分析
3. **增强 Reflection 建议的具体性**：可以让 Reflector 不仅给出策略，还给出具体的标题/方法示例
4. **监控融合概念的应用效果**：通过日志分析，验证添加 Fused Idea Guidance 后 Novelty 分数的提升幅度

