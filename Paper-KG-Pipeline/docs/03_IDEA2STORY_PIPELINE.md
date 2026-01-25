# Idea2Story Pipeline 文档

## 📋 概述

本文档详细说明了从用户Idea到可发表Paper Story的完整生成链路,包括Pattern选择、Idea Fusion、Story生成、Critic评审、智能修正机制、参数配置和运行方式。

---

## 1. 系统架构

### 1.1 整体流程

```
┌─────────────────────────────────────────────────────────────────┐
│                  【Idea2Story Pipeline 完整流程】                 │
└─────────────────────────────────────────────────────────────────┘

用户输入Idea
    │
    ▼
【阶段1: Pattern选择与分类】(约1秒)
    │
    ├─ 召回Top-10 Pattern (来自召回系统)
    │   └─ 路径1(相似Idea) + 路径2(领域) + 路径3(相似Paper)
    │
    ├─ Pattern多维度分类
    │   ├─ Stability (稳健型): Rank前3 + Cluster Size≥15
    │   ├─ Novelty (新颖型): Cluster Size<10
    │   └─ Cross-Domain (跨域型): 不同Domain来源
    │
    └─ 选择初始Pattern (优先Stability维度)
    │
    ▼
【阶段2: Story生成】(约1-2分钟)
    │
    └─ 基于Pattern生成初稿Story
        ├─ 使用skeleton_examples作为模板
        ├─ 注入common_tricks
        └─ 结构化输出(7个字段)
    │
    ▼
【阶段3: Critic评审】(约30秒)
    │
    └─ 多角色评审 (并行)
        ├─ Methodology Critic: 技术可行性/严谨性
        ├─ Novelty Critic: 创新性/问题新颖性
        └─ Storyteller Critic: 叙事连贯性/可读性
        │
        └─ 计算平均分 (avg_score)
    │
    ▼
【阶段4: 判断分支】
    │
    ├─【判断1】评分 >= 7.0?
    │   ├─【是】→ 进入阶段5: RAG查重
    │   └─【否】→ 进入阶段4.1或4.2
    │
    ├─【判断2】新颖性停滞? (novelty_score <= last + 0.5)
    │   ├─【是】→ 阶段4.1: 新颖性模式
    │   └─【否】→ 阶段4.2: 普通修正
    │
    ├─────────────────────────────────────────────────────────────┐
    │              【阶段4.1: 新颖性模式】(3-10分钟)               │
    ├─────────────────────────────────────────────────────────────┤
    │                                                               │
    │  遍历Novelty维度的Pattern (最多10个)                         │
    │      │                                                        │
    │      ├─ For each novelty_pattern:                           │
    │      │                                                        │
    │      ├─ 1. Idea Fusion (概念融合)                           │
    │      │     ├─ 输入: user_idea + current_story + pattern     │
    │      │     ├─ LLM分析: 概念A, 概念B, 融合方式               │
    │      │     └─ 输出: fused_idea (融合后的新Idea)             │
    │      │                                                        │
    │      ├─ 2. Story Reflection (质量评估)                      │
    │      │     ├─ 输入: fused_idea + current_story              │
    │      │     ├─ 评估4个维度                                   │
    │      │     │   ├─ concept_unity: 概念统一性 [0-10]          │
    │      │     │   ├─ technical_soundness: 技术可行性 [0-10]    │
    │      │     │   ├─ novelty_level: 新颖性 [0-10]              │
    │      │     │   └─ narrative_clarity: 叙事清晰度 [0-10]      │
    │      │     └─ 输出: fusion_score + suggestions              │
    │      │                                                        │
    │      ├─ 3. 重新生成Story                                    │
    │      │     └─ 基于fused_idea + reflection_guidance         │
    │      │                                                        │
    │      ├─ 4. Critic评审                                       │
    │      │     └─ 获取新的avg_score                             │
    │      │                                                        │
    │      ├─ 5. 分数退化检测                                     │
    │      │     └─ 如果 avg_score < last_score - 0.1:           │
    │      │         ├─ 回滚到上一版本                            │
    │      │         ├─ 标记Pattern失败                           │
    │      │         └─ 跳过该Pattern                             │
    │      │                                                        │
    │      ├─ 6. 记录最佳结果                                     │
    │      │     └─ 如果 avg_score > best_score:                 │
    │      │         └─ 更新best_score和best_story                │
    │      │                                                        │
    │      ├─ 7. 通过检查                                         │
    │      │     └─ 如果 avg_score >= 7.0:                       │
    │      │         └─ 提前结束,进入阶段5                        │
    │      │                                                        │
    │      └─ 循环结束                                            │
    │           │                                                   │
    │           └─ 兜底: 返回best_story (最高分版本)              │
    │                                                               │
    └─────────────────────────────────────────────────────────────┘
    │
    ├─────────────────────────────────────────────────────────────┐
    │              【阶段4.2: 普通修正】(1-2分钟)                  │
    ├─────────────────────────────────────────────────────────────┤
    │                                                               │
    │  智能注入互补Tricks                                          │
    │      │                                                        │
    │      ├─ 分析Critic反馈                                      │
    │      │   ├─ novelty_score < 6.0 → 缺新颖性                 │
    │      │   ├─ methodology_score < 6.0 → 缺稳健性              │
    │      │   └─ storyteller_score < 6.0 → 缺叙事性              │
    │      │                                                        │
    │      ├─ 选择互补Pattern                                     │
    │      │   ├─ 缺新颖性 → 长尾注入 (Rank 5-10, Novelty类)     │
    │      │   ├─ 缺稳健性 → 头部注入 (Rank 1-3, Stability类)    │
    │      │   └─ 缺叙事性 → 跨域注入 (Cross-Domain类)            │
    │      │                                                        │
    │      └─ 返回阶段2 (重新生成Story)                           │
    │                                                               │
    └─────────────────────────────────────────────────────────────┘
    │
    ▼
【阶段5: RAG查重】(约30秒)
    │
    ├─ 提取关键方法 (techniques)
    │
    ├─ 检索近3年顶会论文 (Embedding召回)
    │
    ├─ 计算相似度
    │
    └─ 判断: 相似度 > 0.75?
        ├─【否】→ 输出Final Story ✅
        └─【是】→ Pivot规避
                  ├─ 分析撞车点
                  ├─ 生成约束 (禁用技术/领域迁移)
                  └─ 返回阶段2
    │
    ▼
输出Final Story (JSON格式)
```

**流程说明**:
- **阶段1-2**: 基础生成链路
- **阶段3**: 质量评估
- **阶段4**: 核心修正机制(两种模式)
  - **新颖性模式**: 深度探索,Fusion+Reflection
  - **普通修正**: 快速注入,互补增强
- **阶段5**: 查重验证

### 1.2 核心模块

| 模块 | 文件 | 作用 |
|------|------|------|
| **Pattern Selector** | `pattern_selector.py` | 多维度Pattern分类与排序 |
| **Story Generator** | `story_generator.py` | 结构化Story生成 |
| **Idea Fusion** | `planner.py` | 融合新Pattern生成创新Idea |
| **Story Reflector** | `story_reflector.py` | 反思融合质量 |
| **Multi-Agent Critic** | `critic.py` | 三角色评审 |
| **Refinement Engine** | `refinement.py` | 智能修正与注入 |
| **RAG Verifier** | `verifier.py` | 查重与规避 |
| **Pipeline Manager** | `manager.py` | 流程编排 |

---

## 2. Pattern选择与分类

### 2.1 多维度分类

**目标**: 将召回的Top-10 Pattern按3个维度分类,确保多样性。

**维度定义**:

| 维度 | 定义 | 选择标准 | 作用 |
|------|------|---------|------|
| **Stability** | 稳健型 | Rank Top-3 + Cluster Size ≥ 15 | 保证基础质量,降低风险 |
| **Novelty** | 新颖型 | Cluster Size < 10 | 提升创新性 |
| **Cross-Domain** | 跨域型 | 来自路径2/3 + Domain不同于Top-1 | 引入跨领域视角 |

**算法**:

```python
def classify_patterns(recalled_patterns, user_idea):
    """多维度分类Pattern"""
    classified = {
        'stability': [],
        'novelty': [],
        'cross_domain': []
    }

    for rank, (pattern_id, pattern_info, score) in enumerate(recalled_patterns):
        metadata = {
            'rank': rank,
            'recall_score': score,
            'cluster_size': pattern_info.get('size', 0)
        }

        # 维度1: Stability (稳健型)
        if rank <= 2 and metadata['cluster_size'] >= 15:
            classified['stability'].append((pattern_id, pattern_info, metadata))

        # 维度2: Novelty (新颖型)
        if metadata['cluster_size'] < 10:
            classified['novelty'].append((pattern_id, pattern_info, metadata))

        # 维度3: Cross-Domain (跨域型)
        if rank >= 3:  # 来自路径2/3
            user_domain = extract_domain(user_idea)
            pattern_domain = pattern_info.get('domain', '')
            if pattern_domain != user_domain:
                classified['cross_domain'].append((pattern_id, pattern_info, metadata))

    return classified
```

### 2.2 Pattern选择策略

```python
# 优先级顺序
1. Stability 维度第一个 (保证基础质量)
2. Novelty 维度第一个 (如果stability为空)
3. Cross-Domain 维度第一个 (兜底)
```

---

## 3. Story生成机制

### 3.1 Story数据结构

```json
{
  "title": "论文标题",
  "abstract": "摘要(150-200词)",
  "problem_definition": "明确的问题定义",
  "gap_pattern": "研究缺口描述",
  "method_skeleton": {
    "overview": "方法概述",
    "core_components": ["组件1", "组件2", "组件3"],
    "technical_details": "技术细节"
  },
  "innovation_claims": [
    "贡献点1",
    "贡献点2",
    "贡献点3"
  ],
  "experiments_plan": {
    "datasets": ["数据集1", "数据集2"],
    "baselines": ["基线方法1", "基线方法2"],
    "metrics": ["评估指标1", "指标2"],
    "ablation_studies": "消融实验设计"
  }
}
```

### 3.2 生成Prompt构建

**初稿生成Prompt**:
```python
def _build_initial_prompt(user_idea, pattern_info):
    prompt = f"""
你是一个顶级AI研究员。请基于以下信息生成一篇ICLR水平的论文Story。

【用户Idea】
{user_idea}

【Pattern指导】
名称: {pattern_info['name']}
代表性想法: {pattern_info['llm_enhanced_summary']['representative_ideas']}
常见问题: {pattern_info['llm_enhanced_summary']['common_problems']}
解决方法: {pattern_info['llm_enhanced_summary']['solution_approaches']}
故事框架: {pattern_info['llm_enhanced_summary']['story']}

【任务】
生成完整的论文Story(JSON格式),包含:
- title: 吸引人的标题
- abstract: 150-200词摘要
- problem_definition: 明确问题定义
- gap_pattern: 研究缺口
- method_skeleton: 方法骨架(overview + core_components + technical_details)
- innovation_claims: 3个核心贡献
- experiments_plan: 实验设计(datasets/baselines/metrics/ablation_studies)
"""
    return prompt
```

**Refinement Prompt**:
```python
def _build_refinement_prompt(story, critic_result, fused_idea, reflection_guidance):
    prompt = f"""
【当前Story】
{json.dumps(story, indent=2)}

【Critic评审结果】
Methodology: {critic_result['methodology']['score']}/10
  问题: {critic_result['methodology']['issues']}

Novelty: {critic_result['novelty']['score']}/10
  问题: {critic_result['novelty']['issues']}

【融合创新指导】
{format_fused_idea(fused_idea)}

【Reflection建议】
{format_reflection_guidance(reflection_guidance)}

⚠️ 【HOW TO USE Fused Idea Guidance】
- **Title & Abstract**: 必须反映融合后的概念创新,而非技术堆砌
- **Problem Framing**: 采用融合idea中的新问题视角
- **Gap Pattern**: 解释为什么现有方法缺乏这种概念统一性
- **Innovation Claims**: 框架为"transforming/reframing X from Y to Z"
- **Method**: 展示技术如何共同演化(CO-EVOLVE)而非共存(CO-EXIST)

【任务】
修正Story,重点解决上述问题,生成改进版JSON。
"""
    return prompt
```

---

## 4. Idea Fusion机制

### 4.1 融合目标

**问题**: 直接拼接Pattern会导致"技术堆砌",缺乏概念统一性。

**目标**: 生成一个**有机融合**的新Idea,使新Pattern与原Idea在**概念层面**统一。

### 4.2 Fusion Prompt

```python
def plan_idea_fusion(user_idea, current_story, new_pattern_info, critic_issues):
    prompt = f"""
你是一个创新研究规划师。请分析如何将新Pattern融合到现有研究中。

【当前研究】
Idea: {user_idea}
Story: {extract_key_points(current_story)}

【新Pattern】
{format_pattern(new_pattern_info)}

【Critic指出的问题】
{critic_issues}

【融合任务】
生成一个融合后的Idea,要求:

1. **概念统一**: 找到新Pattern与原Idea的概念连接点
2. **问题重构**: 重新框架问题,使新Pattern成为自然解决方案
3. **创新点**: 明确融合后的独特贡献

返回JSON:
{
  "fused_core_idea": "融合后的核心想法(单句话)",
  "conceptual_bridge": "概念桥梁:如何连接原Idea和新Pattern",
  "reframed_problem": "重构后的问题定义",
  "innovation_angle": "独特创新点",
  "implementation_hints": ["实现提示1", "提示2"]
}
"""
    return prompt
```

### 4.3 示例

**原Idea**:
```
使用大模型做数据增强
```

**新Pattern**: 课程学习(Curriculum Learning)

**Fusion结果**:
```json
{
  "fused_core_idea": "基于LLM生成的难度自适应课程学习框架",
  "conceptual_bridge": "LLM不仅生成数据,更重要的是可以评估样本难度,从而构建个性化学习路径",
  "reframed_problem": "如何让模型像人类一样从易到难地学习LLM生成的伪标签数据",
  "innovation_angle": "首次将LLM的生成能力和难度评估能力统一在课程学习框架中",
  "implementation_hints": [
    "LLM为每个生成样本打上难度标签",
    "设计难度感知的样本调度器",
    "渐进式训练策略"
  ]
}
```

---

## 5. Story Reflection机制

### 5.1 反思目标

**问题**: Fusion生成了融合Idea,但Story生成器可能:
- 未充分理解融合意图
- 生成了"生硬拼接"而非"有机融合"

**目标**: 在Story生成后,反思融合质量,评估是否真正实现了概念统一。

### 5.2 Reflection流程

```python
def reflect_on_fusion(fused_idea, generated_story):
    """反思融合质量"""
    # 1. 分析融合点
    fusion_points = analyze_fusion_points(fused_idea, generated_story)

    # 2. 检查连贯性
    coherence = check_conceptual_coherence(fusion_points)

    # 3. 评估融合丰富度
    richness = evaluate_fusion_richness(fused_idea, generated_story)

    # 4. 计算质量分数
    quality = 0.4 * coherence + 0.4 * richness + 0.2 * has_fusion_idea_bonus

    # 5. 生成改善建议
    suggestions = generate_improvement_suggestions(quality, fusion_points)

    return {
        'fusion_quality': quality,
        'fusion_points': fusion_points,
        'coherence_score': coherence,
        'fusion_richness': richness,
        'fusion_suggestions': suggestions
    }
```

### 5.3 质量评分

```python
fusion_quality = 0.4 × 连贯性 + 0.4 × 融合丰富度 + 0.2 × Fusion Idea奖励

# 连贯性: 融合点在Story各部分是否连贯出现
coherence_score = len(连贯的融合点) / len(所有融合点)

# 融合丰富度: Story中多少部分体现了融合
richness_score = len(体现融合的Story部分) / len(Story总部分)

# Fusion Idea奖励: 是否使用了fused_idea指导
fusion_idea_bonus = 1.0 if fused_idea else 0.5
```

**阈值**: `fusion_quality >= 0.65` 认为融合成功

---

## 6. Critic评审机制

### 6.1 三角色评审

| 角色 | 关注点 | 评分标准 |
|------|--------|---------|
| **Reviewer A** (Methodology) | 技术合理性、实验完整性 | 方法可行性、实验设计 |
| **Reviewer B** (Novelty) | 创新性、贡献独特性 | 问题新颖度、方法创新度 |
| **Reviewer C** (Storyteller) | 叙事完整性、逻辑连贯性 | 结构完整、逻辑清晰 |

### 6.2 Critic Prompt

```python
def build_critic_prompt(story, role):
    if role == "methodology":
        focus = """
评审重点:
1. 方法是否技术合理?
2. 实验设计是否完整?
3. 是否存在技术风险?
"""
    elif role == "novelty":
        focus = """
评审重点:
1. 问题定义是否新颖?
2. 方法是否有独特创新?
3. 是否仅是技术堆砌?
"""
    elif role == "storyteller":
        focus = """
评审重点:
1. 逻辑是否连贯?
2. 叙事是否完整?
3. 读者能否理解?
"""

    prompt = f"""
你是一个ICLR审稿人,专注于{role}。

【论文Story】
{json.dumps(story, indent=2)}

{focus}

【任务】
返回JSON评审结果:
{{
  "score": 7,  # 1-10分
  "issues": ["问题1", "问题2"],
  "suggestions": ["建议1", "建议2"]
}}
"""
    return prompt
```

### 6.3 通过标准

```python
PASS_SCORE = 7.0

# 所有三个维度的平均分 >= 7.0
avg_score = (methodology_score + novelty_score + storyteller_score) / 3
if avg_score >= PASS_SCORE:
    return "PASS"
else:
    return "FAIL"
```

---

## 7. 智能修正机制

### 7.1 新颖性模式

**触发条件**:
```python
# 新颖性分数停滞
if novelty_score <= last_novelty_score + 0.5:
    activate_novelty_mode()
```

**工作流程**:
```python
def novelty_mode(ranked_patterns):
    """新颖性模式: 遍历所有novelty维度的Pattern"""
    novelty_patterns = ranked_patterns['novelty']
    best_score = 0
    best_story = None

    for pattern in novelty_patterns[:NOVELTY_MODE_MAX_PATTERNS]:
        # 1. Idea Fusion
        fused_idea = plan_idea_fusion(user_idea, current_story, pattern)

        # 2. Story Reflection
        reflection_result = reflect_on_fusion(fused_idea, current_story)

        # 3. 生成终稿Story
        new_story = generate_story(
            pattern,
            fused_idea=fused_idea,
            reflection_guidance=reflection_result['fusion_suggestions']
        )

        # 4. Critic评审
        critic_result = critic.review(new_story)

        # 5. 分数退化检测
        if critic_result['avg_score'] < last_avg_score - 0.1:
            # 回滚
            rollback()
            mark_failure(pattern)
            continue

        # 6. 记录最高分
        if critic_result['avg_score'] > best_score:
            best_score = critic_result['avg_score']
            best_story = new_story

        # 7. 通过检查
        if critic_result['avg_score'] >= PASS_SCORE:
            return new_story

    # 8. 兜底: 返回最高分版本
    return best_story
```

### 7.2 分数退化回滚

**检测条件**:
```python
# 任一维度分数下降超过0.1
if (new_methodology_score < old_methodology_score - 0.1 or
    new_novelty_score < old_novelty_score - 0.1 or
    new_storyteller_score < old_storyteller_score - 0.1):
    trigger_rollback()
```

**回滚流程**:
```python
def rollback():
    """回滚到上一个版本"""
    # 1. 恢复Story
    current_story = last_story_before_refinement

    # 2. 标记失败Pattern
    pattern_failure_map[pattern_id].add(issue_type)

    # 3. 删除注入的Tricks
    injected_tricks.remove(failed_trick)

    # 4. 继续迭代(不增加iterations计数)
```

### 7.3 普通修正模式

**触发条件**: 新颖性未停滞,但评分未通过

**Critic诊断与Pattern维度映射**: 系统将Critic的三个评审角色直接映射到Pattern的三个分类维度,实现统一的修正策略。

| Critic角色 | 评审焦点 | 诊断问题类型 | 映射Pattern维度 | 注入策略 |
|-----------|---------|------------|----------------|---------|
| **Novelty** | 创新性 | `novelty` | **Novelty维度** | 从novelty维度按序选择Pattern,注入创新方法 |
| **Methodology** | 技术合理性 | `stability` | **Stability维度** | 从stability维度按序选择Pattern,注入稳健方法 |
| **Storyteller** | 叙事完整性 | `domain_distance` | **Domain Distance维度** | 从domain_distance维度选择Pattern,引入跨域视角 |

**核心设计理念**:
- **统一映射**: Critic的诊断结果直接映射到Pattern的三个分类维度,避免额外的启发式规则
- **维度一致**: Pattern Selector已按三个维度(稳健度、新颖度、跨域度)对所有Pattern排序,Refinement Engine直接复用这些排序结果
- **策略简化**: 不再需要"解释性注入"、"领域适配注入"等额外策略,所有修正统一通过Pattern维度选择实现

**注入逻辑**:
```python
def refine_with_idea_fusion(main_issue: str, suggestions: List[str],
                            previous_story: Optional[Dict] = None) -> Tuple[List[str], Optional[Dict]]:
    """基于Critic诊断,从对应Pattern维度选择并融合"""

    # Step 1: 维度映射
    dimension_map = {
        'novelty': 'novelty',          # Novelty Critic → Novelty维度
        'stability': 'stability',      # Methodology Critic → Stability维度
        'domain_distance': 'domain_distance'  # Storyteller Critic → Domain Distance维度
    }
    dimension = dimension_map[main_issue]

    # Step 2: 从对应维度选择Pattern
    patterns = ranked_patterns[dimension]
    idx = dimension_indices[dimension]  # 维度内的当前索引

    while idx < len(patterns):
        pattern_id, pattern_info, metadata = patterns[idx]

        # 跳过已失败的Pattern
        if is_pattern_failed_for_issue(pattern_id, main_issue):
            idx += 1
            continue

        # Step 3: Idea Fusion
        fused_result = fusion_engine.fuse(
            user_idea=user_idea,
            pattern_id=pattern_id,
            pattern_info=pattern_info,
            previous_story=previous_story
        )

        # Step 4: 返回融合结果
        return injected_tricks, fused_result
```

**示例场景**:
```
场景: Storyteller Critic给出低分(叙事不连贯)
→ 诊断: domain_distance
→ 选择: 从domain_distance维度(按领域距离升序排列)选择Pattern
→ 效果: 引入不同领域的叙事视角,丰富Story结构
```

---

## 8. RAG查重与规避

### 8.1 查重流程

```python
def verify_collision(story):
    """RAG查重"""
    # 1. 提取关键方法
    method_keywords = extract_method_keywords(story)

    # 2. 构建Query
    query = f"{method_keywords} {story['problem_definition']}"

    # 3. 检索近3年顶会论文
    similar_papers = retrieve_similar_papers(query, top_k=10)

    # 4. 计算相似度
    for paper in similar_papers:
        similarity = compute_similarity(story, paper)
        if similarity > COLLISION_THRESHOLD:
            return {
                'collision': True,
                'collided_paper': paper,
                'similarity': similarity
            }

    return {'collision': False}
```

### 8.2 Pivot规避策略

**触发条件**: `similarity > 0.75`

**规避流程**:
```python
def pivot_to_avoid_collision(story, collided_paper):
    """生成规避约束"""
    # 1. 撞车分析
    collision_analysis = analyze_collision(story, collided_paper)

    # 2. 生成约束
    constraints = {
        'forbidden_techniques': collision_analysis['overlapping_techniques'],
        'pivot_direction': "迁移到无监督设定",
        'domain_shift': "从通用领域迁移到法律文本",
        'additional_constraint': "增加长文本处理模块"
    }

    # 3. 重新生成Story
    new_story = generate_story(pattern, constraints=constraints)

    return new_story
```

---

## 9. 参数配置

### 9.1 Pipeline配置

```python
# scripts/pipeline/config.py

class PipelineConfig:
    """Pipeline配置参数"""

    # Pattern选择
    SELECT_PATTERN_COUNT = 3              # 选择3个不同策略的Pattern
    CONSERVATIVE_RANK_RANGE = (0, 2)      # 稳健型: Rank 1-3
    INNOVATIVE_CLUSTER_SIZE_THRESHOLD = 10 # 创新型: Cluster Size < 10

    # Critic阈值
    PASS_SCORE = 7.0                      # 评分 >= 7 为通过
    MAX_REFINE_ITERATIONS = 3             # 最多修正3轮(普通模式)

    # 新颖性模式配置
    NOVELTY_MODE_MAX_PATTERNS = 10        # 新颖性模式最多尝试的Pattern数
    NOVELTY_SCORE_THRESHOLD = 6.0         # 新颖性得分阈值
    NOVELTY_STAGNATION_DELTA = 0.5        # 停滞判定阈值

    # Reflection配置
    FUSION_QUALITY_THRESHOLD = 0.65       # 融合质量阈值

    # 回滚配置
    SCORE_DEGRADATION_THRESHOLD = 0.1     # 分数下降阈值

    # RAG查重阈值
    COLLISION_THRESHOLD = 0.75            # 相似度 > 0.75 认为撞车

    # Refinement策略
    TAIL_INJECTION_RANK_RANGE = (4, 9)    # 长尾注入: Rank 5-10
    HEAD_INJECTION_RANK_RANGE = (0, 2)    # 头部注入: Rank 1-3
    HEAD_INJECTION_CLUSTER_THRESHOLD = 15 # 头部注入: Cluster Size > 15
```

### 9.2 LLM配置

```python
# scripts/pipeline/config.py

LLM_API_KEY = os.getenv("SILICONFLOW_API_KEY")
LLM_API_URL = "https://api.siliconflow.cn/v1/chat/completions"
LLM_MODEL = "Qwen/Qwen3-14B"  # 可选: Qwen2.5-7B-Instruct
```

---

## 10. 运行方式

### 10.1 完整Pipeline运行

**命令**:
```bash
cd /Users/gaoge/code/mycode/Idea2Paper/Paper-KG-Pipeline
python scripts/idea2story_pipeline.py "你的研究Idea描述"
```

**示例**:
```bash
python scripts/idea2story_pipeline.py "使用强化学习优化大模型的推理效率"
```

**输出**:
```
output/
├── final_story.json          # 最终生成的论文Story
├── pipeline_result.json      # 完整流程结果
└── log.json                  # 详细日志
```

### 10.2 输出结构

**final_story.json**:
```json
{
  "title": "Efficient LLM Reasoning via Reinforcement Learning...",
  "abstract": "We propose...",
  "problem_definition": "...",
  "gap_pattern": "...",
  "method_skeleton": {...},
  "innovation_claims": [...],
  "experiments_plan": {...}
}
```

**pipeline_result.json**:
```json
{
  "success": true,
  "final_story": {...},
  "iterations": 5,
  "selected_patterns": {
    "stability": [...],
    "novelty": [...],
    "cross_domain": [...]
  },
  "review_history": [
    {
      "iteration": 1,
      "methodology": {"score": 6.0, "issues": [...]},
      "novelty": {"score": 5.5, "issues": [...]},
      "storyteller": {"score": 7.0, "issues": []},
      "avg_score": 6.17
    },
    ...
  ],
  "refinement_history": [
    {
      "iteration": 2,
      "action": "idea_fusion",
      "pattern": "pattern_42",
      "fusion_quality": 0.72,
      "result": "success"
    },
    ...
  ]
}
```

### 10.3 监控关键指标

**新颖性模式激活**:
```bash
grep "激活【新颖性模式】" output/log.json
```

**融合质量评分**:
```bash
grep "融合质量评分" output/log.json
```

**回滚事件**:
```bash
grep "【ROLLBACK TRIGGERED】" output/log.json
```

**最终通过情况**:
```bash
grep "🎉 Critic 评审通过" output/log.json
```

---

## 11. 流程详细示例

### 11.1 场景A: 新颖性停滞触发新模式

**初始状态**:
```
Iteration 1: Novelty Score = 5.5
Iteration 2: Novelty Score = 5.6 (仅提升0.1 < 0.5)
→ 触发新颖性模式
```

**新颖性模式流程**:
```
1. 激活新颖性模式
2. 遍历Novelty Pattern列表 (最多10个)

  Pattern 1 (pattern_42):
    ├─ Idea Fusion: 生成融合Idea
    ├─ Story Reflection: 融合质量评分0.72
    ├─ 生成终稿Story (基于reflection建议)
    ├─ Critic评审: 6.5/10 (未通过)
    └─ 继续下一个Pattern

  Pattern 2 (pattern_55):
    ├─ Idea Fusion: 生成融合Idea
    ├─ Story Reflection: 融合质量评分0.68
    ├─ 生成终稿Story
    ├─ Critic评审: 7.2/10 (通过!)
    └─ 进入RAG查重

3. RAG查重: 未撞车
4. 输出Final Story
```

### 11.2 场景B: 分数退化触发回滚

```
Iteration 3:
  当前分数: Methodology=7.0, Novelty=6.0, Storyteller=7.5

  注入Pattern_30:
    ├─ Idea Fusion: ...
    ├─ 生成新Story
    ├─ Critic评审: Methodology=6.2 (下降0.8 > 0.1)
    ├─ 检测到分数退化
    └─ 触发回滚

  回滚操作:
    ├─ 恢复Story到注入前版本
    ├─ 标记Pattern_30失败
    ├─ 删除注入的Tricks
    └─ 继续迭代(不增加计数)

  选择下一个Pattern: Pattern_45
    ├─ Idea Fusion: ...
    ├─ 生成新Story
    ├─ Critic评审: Methodology=7.3 (提升)
    └─ 保存结果
```

---

## 12. 最终版本选择机制

### 12.1 全局最优追踪

**设计理念**: 在整个迭代过程中,每一轮生成的Story可能有不同的优劣,系统需要记录并最终选择最优版本。

**核心机制**:
```python
# 每轮Critic评审后更新全局最佳版本
if current_avg_score > global_best_score:
    global_best_story = current_story
    global_best_score = current_avg_score
    global_best_iteration = iteration_number
    print(f"🏆 更新全局最佳版本: 得分 {global_best_score:.2f}")
```

### 12.2 最终输出逻辑

**优先级规则**:
1. **优先**: 如果有通过Critic评审的版本(avg_score >= 7.0) → 使用通过版本
2. **兜底**: 如果没有通过版本 → 使用全局最佳版本(迭代中得分最高)

**实现流程**:
```python
# 最终版本选择
final_story = current_story  # 默认当前版本
final_is_passed = review_history[-1]['pass']

if not final_is_passed and global_best_story is not None:
    # 未通过但有最佳版本
    if global_best_score > current_score:
        final_story = global_best_story  # 使用最佳版本
        print(f"✅ 使用全局最佳版本(迭代 {global_best_iteration}, 得分 {global_best_score:.2f})")
```

### 12.3 典型场景

**场景A: 逐步提升,最终通过**
```
迭代1: 初稿 → 6.17分 → 更新最佳版本
迭代2: 注入Novelty Pattern → 6.85分 → 更新最佳版本
迭代3: 继续优化 → 7.20分 → 通过! ✅
→ 输出: 迭代3的通过版本
```

**场景B: 起伏波动,未通过**
```
迭代1: 初稿 → 6.17分 → 更新最佳版本
迭代2: 注入Pattern → 6.85分 → 更新最佳版本
迭代3: 回滚后优化 → 6.50分 → 未更新
→ 输出: 迭代2的最佳版本(6.85分)
```

**场景C: 新颖性模式遍历**
```
新颖性模式:
  Pattern 1 → 6.50分 → 更新最佳版本
  Pattern 2 → 6.35分 → 未更新
  Pattern 3 → 6.80分 → 更新最佳版本
  Pattern 4 → 7.10分 → 通过! ✅
→ 输出: Pattern 4的通过版本
```

### 12.4 输出信息

**pipeline_result.json**:
```json
{
  "success": true,
  "final_story": {...},
  "final_story_source": {
    "iteration": 2,
    "score": 6.85,
    "is_best_across_iterations": true
  },
  "iterations": 3,
  "review_history": [...]
}
```

**日志输出**:
```
🎯 最终版本选择逻辑
================================================================================
📊 当前版本: 平均分=6.50, 状态=未通过
🏆 全局最佳版本: 平均分=6.85 (迭代 2)

✅ 使用全局最佳版本作为最终输出（得分更高）
================================================================================

🎉 Pipeline 完成!
================================================================================
✅ 状态: 需人工审核
📊 迭代次数: 3
🏆 最终版本来源: 迭代 2
📝 最终 Story:
   标题: ...
   摘要: ...
================================================================================
```

---

## 13. 故障排查

### 13.1 常见问题

**Q: 新颖性模式遍历完所有Pattern仍未通过**
```
原因: 所有Novelty Pattern都不适配
解决: 兜底策略自动选择最高分版本输出
检查: output/log.json中"兜底策略"关键词
```

**Q: Fusion质量评分总是低于0.65**
```
原因: Pattern与Idea概念距离过大
解决:
1. 检查Pattern选择是否合理
2. 调整FUSION_QUALITY_THRESHOLD (0.65 → 0.60)
3. 改进Fusion Prompt
```

**Q: 回滚频繁发生**
```
原因: 注入的Pattern导致分数下降
检查:
1. pattern_failure_map记录了哪些Pattern失败
2. 是否某些Pattern与Idea完全不兼容
解决: 优化Pattern选择策略
```

**Q: RAG查重总是撞车**
```
原因: Idea本身与现有工作高度重合
解决: Pivot策略生成规避约束
检查: 是否需要调整COLLISION_THRESHOLD (0.75 → 0.80)
```

### 13.2 调试模式

**启用详细日志**:
```python
# 在manager.py中添加
import logging
logging.basicConfig(level=logging.DEBUG)
```

**打印中间结果**:
```python
# 在关键步骤添加print
print(f"[DEBUG] Fused Idea: {fused_idea}")
print(f"[DEBUG] Reflection Quality: {reflection_result['fusion_quality']}")
print(f"[DEBUG] Critic Scores: {critic_result}")
```

---

## 14. 性能优化

### 14.1 并行生成

```python
from concurrent.futures import ThreadPoolExecutor

def parallel_story_generation(patterns):
    """并行生成多个Story"""
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(generate_story, p) for p in patterns]
        results = [f.result() for f in futures]
    return results
```

### 14.2 缓存LLM响应

```python
import hashlib
import json

cache = {}

def cached_llm_call(prompt):
    """缓存LLM调用结果"""
    key = hashlib.md5(prompt.encode()).hexdigest()
    if key in cache:
        return cache[key]

    result = call_llm(prompt)
    cache[key] = result
    return result
```

---

## 15. 总结

### 核心成果

✅ **完整的Idea2Story Pipeline**: 从用户Idea到可发表Story
✅ **Idea Fusion机制**: 实现Pattern的有机融合而非生硬拼接
✅ **Story Reflection**: 确保融合质量,评估概念统一性
✅ **智能修正**: 新颖性模式+分数退化回滚+兜底策略
✅ **多角色Critic**: 三维度评审,全面评估Story质量
✅ **RAG查重**: 避免与现有工作撞车

### 技术特性

✅ **自适应迭代**: 根据评审结果自动选择修正策略
✅ **质量保障**: 多层次质量检查(Reflection+Critic+RAG)
✅ **容错机制**: 回滚+失败标记+兜底策略
✅ **全局最优追踪**: 每轮迭代记录最佳版本,最终输出通过版本或最高分版本
✅ **统一诊断映射**: Critic三角色直接映射到Pattern三维度,实现架构一致性
✅ **完整日志**: 详细记录每一步决策和结果

### 创新点

✅ **概念层面融合**: Idea Fusion关注概念统一而非技术拼接
✅ **融合质量反思**: Story Reflector评估融合效果
✅ **新颖性优先**: 停滞时自动升级为新颖性模式
✅ **智能回滚**: 避免无效修正,提高迭代效率

---

**生成时间**: 2026-01-25
**版本**: V1.0
**作者**: Idea2Paper Team

