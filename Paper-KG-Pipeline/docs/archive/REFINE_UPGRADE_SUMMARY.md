# Refine 系统升级总结

## 🎯 四大核心升级

### ① 创新性优先机制
**问题**：新颖性评分停滞，无法有效改进
**方案**：
- 检测新颖性得分停滞 (score <= prev + 0.5)
- 激活【新颖性模式】，可突破最大迭代次数
- 按 novelty 维度遍历所有 Pattern
- 每个 Pattern 都进行 Idea Fusion + Story 生成 + 评审

**关键代码**：
```python
# manager.py 中的新颖性模式检测
if iterations >= 2 and curr_novelty_score <= prev_novelty_score + 0.5:
    novelty_mode_active = True  # 激活新颖性模式
    novelty_pattern_iterations = 0
    novelty_pattern_results = []
```

---

### ② 分数退化检测与回滚
**问题**：注入 Pattern 后某维度分数反而下降，浪费迭代次数
**方案**：
- 每轮修正后比较该维度的分数变化
- 如果分数下降 > 0.1，自动回滚
- 标记该 Pattern 对该 issue 的失败
- 通知 RefinementEngine 后续跳过该 Pattern

**关键代码**：
```python
# manager.py 中的分数退化检测
if len(review_history) > 0 and last_issue_type:
    curr_score = next((r['score'] for r in critic_result['reviews']
                       if r['role'] == last_issue_type), 0)
    prev_score = next((r['score'] for r in review_history[-1]['reviews']
                       if r['role'] == last_issue_type), 0)

    if curr_score < prev_score - 0.1:  # 触发回滚
        # Step 1: 恢复 Story
        current_story = last_story_before_refinement
        # Step 2: 标记 Pattern 失败
        self.refinement_engine.mark_pattern_failed(pattern_id, issue_type)
        # Step 3: 删除注入的 Tricks
        # Step 4: 继续下一轮
        continue  # 跳过本轮的评审历史记录
```

---

### ③ Story 反思融合机制
**问题**：新 Pattern 注入后与旧 Story 生硬拼接，缺乏逻辑连贯性
**方案**：
- 在生成 Story 前进行融合质量评估
- 分析融合点、检查连贯性、评估质量
- 生成融合建议指导 Story 生成
- 质量评分 >= 0.65 才被认为是良好融合

**关键模块**：`StoryReflector`
```python
# story_reflector.py
class StoryReflector:
    def reflect_on_fusion(self, old_story, new_pattern, fused_idea, ...):
        # Step 1: 分析融合点
        fusion_analysis = self._analyze_fusion_points(...)
        # Step 2: 检查连贯性
        coherence_check = self._check_coherence(...)
        # Step 3: 评估质量
        quality_score = self._evaluate_fusion_quality(...)
        # Step 4: 生成建议
        suggestions = self._generate_fusion_suggestions(...)

        return {
            'fusion_quality_score': quality_score,
            'is_organic': quality_score >= 0.65,
            'coherence_analysis': ...,
            'ready_for_generation': quality_score >= 0.65
        }
```

**融合质量评分公式**：
```
score = 0.4 * coherence_score
      + 0.4 * fusion_richness
      + 0.2 * fused_idea_bonus
```

---

### ④ 兜底策略
**问题**：新颖性模式遍历所有 Pattern 但仍未达到目标分，不知道用哪个版本
**方案**：
- 记录所有尝试的结果
- 从中找出最高分的版本
- 使用该版本作为最终输出
- 即使未达到 7.0，也能保证输出质量最佳

**关键代码**：
```python
# manager.py 中的兜底策略
if novelty_mode_active and not review_history[-1]['pass']:
    # 找出最高分
    best_score = max([r['avg_score'] for r in review_history])
    best_idx = next((i for i, r in enumerate(review_history)
                     if r['avg_score'] == best_score), -1)
    if best_idx >= 0:
        best_novelty_result = review_history[best_idx]
        print(f"✅ 最高分: {best_score:.2f}/10，使用该版本")
```

---

## 🔄 关键流程变化

### 原流程（3 轮固定迭代）
```
初始 Story → 评审 → 修正 → 评审 → 修正 → 评审 → 完成
        ↓      ↓      ↓      ↓      ↓      ↓
       Iter1  Iter2  Iter3  结束
```

### 新流程（新颖性模式可无限）
```
初始 Story → 评审 → 检测新颖性停滞
        ↓                      ↓
       Iter1          激活新颖性模式
                            ↓
                    Pattern 1 → Fusion → Story → 评审
                    Pattern 2 → Fusion → Story → 评审
                    Pattern 3 → Fusion → Story → 评审
                    ...
                    Pattern N → Fusion → Story → 评审
                            ↓
                     达到目标 or 用尽所有 Pattern
                            ↓
                    兜底：选最高分版本
```

---

## 📁 新增和修改的文件

### 新增文件
- **`scripts/pipeline/story_reflector.py`** (311 行)
  - StoryReflector 类实现反思融合机制

### 修改文件
- **`scripts/pipeline/manager.py`**
  - 新增新颖性模式逻辑
  - 新增分数退化检测与回滚
  - 新增 Story 反思融合集成
  - 新增兜底策略

- **`scripts/pipeline/refinement.py`**
  - `refine_with_idea_fusion()` 新增 `force_next_pattern` 参数
  - `_select_pattern_for_fusion()` 支持新颖性模式的循环遍历
  - 新增 `mark_pattern_failed()` 记录失败 Pattern

- **`scripts/pipeline/story_generator.py`**
  - 新增 `_build_reflection_fusion_guidance()` 方法

- **`scripts/pipeline/config.py`**
  - 新增 `NOVELTY_MODE_MAX_PATTERNS` 配置
  - 新增 `NOVELTY_SCORE_THRESHOLD` 配置

---

## ⚡ 使用示例

### 场景 1：检测到新颖性停滞
```
Iteration 2 评审结果：novelty 5.6/10
Iteration 3 评审结果：novelty 5.7/10 (停滞 <= 5.6 + 0.5)

→ 激活新颖性模式
→ 尝试 Pattern 106 (novelty 维度排名 1)
  → Idea Fusion：生成融合想法
  → Story Reflector：检查融合质量 (0.72/1.0 良好)
  → Story Generator：生成修正 Story
  → Critic：评分 6.2/10 (有改善)

→ 尝试 Pattern 107 (novelty 维度排名 2)
  → ... 评分 6.5/10

→ 尝试 Pattern 73 (novelty 维度排名 3)
  → ... 评分 6.8/10 (通过 7.0 检查!)

→ 进入 RAG 查重阶段
```

### 场景 2：分数下降触发回滚
```
Iteration 1: stability 评分 7.0/10 (通过)
Iteration 2: 注入 Pattern 16，评分 6.8/10 (下降 > 0.1)

→ 检测到分数下降
→ 回滚 Story 到 Iteration 1 版本
→ 标记 Pattern 16 对 stability 无效
→ 删除本轮注入的 Tricks
→ 继续 Iteration 3，选择新的 Pattern

Iteration 3: 注入 Pattern 73，评分 7.2/10 (成功!)
```

### 场景 3：兜底策略
```
新颖性模式尝试了 10 个 Pattern：
  Pattern 106: 6.1/10
  Pattern 107: 6.3/10
  Pattern 73:  6.5/10
  Pattern 89:  6.8/10  ← 最高分
  Pattern 90:  6.6/10
  ...

→ 未达到 7.0 目标
→ 兜底策略启动
→ 选择 Pattern 89 的版本 (6.8/10) 作为最终输出
→ 进入 RAG 查重
```

---

## 📊 监控要点

1. **新颖性模式是否激活**
   - 日志中出现"激活【新颖性模式】"

2. **融合质量评分**
   - >= 0.65 为良好融合
   - < 0.5 需要关注

3. **回滚次数**
   - 如果频繁回滚，说明 Pattern 选择有问题
   - 应该调整评分权重或 Pattern 筛选标准

4. **最终选中的版本**
   - 应该是新颖性模式中的最高分
   - 或者是正常流程中的最后一个通过评审的

---

## 🔧 调试技巧

### 打印关键变量
```python
print(f"novelty_mode_active: {novelty_mode_active}")
print(f"pattern_failure_map: {pattern_failure_map}")
print(f"reflection_result: {reflection_result}")
print(f"best_novelty_result: {best_novelty_result}")
```

### 追踪 Pattern 选择
```
在 refinement.py 中添加：
print(f"当前索引: {self.dimension_indices['novelty']}")
print(f"已使用 Pattern: {self.used_patterns}")
print(f"已失败 Pattern: {self.pattern_failure_map}")
```

### 追踪分数变化
```python
print(f"前一轮 {last_issue_type}: {prev_score:.1f}")
print(f"本轮 {last_issue_type}: {curr_score:.1f}")
print(f"差异: {curr_score - prev_score:.1f}")
```

---

## ✅ 验证清单

运行 Pipeline 时检查：
- [ ] 初始 Story 生成成功
- [ ] 第一轮评审完成
- [ ] 如果新颖性停滞，是否激活了新颖性模式
- [ ] 是否进行了 Story 反思融合
- [ ] 是否有回滚日志（如果有分数下降）
- [ ] 最终输出使用了哪个版本
- [ ] 进入 RAG 查重阶段

---

## 🚀 性能对标

| 指标 | 原系统 | 新系统 | 改善 |
|------|------|------|------|
| 最大迭代次数 | 3 轮固定 | 无限（新颖性模式） | 更灵活 |
| 无效修正处理 | 继续尝试 | 回滚 + 标记失败 | 提高效率 |
| 融合质量检查 | 无 | 反思融合评分 | 更可控 |
| 最坏情况输出 | 可能很差 | 选最高分版本 | 有保障 |

---

## 📞 常见问题

**Q: 为什么新颖性模式没有激活？**
A: 检查 novelty 分数变化是否满足 `score <= prev + 0.5` 的条件

**Q: 回滚后为什么分数还是低？**
A: 可能是 Pattern 选择不当，考虑调整 Pattern 排序逻辑

**Q: 融合质量评分为什么这么低？**
A: 检查 Idea Fusion 是否生成了有意义的融合想法

**Q: 兜底策略如何选择最佳版本？**
A: 简单地取所有版本中的最高平均分

---

## 🎓 最佳实践

1. **启用详细日志**：设置 verbose=True 跟踪流程
2. **定期检查 Pattern 失败映射**：了解哪些 Pattern 不起作用
3. **调整融合质量阈值**：根据实际情况设置 0.65 的门槛
4. **定期更新 Pattern 库**：加入新的高质量 Pattern

---

