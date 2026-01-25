# 新颖性模式链路逻辑修复

## 问题描述

从 `log_updated.json` 中观察到，系统在激活新颖性模式后，日志显示：
- 迭代轮次: 3 (新颖性模式 - 遍历Pattern #1)
- 迭代轮次: 4 (新颖性模式 - 遍历Pattern #1)

这说明：
1. **迭代次数仍在递增**：`iterations` 变量在新颖性模式下继续增长，但应该保持在同一个"逻辑迭代"内
2. **模式没有正确遍历**：每次循环都显示为 `Pattern #1`，说明内部遍历逻辑未生效

## 预期行为

当触发新颖性模式后，系统应该：
1. **固定迭代计数**：在遍历所有新颖性pattern期间，对外显示的迭代次数保持不变（如"迭代轮次: 3"）
2. **内部遍历计数**：使用独立的 `novelty_pattern_iterations` 计数器追踪已尝试的pattern数量
3. **独立遍历循环**：进入新颖性模式后，在独立的循环内完成所有pattern的尝试：
   - Refine → Fusion → Generate → Reflect → (如果质量不足则回滚并continue)
   - 如果质量足够 → Critic → (如果不通过则continue下一个pattern)
   - 如果通过 → 退出新颖性模式，break主循环
4. **回滚机制**：
   - 如果post-gen reflection评分 < 0.65，立即标记pattern失败，回滚，尝试下一个（无需Critic）
   - 如果reflection通过但critic不通过，记录该pattern结果，尝试下一个
5. **防止重复触发**：新颖性模式只能被触发一次（通过 `novelty_mode_base_iteration is None` 检查）

## 修复方案

### 1. 引入 `novelty_mode_base_iteration`

```python
novelty_mode_base_iteration = None  # 记录触发新颖性模式时的迭代次数
```

当激活新颖性模式时：
```python
novelty_mode_base_iteration = iterations  # 记录基准迭代次数
```

在输出时使用：
```python
if novelty_mode_active:
    print(f"🔄 迭代轮次: {novelty_mode_base_iteration} (新颖性模式 - 遍历Pattern #{novelty_pattern_iterations + 1})")
```

### 2. 重构Critic评审位置

#### 原逻辑问题
```
while loop:
    iterations += 1
    Critic Review (外层)  ← 新颖性模式下这里会重复评审
    ...
    Refinement
    Story Generation
    Reflection (如果质量不足 continue)
    Accept Story
    (循环结束，回到外层Critic)  ← 导致重复计数
```

#### 新逻辑
```
while loop:
    iterations += 1

    if not novelty_mode_active:
        Critic Review (外层) ← 只用于非新颖性模式
        if pass: break

    Refinement (检测是否激活新颖性模式)
    Story Generation

    if fused_idea:
        Post-Gen Reflection
        if quality < 0.65 and novelty_mode_active:
            mark_failed()
            continue  ← 立即跳过，不做Critic评审

    Accept Story

    if novelty_mode_active:
        novelty_pattern_iterations += 1
        内部Critic评审  ← 独立评审环节
        记录结果到 novelty_pattern_results
        if pass:
            退出新颖性模式
            break
        if 达到最大次数:
            退出新颖性模式
        else:
            continue  ← 继续下一个pattern
```

### 3. 关键修改点

#### A. 防止新颖性模式重复触发
```python
# 只在首次检测到时激活，避免重复触发
if iterations >= 1 and main_issue == 'novelty' and not novelty_mode_active and novelty_mode_base_iteration is None:
    # 激活新颖性模式
    novelty_mode_active = True
    novelty_mode_base_iteration = iterations  # 记录基准迭代次数
```

关键点：
- 添加 `novelty_mode_base_iteration is None` 检查
- 确保新颖性模式在整个pipeline运行期间只被触发一次
- 触发后，`novelty_mode_base_iteration` 被设置为当前迭代次数，后续不会再满足触发条件

#### B. 没有更多Pattern时退出循环
```python
if novelty_mode_active and main_issue == 'novelty' and not fused_idea:
    print(f"\n   ⚠️  没有更多新颖性Pattern可用")
    print("   退出新颖性模式，准备启用兜底策略")
    novelty_mode_active = False
    break  # 跳出当前循环，进入兜底策略
```

#### C. 外层Critic只处理非新颖性模式
```python
# 【说明】在新颖性模式下，Critic评审已在story生成后立即执行
if not novelty_mode_active:
    review_history.append(critic_result)

if critic_result['pass'] and not novelty_mode_active:
    print("\n✅ 评审通过，进入查重验证阶段")
    break
```

#### D. 融合质量不足时的快速回滚（新颖性模式）
```python
if fusion_quality < 0.65:
    if novelty_mode_active and current_pattern_id:
        print(f"\n   ❌ 融合质量不足，标记 {current_pattern_id} 对 {main_issue} 失败")
        self.refinement_engine.mark_pattern_failed(current_pattern_id, main_issue)
        print(f"   ↩️  回滚并立即尝试下一个Pattern（无需Critic评审）")
        continue  # 跳过story接受和critic评审
```

#### E. 生成后立即进行内部Critic评审（新颖性模式）
```python
if novelty_mode_active and main_issue == 'novelty':
    novelty_pattern_iterations += 1
    print(f"🔍 Phase 3: Multi-Agent Critic (评审Pattern #{novelty_pattern_iterations})")

    new_critic_result = self.critic.review(current_story)

    novelty_pattern_results.append({
        'iteration': novelty_mode_base_iteration,
        'pattern_id': current_pattern_id,
        'avg_score': new_critic_result['avg_score'],
        'novelty_score': ...,
        'story': dict(current_story)
    })

    if new_critic_result['pass']:
        review_history.append(new_critic_result)
        novelty_mode_active = False
        break

    if novelty_pattern_iterations >= MAX:
        novelty_mode_active = False
    else:
        continue  # 尝试下一个pattern
```

### 4. RefinementEngine的配合修复

在 `refinement.py` 中，`_select_pattern_for_fusion` 方法需要正确递增索引：

```python
if force_next:
    idx = self.dimension_indices['novelty']

while idx < len(novelty_patterns):
    pattern_id, pattern_info, metadata = novelty_patterns[idx]

    if self._is_pattern_failed_for_issue(pattern_id, main_issue):
        idx += 1
        self.dimension_indices['novelty'] = idx
        continue

    if force_next or pattern_id not in self.used_patterns:
        self.used_patterns.add(pattern_id)
        self.current_pattern_id = pattern_id
        self.dimension_indices['novelty'] = idx + 1  # 更新索引
        return (pattern_id, pattern_info)
    idx += 1
```

## 预期日志输出

修复后，日志应显示为：

```
================================================================================
🔄 迭代轮次: 3 (新颖性模式 - 遍历Pattern #1)
================================================================================

🔧 Phase 3.5: Refinement (创新融合修正)
   🔄 选中 Pattern: pattern_106 - Adaptive Dynamic Reasoning

💡 Phase: Idea Fusion (Conceptual Innovation Fusion)
   ✅ Fusion Complete

🔄 准备重新生成 Story...
   ✅ JSON 解析成功

🔍 Phase 3.6: Story Post-Generation Reflection
   📊 融合质量评分: 0.78/1.0
   ✅ 融合质量良好

🔍 Phase 3: Multi-Agent Critic (评审Pattern #1)
   📊 新颖性Pattern尝试 #1:
      Pattern: pattern_106
      平均分: 6.23/10
      新颖度: 5.5/10
   ❌ 评审未通过
   🔄 继续尝试下一个新颖性Pattern...

================================================================================
🔄 迭代轮次: 3 (新颖性模式 - 遍历Pattern #2)  ← 注意：仍是迭代3
================================================================================

🔧 Phase 3.5: Refinement (创新融合修正)
   🔄 选中 Pattern: pattern_73 - Reframing Retrieval

...（重复上述流程）
```

## 关键改进点总结

1. ✅ **防止重复触发**：新颖性模式只能被激活一次（`novelty_mode_base_iteration is None`检查）
2. ✅ **迭代计数修复**：在新颖性模式下显示固定的基准迭代次数
3. ✅ **独立遍历循环**：新颖性模式在独立循环内完成，不与常规refinement混合
4. ✅ **双重评审消除**：外层Critic只处理非新颖性模式，内部Critic专门处理新颖性模式
5. ✅ **快速回滚**：融合质量不足时立即回滚，不进入Critic评审
6. ✅ **pattern遍历逻辑**：`force_next` 模式下正确递增索引
7. ✅ **结果记录**：所有新颖性pattern尝试都记录到 `novelty_pattern_results`
8. ✅ **优雅退出**：没有更多pattern时break循环，进入兜底策略

## 测试建议

运行pipeline后，验证日志中：
1. 新颖性模式下的迭代次数保持不变
2. Pattern编号正确递增 (#1, #2, #3...)
3. 低质量fusion直接回滚，不进入critic
4. 每个pattern只评审一次
5. 达到最大次数后正确退出

