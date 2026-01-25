# Idea2Story 核心链路

本文档描述了从 **User Idea** 到 **Final Paper Story** 的完整生成链路。

本方案在基础的 "Generate -> Critic -> RAG" 流程上进行了增强，重点完善了 **Refinement (迭代修正)** 机制，确保 Idea 不被轻易丢弃，而是通过进化达到发表标准。

同时，从知识图谱多路召回的长尾Pattern能在迭代修正中发挥作用，既保证其与Idea的强相关性，又保证其多样性，不被浪费。

---

## 1. 整体架构图 (Architecture)

```mermaid
graph TD
    User[用户输入 Idea] --> Recall[KG 召回 Top-K Patterns]

    subgraph Phase 1: 策略选择 (Selection)
        Recall -->|分析适配性| Selector[多样性选择器]
        Selector -->|选出 3 个不同策略| P_List[Pattern A, B, C]
    end

    subgraph Phase 2: 并行生成 (Generation)
        P_List -->|Idea + Pattern| Gen[生成 Story Draft]
    end

    subgraph Phase 3: 多智能体评审与修正 (Critic & Refine)
        Gen --> Critic[多角色评审团]
        Critic -->|Pass| Phase4
        Critic -->|Fail: 新颖性不足| Inject_Tail[**策略: 长尾注入**]
        Inject_Tail -->|注入冷门 Trick| Gen
        Critic -->|Fail: 稳定性不足| Inject_Head[**策略: 头部注入**]
        Inject_Head -->|注入稳健 Trick| Gen
    end

    subgraph Phase 4: 查重与规避 (Verification & Pivot)
        Phase4[RAG 查重] -->|Pass| Final[输出 Final Story]
        Phase4 -->|Fail: 撞车| Pivot[**策略: 支点与约束**]
        Pivot -->|添加约束 & 迁移领域| Gen
    end

    Final --> Output([用户最终获得的 Story])
```

---

## 2. 详细流程设计

### Phase 1: 策略选择 (Pattern Selection)

**目标**：避免只生成一种可能性的 Story，确保产出的多样性。

*   **输入**：User Idea, Top-10 Recall Patterns
*   **逻辑**：
    选择 3 个代表性 Pattern：
    1.  **Conservative (稳健型)**: Score 最高，最符合直觉。
    2.  **Innovative (创新型)**: 聚类较小（Cluster Size < 10），容易产生新颖结合。
    3.  **Cross-Domain (跨域型)**: 来自路径 2（领域相关）或路径 3（Paper 相似）。
*   **输出**：Selected Patterns List `[P_Safe, P_Novel, P_Cross]`

### Phase 2: 结构化 Story 生成 (Structured Generation)

**目标**：将抽象的 Idea 和具体的 Pattern 骨架融合。

*   **Story 数据结构**：
    *   `Title`: 论文标题
    *   `Abstract`: 摘要
    *   `Problem_Definition`: 明确的问题定义
    *   `Method_Skeleton`: 核心方法的步骤（基于 Pattern 的 Skeleton 填充）
    *   `Innovation_Claims`: 3 个核心贡献点（Claims）
    *   `Experiments_Plan`: 验证实验的设计

### Phase 3: 多智能体评审与修正 (Critic & Refine)

**目标**：模拟 Peer Review，但不仅仅是打分，更重要的是提供**修改方向**。

*   **角色设定**：
    1.  **Reviewer A (Methodology)**: 关注技术合理性。
    2.  **Reviewer B (Novelty)**: 关注创新性。
    3.  **Reviewer C (Storyteller)**: 关注叙事完整性。

*   **Refinement 策略**：
    如果 Story 未通过（Score < 6），根据拒绝原因触发不同修正路径（详见 2.1 节）。

### Phase 4: 查重与规避 (Verification & Pivot)

**目标**：确保 Story 不与现有论文撞车，如果撞车，通过**微调**来挽救。

*   **检索策略**：
    *   检索源：近 3 年顶会论文。
    *   Query：基于 `Method_Skeleton` 构造组合关键词。

*   **Collision Resolution 策略**：
    如果查重发现撞车（Similarity > 80%），触发 **Pivot & Constraint**：
    1.  **撞车分析**: LLM 分析 Story 与 Collided Paper 的异同。
    2.  **生成约束**: "禁止使用 [撞车点的具体技术细节]。"
    3.  **支点迁移**: 尝试将应用场景迁移到新领域，或增加限制条件（如“无监督设定”）。
    4.  **重生**: 带着约束重写 Story。

---

## 2.1 核心机制：基于属性互补的 Pattern Injection

本方案的核心在于**动态修正**。系统维护一个 Pattern 属性映射表（基于 Cluster Size, Tricks 分布等元数据），根据 Critic 的具体反馈类型，从 Recall 列表中选择**属性互补**的 Pattern 进行注入。

| Critic 反馈类型 | 诊断问题 | 注入策略 (Injection Strategy) | 注入源头 (Source Pattern) | 示例操作 |
| :--- | :--- | :--- | :--- | :--- |
| **Lack of Novelty** | 方法太平庸，增量小 | **Tail Injection (长尾注入)** | Rank 5-10, Cluster Size < 10 (冷门但有特色) | 注入 "课程学习"、"对比学习负采样" 等特定 Trick |
| **Lack of Stability** | 结果可能不稳定，缺乏鲁棒性 | **Head Injection (头部注入)** | Rank 1-3, Cluster Size > 20 (成熟、稳健) | 注入 "多种子验证"、"对抗训练"、"置信度校准" 等稳健性 Trick |
| **Lack of Interpretability** | 黑盒模型，缺乏解释 | **Explanation Injection** | 包含 "Visualization", "Rationale" 的 Pattern | 注入 "Attention可视化"、"Case Study" 模块 |
| **Domain Mismatch** | 方法不适合该领域 | **Domain Adaptation Injection** | 路径 2 (领域相关) 的 Pattern | 注入该领域的特定预处理或特征工程方法 |

**逻辑说明**：
*   **逆向互补**：缺什么补什么。如果缺新意，就找冷门的；如果缺稳定，就找热门成熟的。
*   **强制融合**：Generator 接收到的 Prompt 会强制要求将新 Trick 融合进现有 Method，而不是简单拼接。

---

## 3. 数据流示例 (Refinement 演示)

### Scenario A: Novelty Fail (新颖性不足)

#### Step 1: Initial Generation
*   **Idea**: "用大模型做数据增强"
*   **Pattern**: "Pattern_1: 伪标签训练"
*   **Draft Story**: "使用 LLM 生成伪标签数据，训练小模型。"

#### Step 2: Critic (Reviewer B)
*   **Feedback**: "太普通了，现在满大街都是 LLM 蒸馏。 (Score: 4)"
*   **Action**: 触发 **Tail Injection** (寻找新颖型 Pattern)。
*   **Retrieval**: 发现 Recall 列表中有一个冷门 Pattern "Pattern_12: 课程学习 (Curriculum Learning)"。
*   **Refinement**: 将 "课程学习" 注入 Story。
*   **New Story**: "使用 LLM 生成伪标签，并设计一个**基于难度的课程学习调度器**，让小模型从易到难学习伪标签数据。"

#### Step 3: Critic (Pass)
*   **Feedback**: "引入课程学习调度器后，新颖性提升了。 (Score: 7)" -> **PASS**

#### Step 4: RAG Check (Collision Fail)
*   **Search**: 发现 ACL 2024 论文 "Curriculum Distillation from LLMs"。
*   **Action**: 触发 **Pivot**。
*   **Refinement**: "将场景限定在**法律文书长文本**。法律文本具有逻辑复杂、篇幅长的特点，普通的课程学习失效。"
*   **Final Story**: "Law-Curriculum: 针对法律长文本的**分层级**课程蒸馏框架..."

---

### Scenario B: Stability Fail (稳定性不足)

#### Step 1: Initial Generation
*   **Idea**: "使用强化学习直接优化生成模型的 BLEU 分数"
*   **Pattern**: "Pattern_8: 强化学习微调"
*   **Draft Story**: "定义 BLEU 为 Reward，使用 Policy Gradient 直接优化 Generator。"

#### Step 2: Critic (Reviewer A)
*   **Feedback**: "RL 在文本生成中极不稳定，容易 Mode Collapse，且 BLEU 奖励稀疏，很难训练收敛。 (Score: 5)"
*   **Action**: 触发 **Head Injection** (寻找稳健型 Pattern)。
*   **Retrieval**: 发现 Recall 列表中 Rank 1 的 Pattern "Pattern_3: 对抗训练与鲁棒性优化" (Cluster Size: 30，成熟套路)。
*   **Refinement**: 注入 "对抗训练" 和 "混合目标函数" Trick。
*   **New Story**: "在 RL 优化目标中加入**对抗扰动正则项**，并采用 **MLE + RL 混合训练**策略以稳定冷启动阶段。"

#### Step 3: Critic (Pass)
*   **Feedback**: "混合训练和对抗正则能有效缓解不稳定性，方案可行。 (Score: 8)" -> **PASS**

---

## 4. 方案优势

1.  **提高前系统利用率**：
    传统的 Pipeline 遇到 Critic 不通过就丢弃，浪费了 Token 和计算。本方案通过 **Injection** 和 **Pivot**，将“平庸”的 Story 改造为“优质” Story。

2.  **利用 KG 的长尾价值**：
    Recall 出的 Top-10 Pattern 中，Rank 5-10 的 Pattern 往往因为置信度低被忽略，但它们恰恰是提升新颖性的最佳素材（Spices）。

3.  **更像人类研究员**：
    人类在发现撞车时，不会放弃 Idea，而是会说：“那我们换个数据集做”、“那我们加个限制条件”。本方案复刻了这一思维过程。

