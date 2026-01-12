# Idea2Story Pipeline API 参考文档

本文档详细说明了重构后的 Idea2Story Pipeline 的代码结构、模块功能及使用注意事项。

## 📂 代码结构

重构后的代码位于 `scripts/pipeline/` 目录下，采用模块化设计：

```
scripts/
├── idea2story_pipeline.py  # 入口脚本
└── pipeline/               # 核心包
    ├── __init__.py         # 导出主要类
    ├── config.py           # 配置与常量
    ├── utils.py            # 通用工具 (LLM调用, JSON解析)
    ├── manager.py          # 主流程编排器 (Idea2StoryPipeline)
    ├── pattern_selector.py # Phase 1: Pattern 选择
    ├── story_generator.py  # Phase 2: Story 生成
    ├── critic.py           # Phase 3: 多智能体评审
    ├── refinement.py       # Phase 3.5: 修正引擎
    └── verifier.py         # Phase 4: RAG 查重
```

---

## 🔧 核心模块详解

### 1. 配置模块 (`config.py`)

包含所有全局配置和常量。

- **`PipelineConfig` 类**:
  - `SELECT_PATTERN_COUNT`: 选择 Pattern 的数量 (默认 3)
  - `PASS_SCORE`: 评审通过分数阈值 (默认 7.0)
  - `MAX_REFINE_ITERATIONS`: 最大修正轮次 (默认 3)
  - `COLLISION_THRESHOLD`: 查重相似度阈值 (默认 0.75)
  - `TAIL_INJECTION_RANK_RANGE`: 长尾注入的 Rank 范围
  - `HEAD_INJECTION_RANK_RANGE`: 头部注入的 Rank 范围

- **环境变量**:
  - `LLM_API_KEY`: SiliconFlow API Key
  - `LLM_MODEL`: 使用的模型名称

### 2. 工具模块 (`utils.py`)

提供底层支持功能。

- **`call_llm(prompt, temperature, max_tokens)`**:
  - 封装了对 SiliconFlow API 的调用。
  - 包含错误处理和模拟输出（当未配置 API Key 时）。

- **`parse_json_from_llm(response)`**:
  - **功能**: 从 LLM 的文本响应中提取并解析 JSON。
  - **特性**:
    - 自动去除 Markdown 代码块标记。
    - 处理非法控制字符（如未转义的换行符）。
    - 包含正则修复逻辑（修复尾部逗号、缺失逗号等常见 JSON 错误）。
    - 是保证 Pipeline 稳定性的关键函数。

### 3. 主编排器 (`manager.py`)

- **`Idea2StoryPipeline` 类**:
  - **功能**: 协调各个子模块，执行完整的 Pipeline 流程。
  - **主要方法**: `run()`
  - **逻辑**:
    1. 调用 `PatternSelector` 选择 Pattern。
    2. 调用 `StoryGenerator` 生成初始 Story。
    3. 进入修正循环：
       - 调用 `MultiAgentCritic` 评审。
       - 若未通过，调用 `RefinementEngine` 获取注入策略。
       - 调用 `StoryGenerator` 进行增量修正。
    4. 评审通过或达到最大轮次后，调用 `RAGVerifier` 进行查重。
    5. 若查重失败，触发 Pivot 策略并重新生成。

### 4. Pattern 选择器 (`pattern_selector.py`)

- **`PatternSelector` 类**:
  - **功能**: 从召回的 Pattern 列表中选择最具代表性的 3 个 Pattern。
  - **策略**:
    - `conservative`: 选择分数最高的 Pattern（稳健）。
    - `innovative`: 选择 Cluster Size 较小的 Pattern（新颖）。
    - `cross_domain`: 选择跨域或次优 Pattern。

### 5. Story 生成器 (`story_generator.py`)

- **`StoryGenerator` 类**:
  - **功能**: 基于 Idea 和 Pattern 生成结构化的 Story JSON。
  - **特性**:
    - **初次生成**: 基于 Pattern 模板生成。
    - **增量修正**: 接受 `previous_story` 和 `review_feedback`，仅修改有问题部分，保留精华。
    - **Prompt 工程**: 包含强约束指令，防止 LLM 输出非 JSON 内容。

### 6. 多智能体评审 (`critic.py`)

- **`MultiAgentCritic` 类**:
  - **功能**: 模拟三个不同角色的评审员对 Story 进行打分。
  - **角色**:
    - `Methodology`: 关注技术合理性。
    - `Novelty`: 关注创新性（包含严格的打分指令）。
    - `Storyteller`: 关注叙事完整性。
  - **输出**: 包含总分、各角色反馈及主要问题诊断。

### 7. 修正引擎 (`refinement.py`)

- **`RefinementEngine` 类**:
  - **功能**: 根据评审反馈，从知识库中检索合适的 Trick 或方法论进行注入。
  - **策略**:
    - `Tail Injection`: 针对 Novelty 问题，注入冷门 Pattern 的核心方法论。
    - `Head Injection`: 针对 Stability 问题，注入成熟 Pattern 的稳定性设计。
    - `Explanation Injection`: 针对 Interpretability 问题，注入可视化/分析手段。

### 8. RAG 查重器 (`verifier.py`)

- **`RAGVerifier` 类**:
  - **功能**: 计算生成的 Story 与现有论文库的相似度，防止撞车。
  - **方法**: 基于 Jaccard 相似度计算 Method Skeleton 的重合度。
  - **Pivot**: 若检测到撞车，生成约束条件（如“禁止使用X技术”、“迁移到Y领域”）。

---

## 🚀 使用指南

### 运行 Pipeline

直接运行入口脚本即可：

```bash
python scripts/idea2story_pipeline.py "你的研究想法"
```

### 扩展开发

如果需要添加新的评审角色或修改生成逻辑：

1. **修改评审角色**: 编辑 `scripts/pipeline/critic.py` 中的 `self.reviewers` 列表。
2. **修改 Prompt**: 编辑 `scripts/pipeline/story_generator.py` 中的 `_build_generation_prompt` 方法。
3. **添加新工具**: 在 `scripts/pipeline/utils.py` 中添加函数，并在其他模块中导入。

### 调试建议

- 所有模块都包含详细的 `print` 输出，可以在控制台查看执行进度。
- 如果 JSON 解析频繁失败，请检查 `utils.py` 中的 `parse_json_from_llm` 函数，或调整 `config.py` 中的 LLM 模型参数。

