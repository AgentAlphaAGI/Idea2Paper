# Idea2Paper 本地化更新日志

## 1. 核心目标
本次更新旨在将 Idea2Paper 项目从依赖云端 API 的模式，扩展为**支持本地私有化部署**的版本，重点解决了 Embedding 模型的本地化运行以及配置的灵活性。

## 2. 主要更新内容

### 2.1 Embedding 系统重构 (本地化 + 双模式)
*   **多 Provider 支持**：
    *   重构了 `i2p_config.json` 和配置读取逻辑，现在支持通过 `"active_provider"` 字段一键切换 `siliconflow` (云端) 和 `ollama` (本地)。
    *   代码位置：`src/idea2paper/config.py`
*   **Ollama 本地集成**：
    *   新增了对 Ollama 接口的完整支持，允许使用本地显卡运行 `qwen3-embedding:8b` 等模型。
    *   **内置本地模型支持**：项目预置了 `Qwen3-Embedding-8B` 的 GGUF 文件，无需联网下载即可快速部署。
    *   代码位置：`src/idea2paper/infra/embeddings.py`

### 2.2 本地缓存层 (KV Cache)
*   **机制原理**：
    *   在 `infra` 层引入了基于 Pickle 的持久化 KV 缓存。
    *   **缓存键 (Key)**：`MD5(模型名 + 文本内容)`，确保模型切换或文本变更时自动失效。
    *   **缓存值 (Value)**：Embedding 向量列表。
*   **增量更新**：
    *   实现了批量处理逻辑：当请求一批文本时，系统会自动过滤出未命中的“新数据”进行计算，并将新结果与缓存中的旧结果合并返回。
    *   **作用**：减少 API 调用或本地推理开销。
*   **代码位置**：`src/idea2paper/infra/embeddings.py`

### 2.3 系统稳定性改进
*   **自动索引重建**：
    *   修复了 `NoveltyIndex` (查重模块) 在缺失离线索引文件时直接降级为关键词匹配的问题。
    *   新增逻辑：当发现索引缺失时，自动调用 Embedding 接口（配合 KV 缓存）现场重建索引，确保查重精度。
    *   代码位置：`src/idea2paper/application/novelty/novelty_index.py`
*   **日志完善**：
    *   为 LLM 和 Embedding 调用增加了详细的控制台日志，显示实时耗时和状态。

## 3. 对比原版

| 特性 | 原版 (Original) | 本地化版本 (Current) |
| :--- | :--- | :--- |
| **Embedding** | 仅支持 SiliconFlow 云端 API | **支持 Ollama 本地** / SiliconFlow 云端 (可切换) |
| **缓存机制** | 仅依赖离线构建的静态 `.npy` 索引 | **新增 KV 动态缓存** + 静态索引 (双层机制) |
| **数据处理** | 新数据需重新跑全量索引构建脚本 | **支持增量计算**，新数据即时写入缓存 |
| **查重精度** | 索引缺失时降级为关键词匹配 | **自动重建索引**，保持向量级精度 |
| **网络依赖** | 强依赖外网 | **可完全离线运行** (LLM 和 Embedding 均可本地化) |

## 4. 依赖更新与安装指引

### 4.1 推荐硬件配置
为了获得最佳的本地化运行体验（特别是 Embedding 模型的推理速度），我们推荐以下硬件配置：

| 组件 | 最低配置 (Minimum) | 推荐配置 (Recommended) |
| :--- | :--- | :--- |
| **显卡 (GPU)** | NVIDIA GeForce RTX 3060 (12GB) 或同级显卡 | NVIDIA GeForce RTX 4080 / 4090 (24GB) |
| **显存 (VRAM)** | ≥ 8GB | ≥ 16GB (支持更大 batch size 加速) |
| **内存 (RAM)** | 16GB | 32GB+ |
| **硬盘** | SSD (至少预留 10GB 空间) | NVMe SSD (提升缓存读写速度) |

> **注意**: 如果没有独立显卡，Ollama 也可以在 CPU 上运行，但处理速度会较慢，建议仅用于测试或小规模数据场景。

### 4.2 新增依赖
本次更新引入了以下 Python 库：
*   `modelscope`: 用于从魔搭社区加速下载 GGUF 模型文件。

### 4.2 安装步骤

1.  **更新依赖**:
    ```bash
    pip install modelscope
    ```
    或者如果您有 `requirements.txt`，请确保包含：
    ```text
    modelscope>=1.10.0
    ```

2.  **启动 Ollama**:
    ```bash
    # 安装 Ollama (如果尚未安装)
    curl -fsSL https://ollama.com/install.sh | sh
    
    # 启动服务
    ollama serve
    ```

3.  **加载本地模型**:
    项目根目录已包含模型文件，直接运行以下命令即可加载：
    ```bash
    ollama create qwen3-embedding:8b -f Modelfile
    ```

4.  **配置切换**:
    修改 `i2p_config.json`，将 `active_provider` 设置为 `"ollama"`。

5.  **运行 Pipeline**:
    ```bash
    python3 Paper-KG-Pipeline/scripts/idea2story_pipeline.py "Your Idea"
    ```
