# Ollama 本地 Embedding 配置指南

## 1. 前置条件

- **Ollama**: 确保您已安装并启动 Ollama 服务。
  - 启动命令: `ollama serve`
- **模型文件**: 本项目根目录下已包含 `Qwen3-Embedding-8B-Q4_K_M.gguf` 模型文件（无需重新下载）。

## 2. 快速配置步骤

### 步骤 1: 准备模型 (国内加速下载)
由于 HuggingFace 或 Ollama 官方源在国内下载速度较慢，我们推荐使用 **ModelScope (魔搭社区)** 进行极速下载。

1.  **安装 ModelScope 库**:
    ```bash
    pip install modelscope
    ```

2.  **下载 GGUF 模型**:
    创建一个下载脚本 `download_model.py` 并运行：
    ```python
    from modelscope.hub.file_download import model_file_download

    # 下载 Qwen3-Embedding-8B 的 GGUF 量化版本
    model_id = 'Qwen/Qwen3-Embedding-8B-GGUF'
    file_name = 'Qwen3-Embedding-8B-Q4_K_M.gguf'
    
    print("🚀 正在从 ModelScope 下载模型...")
    path = model_file_download(model_id, file_name, local_dir='.')
    print(f"✅ 下载完成: {path}")
    ```

### 步骤 2: 创建 Ollama 模型
下载完成后，在项目根目录下运行：

```bash
# 1. 确认已下载文件
ls Qwen3-Embedding-8B-Q4_K_M.gguf

# 2. 使用 Modelfile 创建模型
ollama create qwen3-embedding:8b -f Modelfile
```

**验证安装**:
运行 `ollama list`，如果您看到 `qwen3-embedding:8b`，说明模型已就绪。

### 步骤 3: 确认项目配置

检查项目根目录下的 `i2p_config.json` 文件，确保配置如下：

```json
{
  "llm": {
    "api_url": "https://ark.cn-beijing.volces.com/api/v3/chat/completions",
    "model": "doubao-seed-1-6-251015"
  },
  "embedding": {
    "active_provider": "ollama", 
    "providers": {
      "siliconflow": {
        "api_url": "https://api.siliconflow.cn/v1/embeddings",
        "model": "Qwen/Qwen3-Embedding-8B"
      },
      "ollama": {
        "api_url": "http://localhost:11434/v1/embeddings",
        "model": "qwen3-embedding:8b"
      }
    }
  }
}
```

*   **启用本地模式**: 确保 `"active_provider"` 设置为 `"ollama"`。
*   **切换回云端**: 如果想切回 SiliconFlow API，只需将 `"active_provider"` 改为 `"siliconflow"`。

## 3. 运行 Pipeline

配置完成后，您可以直接运行 Pipeline 或演示脚本，系统会自动调用本地的 Ollama Embedding 服务。

```bash
# 1. 首先进入项目根目录
cd /root/Idea2Paper

# 2. 运行演示脚本
python3 Paper-KG-Pipeline/scripts/demos/demo_pipeline.py
```

## 4. 常见问题

*   **Ollama 服务未启动**: 如果报错 `Connection refused`，请在一个新的终端窗口运行 `ollama serve`。
*   **显存不足**: 8B 模型量化版约需 5GB 显存/内存。如果运行缓慢，请检查系统资源。
