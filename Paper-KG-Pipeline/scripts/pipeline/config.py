import os
from pathlib import Path

# ===================== 路径配置 =====================
# scripts/pipeline/config.py -> scripts/pipeline -> scripts -> Paper-KG-Pipeline
CURRENT_DIR = Path(__file__).parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"

# ===================== LLM API 配置 =====================
LLM_API_KEY = os.getenv("SILICONFLOW_API_KEY", "")
LLM_API_URL = os.getenv("LLM_API_URL", "https://api.siliconflow.cn/v1/chat/completions")
LLM_MODEL = os.getenv("LLM_MODEL", "MiniMaxAI/MiniMax-M2")

# ===================== Pipeline 配置 =====================
class PipelineConfig:
    """Pipeline 配置参数"""
    # Pattern 选择
    SELECT_PATTERN_COUNT = 3  # 选择 3 个不同策略的 Pattern
    CONSERVATIVE_RANK_RANGE = (0, 2)  # 稳健型: Rank 1-3
    INNOVATIVE_CLUSTER_SIZE_THRESHOLD = 10  # 创新型: Cluster Size < 10

    # Critic 阈值
    PASS_SCORE = 7.0  # 评分 >= 7 为通过
    MAX_REFINE_ITERATIONS = 3  # 最多修正 3 轮

    # 新颖性模式配置
    NOVELTY_MODE_MAX_PATTERNS = 10  # 新颖性模式最多尝试的 Pattern 数
    NOVELTY_SCORE_THRESHOLD = 6.0  # 新颖性得分阈值

    # RAG 查重阈值
    COLLISION_THRESHOLD = 0.75  # 相似度 > 0.75 认为撞车

    # Refinement 策略
    TAIL_INJECTION_RANK_RANGE = (4, 9)  # 长尾注入: Rank 5-10
    HEAD_INJECTION_RANK_RANGE = (0, 2)  # 头部注入: Rank 1-3
    HEAD_INJECTION_CLUSTER_THRESHOLD = 15  # 头部注入: Cluster Size > 15

