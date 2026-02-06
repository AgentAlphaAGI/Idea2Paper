import hashlib
import os
import re
from pathlib import Path

from .infra.dotenv import load_dotenv
from .infra.user_config import get_config_path, load_user_config

# ===================== 路径配置 =====================
# scripts/pipeline/config.py -> scripts/pipeline -> scripts -> Paper-KG-Pipeline
CURRENT_DIR = Path(__file__).parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
REPO_ROOT = PROJECT_ROOT.parent
OUTPUT_DIR = PROJECT_ROOT / "output"

# 尝试加载 .env（入口脚本也会加载，这里作为兜底）
_DOTENV_STATUS = load_dotenv(REPO_ROOT / ".env", override=False)

# 加载用户配置文件（非敏感参数）
_CONFIG_PATH = get_config_path(REPO_ROOT)
_USER_CONFIG = load_user_config(_CONFIG_PATH)


def _get_from_cfg(cfg: dict, path: list | None):
    if not path:
        return None
    cur = cfg
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _to_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip() == "1"
    return bool(value)


def _cast(value, cast):
    if cast is None:
        return value
    if cast is bool:
        return _to_bool(value)
    if cast is int:
        return int(value)
    if cast is float:
        return float(value)
    if cast is str:
        return str(value)
    if cast is Path:
        return Path(value)
    return cast(value)


def _cast_list_float(value):
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",") if p.strip()]
        return [float(p) for p in parts]
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    return value


def _get(key: str, default, cast=None, cfg_path: list | None = None):
    env_val = os.getenv(key)
    if env_val is not None:
        value = env_val
    else:
        cfg_val = _get_from_cfg(_USER_CONFIG, cfg_path)
        value = cfg_val if cfg_val is not None else default
    return _cast(value, cast) if cast else value

# ===================== LLM API 配置 =====================
LLM_API_KEY = _get(
    "LLM_API_KEY",
    "",  # 移除对旧环境变量 SILICONFLOW_API_KEY 的默认依赖，强制使用新标准
    cast=str,
    cfg_path=["llm", "api_key"],
)

LLM_API_URL = _get(
    "LLM_API_URL",
    "https://api.siliconflow.cn/v1/chat/completions",
    cast=str,
    cfg_path=["llm", "api_url"],
)
LLM_MODEL = _get(
    "LLM_MODEL",
    "Pro/zai-org/GLM-4.7",
    cast=str,
    cfg_path=["llm", "model"],
)

# ===================== Embedding API 配置 =====================
# 1. Determine active provider
EMBEDDING_PROVIDER = _get(
    "EMBEDDING_PROVIDER",
    None,
    cast=str,
    cfg_path=["embedding", "active_provider"],
)
# Fallback to legacy "provider" key if active_provider not set
if not EMBEDDING_PROVIDER:
    EMBEDDING_PROVIDER = _get(
        "EMBEDDING_PROVIDER",
        "siliconflow",
        cast=str,
        cfg_path=["embedding", "provider"],
    )

# 2. Resolve defaults based on provider
_prov_cfg = _get_from_cfg(_USER_CONFIG, ["embedding", "providers", EMBEDDING_PROVIDER])
if _prov_cfg:
    _default_url = _prov_cfg.get("api_url")
    _default_model = _prov_cfg.get("model")
else:
    # Hardcoded defaults for known providers if config is missing details
    if EMBEDDING_PROVIDER == "ollama":
        _default_url = "http://localhost:11434/v1/embeddings"
        _default_model = "qwen3-embedding:8b"
    else:
        # Default for siliconflow or others
        _default_url = "https://api.siliconflow.cn/v1/embeddings"
        _default_model = "Qwen/Qwen3-Embedding-8B"

EMBEDDING_API_URL = _get(
    "EMBEDDING_API_URL",
    _default_url,
    cast=str,
    # Legacy flat config can still override if present
    cfg_path=["embedding", "api_url"],
)
EMBEDDING_MODEL = _get(
    "EMBEDDING_MODEL",
    _default_model,
    cast=str,
    cfg_path=["embedding", "model"],
)
# Secret: only from env/.env; fallback to SILICONFLOW_API_KEY (LLM_API_KEY)
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "") or LLM_API_KEY

# ===================== Run Logging 配置 =====================
LOG_ROOT = _get(
    "I2P_LOG_DIR",
    str(REPO_ROOT / "log"),
    cast=Path,
    cfg_path=["logging", "dir"],
)
ENABLE_RUN_LOGGING = _get(
    "I2P_ENABLE_LOGGING",
    True,
    cast=bool,
    cfg_path=["logging", "enable"],
)
LOG_MAX_TEXT_CHARS = _get(
    "I2P_LOG_MAX_TEXT_CHARS",
    20000,
    cast=int,
    cfg_path=["logging", "max_text_chars"],
)

# ===================== Results Bundling 配置 =====================
RESULTS_ROOT = _get(
    "I2P_RESULTS_DIR",
    str(REPO_ROOT / "results"),
    cast=Path,
    cfg_path=["results", "dir"],
)
RESULTS_ENABLE = _get(
    "I2P_RESULTS_ENABLE",
    True,
    cast=bool,
    cfg_path=["results", "enable"],
)
RESULTS_MODE = _get(
    "I2P_RESULTS_MODE",
    "link",
    cast=str,
    cfg_path=["results", "mode"],
)
RESULTS_KEEP_LOG = _get(
    "I2P_RESULTS_KEEP_LOG",
    True,
    cast=bool,
    cfg_path=["results", "keep_log"],
)

# ===================== Index Dir Mode 配置 =====================
INDEX_DIR_MODE = _get(
    "I2P_INDEX_DIR_MODE",
    "manual",
    cast=str,
    cfg_path=["index", "dir_mode"],
)

_PROFILE_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _sanitize_profile_component(value: str) -> str:
    if value is None:
        return ""
    text = str(value)
    text = text.replace("/", "_").replace(" ", "_")
    return _PROFILE_SAFE_RE.sub("_", text)


def _compute_profile_id(provider: str, model: str, api_url: str) -> str:
    provider_s = _sanitize_profile_component(provider)
    model_s = _sanitize_profile_component(model)
    url_hash = hashlib.sha256(str(api_url).encode("utf-8")).hexdigest()[:8]
    return f"{provider_s}__{model_s}__{url_hash}"


if INDEX_DIR_MODE == "auto_profile":
    _PROFILE_ID = _compute_profile_id(EMBEDDING_PROVIDER, EMBEDDING_MODEL, EMBEDDING_API_URL)
    _DEFAULT_NOVELTY_INDEX_DIR = str(OUTPUT_DIR / f"novelty_index__{_PROFILE_ID}")
    _DEFAULT_RECALL_INDEX_DIR = str(OUTPUT_DIR / f"recall_index__{_PROFILE_ID}")
else:
    _PROFILE_ID = None
    _DEFAULT_NOVELTY_INDEX_DIR = str(OUTPUT_DIR / "novelty_index")
    _DEFAULT_RECALL_INDEX_DIR = str(OUTPUT_DIR / "recall_index")

# ===================== Novelty Check 配置 =====================
NOVELTY_ENABLE = _get(
    "I2P_NOVELTY_ENABLE",
    True,
    cast=bool,
    cfg_path=["novelty", "enable"],
)
NOVELTY_TOPK = _get(
    "I2P_NOVELTY_TOPK",
    100,
    cast=int,
    cfg_path=["novelty", "top_k"],
)
NOVELTY_HIGH_TH = _get(
    "I2P_NOVELTY_HIGH_TH",
    0.88,
    cast=float,
    cfg_path=["novelty", "high_th"],
)
NOVELTY_MEDIUM_TH = _get(
    "I2P_NOVELTY_MEDIUM_TH",
    0.82,
    cast=float,
    cfg_path=["novelty", "medium_th"],
)
NOVELTY_INDEX_DIR = _get(
    "I2P_NOVELTY_INDEX_DIR",
    _DEFAULT_NOVELTY_INDEX_DIR,
    cast=Path,
    cfg_path=["novelty", "index_dir"],
)
NOVELTY_AUTO_BUILD_INDEX = _get(
    "I2P_NOVELTY_AUTO_BUILD_INDEX",
    False,
    cast=bool,
    cfg_path=["novelty", "auto_build_index"],
)
NOVELTY_INDEX_BUILD_BATCH_SIZE = _get(
    "I2P_NOVELTY_INDEX_BUILD_BATCH_SIZE",
    32,
    cast=int,
    cfg_path=["novelty", "index_batch_size"],
)
NOVELTY_INDEX_BUILD_RESUME = _get(
    "I2P_NOVELTY_INDEX_BUILD_RESUME",
    True,
    cast=bool,
    cfg_path=["novelty", "index_resume"],
)
NOVELTY_INDEX_BUILD_MAX_RETRIES = _get(
    "I2P_NOVELTY_INDEX_BUILD_MAX_RETRIES",
    3,
    cast=int,
    cfg_path=["novelty", "index_max_retries"],
)
NOVELTY_INDEX_BUILD_SLEEP_SEC = _get(
    "I2P_NOVELTY_INDEX_BUILD_SLEEP_SEC",
    1.0,
    cast=float,
    cfg_path=["novelty", "index_sleep_sec"],
)
NOVELTY_ACTION = _get(
    "I2P_NOVELTY_ACTION",
    "pivot",
    cast=str,
    cfg_path=["novelty", "action"],
)
NOVELTY_MAX_PIVOTS = _get(
    "I2P_NOVELTY_MAX_PIVOTS",
    2,
    cast=int,
    cfg_path=["novelty", "max_pivots"],
)
NOVELTY_REQUIRE_EMBEDDING = _get(
    "I2P_NOVELTY_REQUIRE_EMBEDDING",
    False,
    cast=bool,
    cfg_path=["novelty", "require_embedding"],
)
NOVELTY_REPORT_IN_OUTPUT = _get(
    "I2P_NOVELTY_REPORT_IN_OUTPUT",
    False,
    cast=bool,
    cfg_path=["novelty", "report_in_output"],
)

# ===================== Pipeline 配置 =====================
class PipelineConfig:
    """Pipeline 配置参数"""
    # Pattern 选择
    SELECT_PATTERN_COUNT = 3  # 选择 3 个不同策略的 Pattern
    CONSERVATIVE_RANK_RANGE = (0, 2)  # 稳健型: Rank 1-3
    INNOVATIVE_CLUSTER_SIZE_THRESHOLD = 10  # 创新型: Cluster Size < 10

    # Critic 阈值
    PASS_SCORE = _get(
        "I2P_PASS_SCORE",
        7.0,
        cast=float,
        cfg_path=["pass", "fixed_score"],
    )  # 评分 >= 7 为通过
    MAX_REFINE_ITERATIONS = 3  # 最多修正 3 轮

    # Pass mode (pattern-aware)
    PASS_MODE = _get(
        "I2P_PASS_MODE",
        "two_of_three_q75_and_avg_ge_q50",
        cast=str,
        cfg_path=["pass", "mode"],
    )
    PASS_MIN_PATTERN_PAPERS = _get(
        "I2P_PASS_MIN_PATTERN_PAPERS",
        20,
        cast=int,
        cfg_path=["pass", "min_pattern_papers"],
    )
    PASS_FALLBACK = _get(
        "I2P_PASS_FALLBACK",
        "global",
        cast=str,
        cfg_path=["pass", "fallback"],
    )  # global|fixed

    # 新颖性模式配置
    NOVELTY_MODE_MAX_PATTERNS = 3  # 新颖性模式最多尝试的 Pattern 数
    NOVELTY_SCORE_THRESHOLD = 6.0  # 新颖性得分阈值

    # 召回审计配置（召回候选与分数落盘）
    RECALL_AUDIT_ENABLE = _get(
        "I2P_RECALL_AUDIT_ENABLE",
        True,
        cast=bool,
        cfg_path=["recall", "audit_enable"],
    )
    RECALL_AUDIT_TOPN = _get(
        "I2P_RECALL_AUDIT_TOPN",
        50,
        cast=int,
        cfg_path=["recall", "audit_topn"],
    )
    RECALL_AUDIT_SNIPPET_CHARS = _get(
        "I2P_RECALL_AUDIT_SNIPPET_CHARS",
        240,
        cast=int,
        cfg_path=["recall", "audit_snippet_chars"],
    )
    RECALL_AUDIT_IN_EVENTS = _get(
        "I2P_RECALL_AUDIT_IN_EVENTS",
        True,
        cast=bool,
        cfg_path=["recall", "audit_in_events"],
    )
    RECALL_EMBED_BATCH_SIZE = _get(
        "I2P_RECALL_EMBED_BATCH_SIZE",
        32,
        cast=int,
        cfg_path=["recall", "embed_batch_size"],
    )
    RECALL_EMBED_MAX_RETRIES = _get(
        "I2P_RECALL_EMBED_MAX_RETRIES",
        3,
        cast=int,
        cfg_path=["recall", "embed_max_retries"],
    )
    RECALL_EMBED_SLEEP_SEC = _get(
        "I2P_RECALL_EMBED_SLEEP_SEC",
        0.5,
        cast=float,
        cfg_path=["recall", "embed_sleep_sec"],
    )
    RECALL_USE_OFFLINE_INDEX = _get(
        "I2P_RECALL_USE_OFFLINE_INDEX",
        False,
        cast=bool,
        cfg_path=["recall", "use_offline_index"],
    )
    RECALL_INDEX_DIR = _get(
        "I2P_RECALL_INDEX_DIR",
        _DEFAULT_RECALL_INDEX_DIR,
        cast=Path,
        cfg_path=["recall", "index_dir"],
    )

    # Index preflight (auto-prepare before run)
    INDEX_AUTO_PREPARE = _get(
        "I2P_INDEX_AUTO_PREPARE",
        True,
        cast=bool,
        cfg_path=["index", "auto_prepare"],
    )
    INDEX_ALLOW_BUILD = _get(
        "I2P_INDEX_ALLOW_BUILD",
        True,
        cast=bool,
        cfg_path=["index", "allow_build"],
    )

    # Phase 4 查重开关
    VERIFICATION_ENABLE = _get(
        "I2P_VERIFICATION_ENABLE",
        True,
        cast=bool,
        cfg_path=["verification", "enable"],
    )

    # RAG 查重阈值
    COLLISION_THRESHOLD = _get(
        "I2P_COLLISION_THRESHOLD",
        0.75,
        cast=float,
        cfg_path=["verification", "collision_threshold"],
    )  # 相似度 > 阈值 认为撞车

    # Refinement 策略
    TAIL_INJECTION_RANK_RANGE = (4, 9)  # 长尾注入: Rank 5-10
    HEAD_INJECTION_RANK_RANGE = (0, 2)  # 头部注入: Rank 1-3
    HEAD_INJECTION_CLUSTER_THRESHOLD = 15  # 头部注入: Cluster Size > 15

    # Anchored Critic 配置
    ANCHOR_QUANTILES = _get(
        "I2P_ANCHOR_QUANTILES",
        [0.1, 0.25, 0.5, 0.75, 0.9],
        cast=_cast_list_float,
        cfg_path=["anchors", "quantiles"],
    )
    ANCHOR_MAX_INITIAL = _get(
        "I2P_ANCHOR_MAX_INITIAL",
        7,
        cast=int,
        cfg_path=["anchors", "max_initial"],
    )
    ANCHOR_MAX_TOTAL = _get(
        "I2P_ANCHOR_MAX_TOTAL",
        9,
        cast=int,
        cfg_path=["anchors", "max_total"],
    )
    ANCHOR_MAX_EXEMPLARS = _get(
        "I2P_ANCHOR_MAX_EXEMPLARS",
        2,
        cast=int,
        cfg_path=["anchors", "max_exemplars"],
    )
    DENSIFY_OFFSETS = _get(
        "I2P_DENSIFY_OFFSETS",
        [-0.5, 0.5, -0.25, 0.25],
        cast=_cast_list_float,
        cfg_path=["anchors", "densify_offsets"],
    )
    SIGMOID_K = _get(
        "I2P_SIGMOID_K",
        1.2,
        cast=float,
        cfg_path=["anchors", "sigmoid_k"],
    )
    GRID_STEP = _get(
        "I2P_GRID_STEP",
        0.01,
        cast=float,
        cfg_path=["anchors", "grid_step"],
    )
    DENSIFY_LOSS_THRESHOLD = _get(
        "I2P_DENSIFY_LOSS_THRESHOLD",
        0.03,
        cast=float,
        cfg_path=["anchors", "densify_loss_threshold"],
    )
    DENSIFY_MIN_AVG_CONF = _get(
        "I2P_DENSIFY_MIN_AVG_CONF",
        0.45,
        cast=float,
        cfg_path=["anchors", "densify_min_avg_conf"],
    )
    ANCHOR_DENSIFY_ENABLE = _get(
        "I2P_ANCHOR_DENSIFY_ENABLE",
        True,
        cast=bool,
        cfg_path=["anchors", "densify_enable"],
    )

    # Critic JSON reliability (quality-first)
    CRITIC_STRICT_JSON = _get(
        "I2P_CRITIC_STRICT_JSON",
        True,
        cast=bool,
        cfg_path=["critic", "strict_json"],
    )
    CRITIC_JSON_RETRIES = _get(
        "I2P_CRITIC_JSON_RETRIES",
        2,
        cast=int,
        cfg_path=["critic", "json_retries"],
    )
