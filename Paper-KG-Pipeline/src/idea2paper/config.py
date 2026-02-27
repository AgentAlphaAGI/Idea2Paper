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


def _get_with_legacy(
    key: str,
    legacy_key: str,
    default,
    cast=None,
    cfg_path: list | None = None,
    legacy_cfg_path: list | None = None,
):
    env_val = os.getenv(key)
    if env_val is None and legacy_key:
        env_val = os.getenv(legacy_key)

    if env_val is not None:
        value = env_val
    else:
        cfg_val = _get_from_cfg(_USER_CONFIG, cfg_path)
        if cfg_val is None and legacy_cfg_path:
            cfg_val = _get_from_cfg(_USER_CONFIG, legacy_cfg_path)
        value = cfg_val if cfg_val is not None else default

    return _cast(value, cast) if cast else value

# ===================== LLM API 配置 =====================
# Secret: only from env/.env (do not put in i2p_config.json)
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_PROVIDER = _get(
    "LLM_PROVIDER",
    "openai_compatible_chat",
    cast=str,
    cfg_path=["llm", "provider"],
)
LLM_BASE_URL = _get(
    "LLM_BASE_URL",
    "",
    cast=str,
    cfg_path=["llm", "base_url"],
)
LLM_API_URL = _get(
    "LLM_API_URL",
    "",
    cast=str,
    cfg_path=["llm", "api_url"],
)
LLM_MODEL = _get(
    "LLM_MODEL",
    "gpt-4o-mini",
    cast=str,
    cfg_path=["llm", "model"],
)
LLM_ANTHROPIC_VERSION = _get(
    "LLM_ANTHROPIC_VERSION",
    "2023-06-01",
    cast=str,
    cfg_path=["llm", "anthropic_version"],
)
LLM_EXTRA_HEADERS = _get(
    "LLM_EXTRA_HEADERS_JSON",
    None,
    cfg_path=["llm", "extra_headers"],
)
LLM_EXTRA_BODY = _get(
    "LLM_EXTRA_BODY_JSON",
    None,
    cfg_path=["llm", "extra_body"],
)

# ===================== Embedding API 配置 =====================
# Embedding 可独立配置；默认使用 OpenAI-compatible /v1/embeddings 形态。
EMBEDDING_PROVIDER = _get(
    "EMBEDDING_PROVIDER",
    "openai_compatible",
    cast=str,
    cfg_path=["embedding", "provider"],
)
EMBEDDING_API_URL = _get(
    "EMBEDDING_API_URL",
    "https://api.openai.com/v1/embeddings",
    cast=str,
    cfg_path=["embedding", "api_url"],
)
EMBEDDING_MODEL = _get(
    "EMBEDDING_MODEL",
    "text-embedding-3-large",
    cast=str,
    cfg_path=["embedding", "model"],
)
# Secret: only from env/.env; fallback to LLM_API_KEY
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
# Hard-coded: always copy results into `results/run_.../` (no symlinks).
# This avoids platform-specific symlink issues and makes results fully portable.
RESULTS_MODE = "copy"
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


def _compute_profile_id(model: str) -> str:
    model_s = _sanitize_profile_component(model)
    return model_s or "unknown_model"


if INDEX_DIR_MODE == "auto_profile":
    _PROFILE_ID = _compute_profile_id(EMBEDDING_MODEL)
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

    # Per-role 最低分硬约束：即使整体 avg 通过，单个 role 低于此值也视为 fail
    # 设为 0.0 表示不启用（默认不限制）；设为 5.0 则 Novelty < 5 强制不通过
    PASS_MIN_NOVELTY_SCORE = _get(
        "I2P_PASS_MIN_NOVELTY_SCORE",
        5.0,
        cast=float,
        cfg_path=["pass", "min_novelty_score"],
    )  # Novelty 单项最低分；0.0 = 不限制
    PASS_MIN_METHODOLOGY_SCORE = _get(
        "I2P_PASS_MIN_METHODOLOGY_SCORE",
        0.0,
        cast=float,
        cfg_path=["pass", "min_methodology_score"],
    )  # Methodology 单项最低分；0.0 = 不限制

    # LLM Temperature (per stage; defaults preserve current behavior)
    LLM_TEMPERATURE_DEFAULT = _get(
        "I2P_LLM_TEMPERATURE_DEFAULT",
        0.7,
        cast=float,
        cfg_path=["llm", "temperature", "default"],
    )
    LLM_TEMPERATURE_STORY_GENERATOR = _get(
        "I2P_LLM_TEMPERATURE_STORY_GENERATOR",
        0.7,
        cast=float,
        cfg_path=["llm", "temperature", "story_generator"],
    )
    LLM_TEMPERATURE_STORY_GENERATOR_REWRITE = _get(
        "I2P_LLM_TEMPERATURE_STORY_GENERATOR_REWRITE",
        0.3,
        cast=float,
        cfg_path=["llm", "temperature", "story_generator_rewrite"],
    )
    LLM_TEMPERATURE_STORY_REFLECTOR = _get(
        "I2P_LLM_TEMPERATURE_STORY_REFLECTOR",
        0.5,
        cast=float,
        cfg_path=["llm", "temperature", "story_reflector"],
    )
    LLM_TEMPERATURE_PATTERN_SELECTOR = _get(
        "I2P_LLM_TEMPERATURE_PATTERN_SELECTOR",
        0.3,
        cast=float,
        cfg_path=["llm", "temperature", "pattern_selector"],
    )
    LLM_TEMPERATURE_IDEA_FUSION = _get(
        "I2P_LLM_TEMPERATURE_IDEA_FUSION",
        0.7,
        cast=float,
        cfg_path=["llm", "temperature", "idea_fusion"],
    )
    LLM_TEMPERATURE_IDEA_FUSION_STAGE2 = _get(
        "I2P_LLM_TEMPERATURE_IDEA_FUSION_STAGE2",
        0.8,
        cast=float,
        cfg_path=["llm", "temperature", "idea_fusion_stage2"],
    )
    LLM_TEMPERATURE_IDEA_FUSION_STAGE3 = _get(
        "I2P_LLM_TEMPERATURE_IDEA_FUSION_STAGE3",
        0.9,
        cast=float,
        cfg_path=["llm", "temperature", "idea_fusion_stage3"],
    )
    LLM_TEMPERATURE_CRITIC_MAIN = _get(
        "I2P_LLM_TEMPERATURE_CRITIC_MAIN",
        0.0,
        cast=float,
        cfg_path=["llm", "temperature", "critic_main"],
    )
    LLM_TEMPERATURE_CRITIC_REPAIR = _get(
        "I2P_LLM_TEMPERATURE_CRITIC_REPAIR",
        0.0,
        cast=float,
        cfg_path=["llm", "temperature", "critic_repair"],
    )
    LLM_TEMPERATURE_CRITIC_ANCHORED = _get(
        "I2P_LLM_TEMPERATURE_CRITIC_ANCHORED",
        0.3,
        cast=float,
        cfg_path=["llm", "temperature", "critic_anchored"],
    )

    # Idea Packaging (optional; defaults preserve current behavior)
    IDEA_PACKAGING_ENABLE = _get(
        "I2P_IDEA_PACKAGING_ENABLE",
        False,
        cast=bool,
        cfg_path=["idea", "packaging_enable"],
    )
    IDEA_PACKAGING_TOPN_PATTERNS = _get(
        "I2P_IDEA_PACKAGING_TOPN_PATTERNS",
        5,
        cast=int,
        cfg_path=["idea", "packaging_topn_patterns"],
    )
    IDEA_PACKAGING_MAX_EXEMPLAR_PAPERS = _get(
        "I2P_IDEA_PACKAGING_MAX_EXEMPLAR_PAPERS",
        8,
        cast=int,
        cfg_path=["idea", "packaging_max_exemplar_papers"],
    )
    IDEA_PACKAGING_CANDIDATE_K = _get(
        "I2P_IDEA_PACKAGING_CANDIDATE_K",
        3,
        cast=int,
        cfg_path=["idea", "packaging_candidate_k"],
    )
    IDEA_PACKAGING_SELECT_MODE = _get(
        "I2P_IDEA_PACKAGING_SELECT_MODE",
        "llm_then_recall",
        cast=str,
        cfg_path=["idea", "packaging_select_mode"],
    )
    IDEA_PACKAGING_FORCE_EN_QUERY = _get(
        "I2P_IDEA_PACKAGING_FORCE_EN_QUERY",
        True,
        cast=bool,
        cfg_path=["idea", "packaging_force_en_query"],
    )

    # Idea Packaging LLM temperatures
    LLM_TEMPERATURE_IDEA_PACKAGING_PARSE = _get(
        "I2P_LLM_TEMPERATURE_IDEA_PACKAGING_PARSE",
        0.0,
        cast=float,
        cfg_path=["llm", "temperature", "idea_packaging_parse"],
    )
    LLM_TEMPERATURE_IDEA_PACKAGING_PATTERN_GUIDED = _get(
        "I2P_LLM_TEMPERATURE_IDEA_PACKAGING_PATTERN_GUIDED",
        0.3,
        cast=float,
        cfg_path=["llm", "temperature", "idea_packaging_pattern_guided"],
    )
    LLM_TEMPERATURE_IDEA_PACKAGING_JUDGE = _get(
        "I2P_LLM_TEMPERATURE_IDEA_PACKAGING_JUDGE",
        0.0,
        cast=float,
        cfg_path=["llm", "temperature", "idea_packaging_judge"],
    )

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
    SUBDOMAIN_TAXONOMY_ENABLE = _get(
        "I2P_SUBDOMAIN_TAXONOMY_ENABLE",
        False,
        cast=bool,
        cfg_path=["recall", "subdomain_taxonomy_enable"],
    )
    SUBDOMAIN_TAXONOMY_PATH = _get(
        "I2P_SUBDOMAIN_TAXONOMY_PATH",
        "",
        cast=str,
        cfg_path=["recall", "subdomain_taxonomy_path"],
    )
    SUBDOMAIN_TAXONOMY_STOPLIST_MODE = _get(
        "I2P_SUBDOMAIN_TAXONOMY_STOPLIST_MODE",
        "drop",
        cast=str,
        cfg_path=["recall", "subdomain_taxonomy_stoplist_mode"],
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
        [0.05, 0.15, 0.25, 0.35, 0.5, 0.65, 0.75, 0.85, 0.95],
        cast=_cast_list_float,
        cfg_path=["anchors", "quantiles"],
    )
    ANCHOR_MAX_INITIAL = _get(
        "I2P_ANCHOR_MAX_INITIAL",
        11,
        cast=int,
        cfg_path=["anchors", "max_initial"],
    )
    ANCHOR_MAX_TOTAL = _get(
        "I2P_ANCHOR_MAX_TOTAL",
        13,
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
        [-0.6, -0.4, -0.2, 0.2, 0.4, 0.6],
        cast=_cast_list_float,
        cfg_path=["anchors", "densify_offsets"],
    )
    ANCHOR_BUCKET_SIZE = _get(
        "I2P_ANCHOR_BUCKET_SIZE",
        1.0,
        cast=float,
        cfg_path=["anchors", "bucket_size"],
    )
    ANCHOR_BUCKET_COUNT = _get(
        "I2P_ANCHOR_BUCKET_COUNT",
        3,
        cast=int,
        cfg_path=["anchors", "bucket_count"],
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
        0.05,
        cast=float,
        cfg_path=["anchors", "densify_loss_threshold"],
    )
    DENSIFY_MIN_AVG_CONF = _get(
        "I2P_DENSIFY_MIN_AVG_CONF",
        0.35,
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

    # ===================== Agentic Search 配置（兼容旧 PATH4 命名） =====================
    AGENTIC_SEARCH_ENABLE = _get_with_legacy(
        "I2P_AGENTIC_SEARCH_ENABLE",
        "I2P_PATH4_ENABLE",
        False,
        cast=bool,
        cfg_path=["agenticSearch", "enable"],
        legacy_cfg_path=["path4", "enable"],
    )
    AGENTIC_SEARCH_MAX_QUERIES = _get_with_legacy(
        "I2P_AGENTIC_SEARCH_MAX_QUERIES",
        "I2P_PATH4_MAX_QUERIES",
        8,
        cast=int,
        cfg_path=["agenticSearch", "max_queries"],
        legacy_cfg_path=["path4", "max_queries"],
    )
    AGENTIC_SEARCH_FINAL_QUERIES = _get_with_legacy(
        "I2P_AGENTIC_SEARCH_FINAL_QUERIES",
        "I2P_PATH4_FINAL_QUERIES",
        5,
        cast=int,
        cfg_path=["agenticSearch", "final_queries"],
        legacy_cfg_path=["path4", "final_queries"],
    )
    AGENTIC_SEARCH_YEAR_RANGE = _get_with_legacy(
        "I2P_AGENTIC_SEARCH_YEAR_RANGE",
        "I2P_PATH4_YEAR_RANGE",
        "2024-2026",
        cast=str,
        cfg_path=["agenticSearch", "year_range"],
        legacy_cfg_path=["path4", "year_range"],
    )
    AGENTIC_SEARCH_LIMIT_PER_QUERY = _get_with_legacy(
        "I2P_AGENTIC_SEARCH_LIMIT_PER_QUERY",
        "I2P_PATH4_SEARCH_LIMIT_PER_QUERY",
        50,
        cast=int,
        cfg_path=["agenticSearch", "search_limit_per_query"],
        legacy_cfg_path=["path4", "search_limit_per_query"],
    )
    AGENTIC_SEARCH_MIN_WHITELIST_PER_QUERY = _get_with_legacy(
        "I2P_AGENTIC_SEARCH_MIN_WHITELIST_PER_QUERY",
        "I2P_PATH4_SEARCH_MIN_WHITELIST_PER_QUERY",
        5,
        cast=int,
        cfg_path=["agenticSearch", "search_min_whitelist_per_query"],
        legacy_cfg_path=["path4", "search_min_whitelist_per_query"],
    )
    AGENTIC_SEARCH_MAX_PAGES_PER_QUERY = _get_with_legacy(
        "I2P_AGENTIC_SEARCH_MAX_PAGES_PER_QUERY",
        "I2P_PATH4_SEARCH_MAX_PAGES_PER_QUERY",
        3,
        cast=int,
        cfg_path=["agenticSearch", "search_max_pages_per_query"],
        legacy_cfg_path=["path4", "search_max_pages_per_query"],
    )
    AGENTIC_SEARCH_VENUE_WHITELIST_MODE = _get_with_legacy(
        "I2P_AGENTIC_SEARCH_VENUE_WHITELIST_MODE",
        "I2P_PATH4_VENUE_WHITELIST_MODE",
        "cs_top",
        cast=str,
        cfg_path=["agenticSearch", "venue_whitelist_mode"],
        legacy_cfg_path=["path4", "venue_whitelist_mode"],
    )
    LLM_TEMPERATURE_AGENTIC_SEARCH_PLANNER = _get_with_legacy(
        "I2P_LLM_TEMPERATURE_AGENTIC_SEARCH_PLANNER",
        "I2P_LLM_TEMPERATURE_PATH4_PLANNER",
        0.0,
        cast=float,
        cfg_path=["llm", "temperature", "agentic_search_planner"],
        legacy_cfg_path=["llm", "temperature", "path4_planner"],
    )
    AGENTIC_SEARCH_BACKTRACK_ENABLE = _get_with_legacy(
        "I2P_AGENTIC_SEARCH_BACKTRACK_ENABLE",
        "I2P_PATH4_BACKTRACK_ENABLE",
        True,
        cast=bool,
        cfg_path=["agenticSearch", "backtrack_enable"],
        legacy_cfg_path=["path4", "backtrack_enable"],
    )
    AGENTIC_SEARCH_BACKTRACK_THRESHOLD = _get_with_legacy(
        "I2P_AGENTIC_SEARCH_BACKTRACK_THRESHOLD",
        "I2P_PATH4_BACKTRACK_THRESHOLD",
        5,
        cast=int,
        cfg_path=["agenticSearch", "backtrack_threshold"],
        legacy_cfg_path=["path4", "backtrack_threshold"],
    )
    AGENTIC_SEARCH_BACKTRACK_EXTRA_QUERIES = _get_with_legacy(
        "I2P_AGENTIC_SEARCH_BACKTRACK_EXTRA_QUERIES",
        "I2P_PATH4_BACKTRACK_EXTRA_QUERIES",
        3,
        cast=int,
        cfg_path=["agenticSearch", "backtrack_extra_queries"],
        legacy_cfg_path=["path4", "backtrack_extra_queries"],
    )

    # Agentic Search Extractor
    AGENTIC_SEARCH_EXTRACTOR_BATCH_SIZE = _get_with_legacy(
        "I2P_AGENTIC_SEARCH_EXTRACTOR_BATCH_SIZE",
        "I2P_PATH4_EXTRACTOR_BATCH_SIZE",
        5,
        cast=int,
        cfg_path=["agenticSearch", "extractor_batch_size"],
        legacy_cfg_path=["path4", "extractor_batch_size"],
    )
    LLM_TEMPERATURE_AGENTIC_SEARCH_EXTRACTOR = _get_with_legacy(
        "I2P_LLM_TEMPERATURE_AGENTIC_SEARCH_EXTRACTOR",
        "I2P_LLM_TEMPERATURE_PATH4_EXTRACTOR",
        0.2,
        cast=float,
        cfg_path=["llm", "temperature", "agentic_search_extractor"],
        legacy_cfg_path=["llm", "temperature", "path4_extractor"],
    )

    # Agentic Search fusion
    AGENTIC_SEARCH_RRF_ENABLE = _get_with_legacy(
        "I2P_AGENTIC_SEARCH_RRF_ENABLE",
        "I2P_PATH4_RRF_ENABLE",
        True,
        cast=bool,
        cfg_path=["agenticSearch", "rrf_enable"],
        legacy_cfg_path=["path4", "rrf_enable"],
    )
    AGENTIC_SEARCH_RRF_K = _get_with_legacy(
        "I2P_AGENTIC_SEARCH_RRF_K",
        "I2P_PATH4_RRF_K",
        60,
        cast=int,
        cfg_path=["agenticSearch", "rrf_k"],
        legacy_cfg_path=["path4", "rrf_k"],
    )
    AGENTIC_SEARCH_RRF_WEIGHT_OLD = _get_with_legacy(
        "I2P_AGENTIC_SEARCH_RRF_WEIGHT_OLD",
        "I2P_PATH4_RRF_WEIGHT_OLD",
        0.7,
        cast=float,
        cfg_path=["agenticSearch", "rrf_weight_old"],
        legacy_cfg_path=["path4", "rrf_weight_old"],
    )
    AGENTIC_SEARCH_RRF_WEIGHT_AGENTIC = _get_with_legacy(
        "I2P_AGENTIC_SEARCH_RRF_WEIGHT_AGENTIC",
        "I2P_PATH4_RRF_WEIGHT",
        0.3,
        cast=float,
        cfg_path=["agenticSearch", "rrf_weight_agentic"],
        legacy_cfg_path=["path4", "rrf_weight"],
    )
    AGENTIC_SEARCH_RRF_TOP_N_OLD = _get_with_legacy(
        "I2P_AGENTIC_SEARCH_RRF_TOP_N_OLD",
        "I2P_PATH4_RRF_TOP_N_OLD",
        10,
        cast=int,
        cfg_path=["agenticSearch", "rrf_top_n_old"],
        legacy_cfg_path=["path4", "rrf_top_n_old"],
    )
    AGENTIC_SEARCH_RRF_TOP_M = _get_with_legacy(
        "I2P_AGENTIC_SEARCH_RRF_TOP_M",
        "I2P_PATH4_TOP_M",
        8,
        cast=int,
        cfg_path=["agenticSearch", "rrf_top_m_agentic"],
        legacy_cfg_path=["path4", "top_m"],
    )
    AGENTIC_SEARCH_RRF_FINAL_TOP_K = _get_with_legacy(
        "I2P_AGENTIC_SEARCH_RRF_FINAL_TOP_K",
        "I2P_PATH4_RRF_FINAL_TOP_K",
        10,
        cast=int,
        cfg_path=["agenticSearch", "rrf_final_top_k"],
        legacy_cfg_path=["path4", "rrf_final_top_k"],
    )

    # Backward-compatible aliases (keep existing code working)
    PATH4_ENABLE = AGENTIC_SEARCH_ENABLE
    PATH4_MAX_QUERIES = AGENTIC_SEARCH_MAX_QUERIES
    PATH4_FINAL_QUERIES = AGENTIC_SEARCH_FINAL_QUERIES
    PATH4_YEAR_RANGE = AGENTIC_SEARCH_YEAR_RANGE
    PATH4_SEARCH_LIMIT_PER_QUERY = AGENTIC_SEARCH_LIMIT_PER_QUERY
    PATH4_SEARCH_MIN_WHITELIST_PER_QUERY = AGENTIC_SEARCH_MIN_WHITELIST_PER_QUERY
    PATH4_SEARCH_MAX_PAGES_PER_QUERY = AGENTIC_SEARCH_MAX_PAGES_PER_QUERY
    PATH4_VENUE_WHITELIST_MODE = AGENTIC_SEARCH_VENUE_WHITELIST_MODE
    LLM_TEMPERATURE_PATH4_PLANNER = LLM_TEMPERATURE_AGENTIC_SEARCH_PLANNER
    PATH4_BACKTRACK_ENABLE = AGENTIC_SEARCH_BACKTRACK_ENABLE
    PATH4_BACKTRACK_THRESHOLD = AGENTIC_SEARCH_BACKTRACK_THRESHOLD
    PATH4_BACKTRACK_EXTRA_QUERIES = AGENTIC_SEARCH_BACKTRACK_EXTRA_QUERIES
    PATH4_EXTRACTOR_BATCH_SIZE = AGENTIC_SEARCH_EXTRACTOR_BATCH_SIZE
    LLM_TEMPERATURE_PATH4_EXTRACTOR = LLM_TEMPERATURE_AGENTIC_SEARCH_EXTRACTOR

    # Blind Judge tau config
    JUDGE_TAU_PATH = _get(
        "I2P_JUDGE_TAU_PATH",
        str(OUTPUT_DIR / "judge_tau.json"),
        cast=Path,
        cfg_path=["critic", "tau_path"],
    )
    JUDGE_TAU_DEFAULT = _get(
        "I2P_JUDGE_TAU_DEFAULT",
        1.0,
        cast=float,
        cfg_path=["critic", "tau_default"],
    )
    TAU_METHODOLOGY = _get(
        "I2P_TAU_METHODOLOGY",
        1.0,
        cast=float,
        cfg_path=["critic", "tau_methodology"],
    )
    TAU_NOVELTY = _get(
        "I2P_TAU_NOVELTY",
        1.0,
        cast=float,
        cfg_path=["critic", "tau_novelty"],
    )
    TAU_STORYTELLER = _get(
        "I2P_TAU_STORYTELLER",
        1.0,
        cast=float,
        cfg_path=["critic", "tau_storyteller"],
    )
    CRITIC_COACH_ENABLE = _get(
        "I2P_CRITIC_COACH_ENABLE",
        True,
        cast=bool,
        cfg_path=["critic", "coach_enable"],
    )
    CRITIC_COACH_TEMPERATURE = _get(
        "I2P_CRITIC_COACH_TEMPERATURE",
        0.3,
        cast=float,
        cfg_path=["critic", "coach_temperature"],
    )
    CRITIC_COACH_MAX_TOKENS = _get(
        "I2P_CRITIC_COACH_MAX_TOKENS",
        4096,
        cast=int,
        cfg_path=["critic", "coach_max_tokens"],
    )
