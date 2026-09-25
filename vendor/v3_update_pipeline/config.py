"""
Memory System V2 — Global Configuration

V2 改动（相对 v1）：
  - embedding 4路 → 2路：unified_emb（text+caption拼接）+ caption_emb（单独）
  - 删除 text_index / image_index / text_clip_index（image_emb 和 text_clip_emb 从不被检索用）
  - 新增 unified_index（LTMemory 式单路检索，caption 拼进文本）
  - 其余配置（LLM、检索参数、PREFIX、STORAGE_DIR 等）全部不变
"""

import os as _os
from pathlib import Path

# ── Base Paths ───────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
# bench 数据根（audio_mem_bench / 公开数据集 的父目录）
BENCH_ROOT = BASE_DIR.parent
# 模型缓存根
MODEL_ROOT = BENCH_ROOT / "huggingface.co"

# ── Embedding Models ────────────────────────────────────────
# Qwen3-VL-Embedding（text+image 统一 encoder，2048 维同空间）
QWEN3VL_EMBED_PATH = _os.environ.get(
    "QWEN3VL_EMBED_PATH", "Qwen/Qwen3-VL-Embedding-2B"
)

# Embedding 维度（V2: 2路，统一用 Qwen3-VL-Embedding 2048维）
UNIFIED_EMB_DIM = 2048   # unified: text+caption 拼接后的单一路
CAPTION_EMB_DIM = 2048   # caption: 图描述单独一路（同 encoder 同空间）

# ── LLM API（美团统一入口，两个 key 轮换）─────────────────
LLM_API_KEYS = [
    value.strip()
    for value in _os.environ.get("LLM_API_KEYS", _os.environ.get("OPENAI_API_KEY", "")).split(",")
    if value.strip()
]
LLM_API_KEY = LLM_API_KEYS[0] if LLM_API_KEYS else ""
LLM_BASE_URL = _os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
# 多模型轮询（429/限流时自动换下一个）
LLM_MODELS = [
    "gpt-4o-2024-11-20",
    "gpt-4o-2024-08-06",
    "gpt-4o-2024-05-13",
    "gpt-4.1",
]
LLM_MODEL = "gpt-4o-2024-11-20"  # 对齐 LTMemory 原版默认模型
LLM_TEMPERATURE = 0.0  # 全程temp=0(答题/judge/route/rewrite/sufficiency)保证可复现
LLM_MAX_TOKENS = 4096
# 429 限流：超过此时长（秒）直接 raise，不跳过 turn
LLM_RATE_LIMIT_TIMEOUT = int(_os.environ.get("LLM_RATE_LIMIT_TIMEOUT", 4 * 3600))
LLM_MAX_RETRIES = 5  # 增加重试避免 vLLM 连接错误导致崩溃
LLM_RETRY_DELAY = int(_os.environ.get("LLM_RETRY_DELAY", 10))  # seconds

# ── 检索期 LLM 后端（route / sufficiency / rewrite 用哪个 LLM）─────────
# None   = 原行为：call_llm 走美团 gpt-4.1/4o 轮换（默认，pipeline 完全照旧）
# "vllm" = 调本地 vllm 部署的 Qwen3-Omni 或 Qwen3-VL（utils.vllm_client.call_vllm）
RETRIEVAL_LLM_BACKEND = _os.environ.get("RETRIEVAL_LLM_BACKEND", "vllm")

# ── 答题 LLM 后端（answer generation 用哪个 LLM）────────────────────────
ANSWER_LLM_BACKEND = __import__('os').environ.get("ANSWER_LLM_BACKEND") or None

# 答题专用 vllm 实例（仅 ANSWER_LLM_BACKEND=vllm 时生效）
ANSWER_VLLM_BASE_URL = __import__('os').environ.get("ANSWER_VLLM_BASE_URL") or None

# Qwen3-VL-8B 配置（默认 8901，可用 VLLM_BASE_URL 环境变量覆盖到其他实例，如 8902）
VLLM_BASE_URL = __import__('os').environ.get("VLLM_BASE_URL", "http://localhost:8901/v1")
VLLM_MODEL = "Qwen3-VL-8B-Instruct"

# Qwen3-VL-30B 配置（端口 8902）- 备选
VLLM_QWEN3VL_URL = "http://localhost:8902/v1"
VLLM_QWEN3VL_MODEL = "Qwen3-VL-30B-A3B-Instruct"

# ── 检索流程版本（仅 RETRIEVAL_LLM_BACKEND="vllm" 时生效；None=gpt 走原流程）─────────
RETRIEVAL_MODE = "orig"

# ── 答题 context 是否按 session 分组排序 ──
SESSION_SORT = True

# ── Storage Paths ───────────────────────────────────────────
STORAGE_DIR = Path(_os.environ.get("VOXPOLYMEM_STORAGE_DIR", BASE_DIR / "storage"))
STORAGE_DIR.mkdir(exist_ok=True)

RAW_DB_PATH = STORAGE_DIR / "raw.db"
SEMANTIC_DB_PATH = STORAGE_DIR / "semantic.db"
EPISODIC_DB_PATH = STORAGE_DIR / "episodic.db"
REGISTRY_DB_PATH = STORAGE_DIR / "registry.db"

# faiss 索引文件（V2: 2路）
FAISS_DIR = STORAGE_DIR / "faiss"
FAISS_DIR.mkdir(exist_ok=True)
UNIFIED_INDEX_PATH = FAISS_DIR / "unified_emb.index"    # V2: text+caption 拼接单路
CAPTION_INDEX_PATH = FAISS_DIR / "caption_emb.index"     # V2: caption 单独（search_by_caption 用）
FACT_INDEX_PATH = FAISS_DIR / "fact_emb.index"           # V3: fact 层单独（search_fact 用，原子事实）
PROFILE_INDEX_PATH = FAISS_DIR / "profile_emb.index"     # V3: profile 层单独（search_profile 用，用户长期画像）
VOICEPRINT_INDEX_PATH = FAISS_DIR / "voiceprint.index"

# ── 检索参数 ─────────────────────────────────────────────────
DEDUP_THRESHOLD = 0.85        # check_dedup 相似度阈值
TOP_K_DEFAULT = 30            # 对齐 LTMemory (DEFAULT_RETRIEVAL_TOP_K=30)
MAX_ITER = 5                  # 多轮检索最大迭代
SUFFICIENT_CONFIDENCE = 0.7   # 充分性置信度阈值
EARLY_STOP_NO_NEW = 2         # 连续N轮无新增则早停

# ── mem_id 前缀 ──────────────────────────────────────────────
PREFIX_RAW = "raw"
PREFIX_SEM = "sem"
PREFIX_EPI = "epi"
PREFIX_COR = "cor"
