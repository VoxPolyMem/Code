"""
H2HMEM 专用 prompt/检索适配 —— 不改标准文件，运行时 monkey-patch 覆盖 H2HMEM 需要的部分。

标准文件（bench_prompts.py / retrieval/tools.py）保持不变，H2HMEM 的差异全在这里：
  1. tools.RetrievalTools._to_retrieved 补 session_id（H2HMEM 图片定位题需要 session 归属）
  2. bench_prompts.FORMAT_CONSTRAINTS["CD"] 强化（H2HMEM 的 CD 是"某 speaker 是否说了 X"
     的说话人归因矛盾，不是 Mem-Gallery 的"值随时间变化"）
  3. bench_prompts.format_memory_context 加 session 标注（图片定位题需要 session 号）

用法：eval 脚本里 `from eval.h2hmem_prompts import ...`（而不是 bench_prompts），
     并在检索前先 `import eval.h2hmem_prompts`（触发 monkey-patch）。
"""
from eval import bench_prompts as _bp
from retrieval import tools as _tools

# ── 1. _to_retrieved 补 session_id ────────────────────────────────
_orig_to_retrieved = _tools.RetrievalTools._to_retrieved


def _to_retrieved_with_session(self, mem):
    r = _orig_to_retrieved(self, mem)
    r["session_id"] = mem.get("session_id", "")
    return r


_tools.RetrievalTools._to_retrieved = _to_retrieved_with_session

# ── 2. CD constraint 强化（说话人归因）────────────────────────────
_bp.FORMAT_CONSTRAINTS = dict(_bp.FORMAT_CONSTRAINTS)  # 复制，不改标准 dict
_bp.FORMAT_CONSTRAINTS["CD"] = (
    "This is a CONTRADICTION DETECTION task. The question makes a claim about who "
    "said/did something, but the claim may be WRONG (attributed to the wrong speaker, "
    "or facts that don't match the conversation). Carefully check WHO actually said/did "
    "what in the retrieved memories. If the claim attributes something to the WRONG "
    "speaker or contradicts the actual conversation, that is a CONTRADICTION — answer Yes. "
    "Otherwise answer No. Reply strictly with either “Yes.” or “No.”"
)

# ── 3. format_memory_context 加 session 标注 ──────────────────────
_orig_format = _bp.format_memory_context


def _format_with_session(context):
    if not context:
        return "None"
    processed = []
    for idx, m in enumerate(context):
        text = m.get("text", "")
        ts = m.get("date") or m.get("timestamp", "")
        sid = m.get("session_id", "")
        header_parts = []
        if sid:
            header_parts.append(f"session: {sid}")
        if ts:
            header_parts.append(f"timestamp: {ts}")
        header = " ".join(header_parts)
        if header:
            formatted = f"{header}\n{text}"
        else:
            formatted = text
        processed.append(f"[Memory {idx}] {formatted}")
    return "\n".join(["[Memory Start]"] + processed + ["[Memory End]"])


_bp.format_memory_context = _format_with_session

# ── 重新导出（覆盖后的版本）──────────────────────────────────────
from eval.bench_prompts import *  # noqa: F401,F403
