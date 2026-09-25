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

# ── 3. format_memory_context 三层 + session 标注 ──────────────────
# 保持 v3 三层结构（Profile → Fact → Raw），每条 entry 加 session 归属（H2HMEM 图片定位题需要）。
# 前缀用 R/F/P（不用 D，避免和 image_id 冲突，对齐 v3 三层）。
_orig_format = _bp.format_memory_context


def _format_with_session(context):
    if not context:
        return "None"

    profile = [m for m in context if m.get("memory_type") == "profile"]
    fact = [m for m in context if m.get("memory_type") == "fact"]
    raw = [m for m in context if m.get("memory_type") not in ("fact", "profile")]

    def _fmt(i, m, prefix):
        # 邻居扩窗片段：text 已自包含 speaker 流 + 命中标记，header 用无编号标签
        # （不能用 [R{i}] 编号——图片题会把它误当成 session 号/文件名）
        if m.get("neighbor_window"):
            label = "事实及来源对话" if prefix == "F" else "对话片段"
            return f"[{label}]\n{m.get('text', '')}"
        sid = m.get("session_id", "")
        ts = m.get("date") or m.get("timestamp", "")
        seq = m.get("turn_seq")
        if seq is not None:
            ts = f"{ts} #{seq}"   # 日内绝对序号（同日排序 + fact/raw #N 互相对齐）
        speaker = m.get("speaker") or "?"
        text = m.get("text", "")
        header_parts = [f"[{prefix}{i}]"]
        if sid:
            header_parts.append(f"session: {sid}")
        if ts:
            header_parts.append(f"timestamp: {ts}")
        header_parts.append(f"speaker: {speaker}")
        body = " ".join(header_parts) + f"\n{text}"
        # fact 层带图属性 caption（raw 层 caption 已拼进 text，这里只补 fact 的图-人绑定）
        if prefix == "F":
            caps = m.get("caption", {})
            if caps and isinstance(caps, dict):
                for img_id, cap_text in caps.items():
                    body += f"\n  Image {img_id} caption: {cap_text}"
        return body

    parts = []
    if profile:
        parts.append("[Profile Memory] 用户长期画像（身份、偏好、习惯、健康、目标、性格）")
        for i, m in enumerate(profile):
            parts.append(_fmt(i, m, "P"))
    if fact:
        parts.append("[Fact Memory] LLM 提取的原子事实（指代已消解、结构化）")
        for i, m in enumerate(fact):
            parts.append(_fmt(i, m, "F"))
    if raw:
        parts.append("[Raw_Dialogue] 原始对话 turn（含图片 caption）")
        for i, m in enumerate(raw):
            parts.append(_fmt(i, m, "R"))

    id_note = (
        "[Note] Image references in the memory use the id format "
        "<session>:<image_file> (e.g. \"session1:img10.jpg\" means the image file "
        "img10.jpg shown in session1; the same bare file name in a different "
        "session is a DIFFERENT image). When a question asks for the session "
        "number and image file name, decompose the id into those two parts."
    )
    return "\n".join(["[Memory Start]", id_note] + parts + ["[Memory End]"])


_bp.format_memory_context = _format_with_session

# ── 重新导出（覆盖后的版本）──────────────────────────────────────
from eval.bench_prompts import *  # noqa: F401,F403
