"""
Benchmark Prompts V2 — 完全对齐 LTMemory (ttyclear Mem-Gallery run_bench.py)

V2 改动（相对 v1）：
  完全照搬 LTMemory 的 prompt 结构，不再用自己写的 answer_system_prompt/answer_user_prompt。
  - system prompt = sys_prompt.txt 原文（LTMemory 的 SystemPrompt）
  - user prompt = TextMsgPrompt(memory_context) + DialogueAgentPrompt(question, speaker_a, speaker_b, format_constraint)
  - context 格式 = ConcateUtilization: "[Memory {idx}] timestamp: {date}\n{text}"
  - format_constraint: AR/CD/VS 用官方 prompt 文件原文，其他题型无 constraint
  - judge prompt = 官方 llm_judge.txt 原文（详细 5 档评分标准）

这样 baseline 模式就是 LTMemory 的精确复现。
multi-round 模式也用同一套 prompt，保证可比。
"""

# ── LTMemory 官方 prompt 原文（从 run_bench.py / prompt/ 目录照搬）──────────────

import os

SYSTEM_PROMPT = (
    "You are an AI assistant evaluated on multimodal long-term conversational memory.\n"
    "For the given question-answering task, your responses must be concise, yet complete enough to accurately answer the questions.\n"
    "If multiple pieces of information about the same event appear in the conversation, always rely on the most recent information.\n\n"
    "The question-answering evaluation will contain several multimodal task types:\n\n"
    "Factual Retrieval: Retrieve explicit facts mentioned in the conversation for the answer.\n\n"
    "Multi-entity Reasoning: Combine the retrieved information to reason and infer an answer.\n\n"
    "Temporal Reasoning: Resolve time-dependent questions.\n\n"
    "Visual-centric Reasoning: Besides textual information, answer questions using visual images in the conversation.\n\n"
    "Test-time Learning: Learn new visual knowledge from provided images within historical dialogue and use it in question-answering.\n\n"
    "Visual-centric Search: Find the image(s) that match the information in a given query and return their image ID(s).\n\n"
    "Conflict Detection: Detect contradictions between the conversation history and the information provided in the question.\n\n"
    "Knowledge Resolution: Resolve knowledge conflicts or updates by prioritizing the most recent information.\n\n"
    "Answer Refusal: Decline to answer when the information does not exist in the conversation history.\n\n"
    "Follow all instructions strictly. Only answer using information contained within the multimodal conversation. "
    "Do not hallucinate. Always remain consistent and grounded in the dialogue history."
)

# TextMsgPrompt（run_bench.py line 110-114）
TEXT_MSG_PROMPT = (
    "\nThe retrieved memory contents are as follows:\n\n"
    "{memory_context}\n"
)

# DialogueAgentPrompt（run_bench.py line 129-137）
DIALOGUE_AGENT_PROMPT = (
    "\nYour task is to answer the question about the conversation between {speaker_a} and {speaker_b} "
    "in a concise manner with the help of memory content.\n"
    "Please only provide the content of the answer, without including introductory phrases like 'answer:'.\n"
    "For questions that require answering a date or time, strictly follow the format and provide a specific date or time whenever possible.\n"
    "Generate answers primarily concise, yet complete enough to accurately answer the questions.\n\n"
    "The current question is as follows:\n"
    "{observation} {format_constraint}\n"
)

# DialogueAgentPromptImage（run_bench.py line 139-141）
DIALOGUE_AGENT_IMAGE_PROMPT = "\nHere is the attached image of the question:\n"

# ── per-type format_constraint（官方 prompt/ar_prompt.txt, cd_prompt.txt, vs_prompt.txt）──

FORMAT_CONSTRAINTS = {
    "AR": "Provide your answer based on the information in the conversation. Only if the information about the question is not present in the conversation, reply with: “Not mentioned.”",
    "CD": "Please check whether this information conflicts with the conversation, and reply strictly with either “Yes.” or “No.”",
    "VS": "Return the image_id of the image(s). If there are multiple images, sort them in ascending order and separate them with commas. Format example: “D2:IMG_003, D2:IMG_010, D10:IMG_002” (for format reference only).",
}


def get_format_constraint(point: str) -> str:
    """AR/CD/VS 返回官方 format_constraint，其他题型返回空。"""
    return FORMAT_CONSTRAINTS.get(point, "")


# 旧接口兼容（eval_all_mem_gallery.py 调用 get_hint）
def get_hint(point: str, dataset: str = "mem_gallery") -> str:
    """返回 format_constraint（对齐 LTMemory 的 per-type prompt）。"""
    if dataset == "mem_gallery":
        return get_format_constraint(point)
    return ""


# ── LMTruncation（对齐 LTMemory: token mode, 4096, keep last）──────────────────

_QWEN_TOKENIZER = None
_QWEN_TOKENIZER_PATH = os.environ.get(
    "QWEN_TOKENIZER_PATH", "Qwen/Qwen3-VL-8B-Instruct"
)

def _get_qwen_tokenizer():
    """Lazy-load Qwen3-VL-8B tokenizer（对齐 LTMemory DEFAULT_BACKBONE_PATH）。"""
    global _QWEN_TOKENIZER
    if _QWEN_TOKENIZER is None:
        try:
            from transformers import AutoTokenizer
            _QWEN_TOKENIZER = AutoTokenizer.from_pretrained(
                _QWEN_TOKENIZER_PATH, trust_remote_code=True)
        except Exception as e:
            print(f"  [warn] Qwen tokenizer load failed: {e}, truncation disabled")
            _QWEN_TOKENIZER = False
    return _QWEN_TOKENIZER if _QWEN_TOKENIZER is not False else None


def truncate_memory_context(text: str, max_tokens: int = 4096) -> str:
    """对齐 LTMemory LMTruncation.truncate_by_token:
    tokenize → 如果超 max_tokens → 保留最后 max_tokens 个 token → 转回 string。

    实测发现：keep-last 会丢掉最佳记忆（排在前面的 high-score mem），
    对 VS 题型（需要 caption）伤害大（0.885→0.615）。AI_Robotics 大部分
    question 的 30 条记忆 < 4096 tokens（不截断），只有少数 VS 题超限。
    LTMemory 也用 keep-last，但其总体 86.49% 说明大部分 case 不触发截断。
    这里保留函数但暂不启用（return text unchanged），等需要时再开。
    """
    # ── 暂不启用截断（实测 keep-last 伤害 VS）──
    return text
    # tok = _get_qwen_tokenizer()
    # if tok is None:
    #     return text
    # tokens = tok.tokenize(text)
    # if len(tokens) > max_tokens:
    #     tokens = tokens[-max_tokens:]
    #     text = tok.convert_tokens_to_string(tokens)
    # return text


# ── context 格式化（对齐 LTMemory ConcateUtilization）─────────────────────────

def format_memory_context(context: list[dict]) -> str:
    """V3: 三层分层展示（Profile → Fact → Raw）。

    - Profile 层（memory_type=profile，用户长期画像）在最上面
    - Fact 层（memory_type=fact，LLM 提取的原子事实）在中间
    - Raw 层（memory_type=episodic，原文对话 + caption）在最下面
    - 每层带标题 + 一句简单解释；每条打印 [id] + timestamp + speaker + text
    - 保持传入顺序（orchestrator 已按 score / TR-time 排好），这里只做分层展示、不重排
    - 兼容 flat（无 fact/profile 层时只展示 raw 块，行为接近原来的 LTMemory 格式）
    """
    if not context:
        return "None"

    profile = [m for m in context if m.get("memory_type") == "profile"]
    fact = [m for m in context if m.get("memory_type") == "fact"]
    raw = [m for m in context if m.get("memory_type") not in ("fact", "profile")]

    def _fmt_entry(i: int, m: dict, prefix: str) -> str:
        # 邻居扩窗片段：text 已自包含 speaker 流 + 命中标记，header 用无编号标签
        # （不能用 [R{i}] 编号——图片题会把它误当成 session 号/文件名）
        if m.get("neighbor_window"):
            label = "事实及来源对话" if prefix == "F" else "对话片段"
            return f"[{label}]\n{m.get('text', '')}"
        ts = m.get("date") or m.get("timestamp", "")
        seq = m.get("turn_seq")
        if seq is not None:
            ts = f"{ts} #{seq}"   # 日内绝对序号（同日排序 + fact/raw #N 互相对齐）
        speaker = m.get("speaker") or "?"
        text = m.get("text", "")
        header = f"[{prefix}{i}] timestamp: {ts} | speaker: {speaker}"
        body = f"{header}\n{text}" if ts or speaker else f"[{prefix}{i}] {text}"
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
            parts.append(_fmt_entry(i, m, "P"))
    if fact:
        parts.append("[Fact Memory] LLM 提取的原子事实（指代已消解、结构化）")
        for i, m in enumerate(fact):
            parts.append(_fmt_entry(i, m, "F"))
    if raw:
        parts.append("[Raw_Dialogue] 原始对话 turn（含图片 caption）")
        for i, m in enumerate(raw):
            # 前缀用 R 不用 D：D{n} 会和 Mem-Gallery 的 image_id 格式 D{n}:IMG_{m} 冲突
            #（答题模型把 [D0] 误当成图片编号，输出 D0:IMG_002 而非 D1:IMG_002）
            parts.append(_fmt_entry(i, m, "R"))

    return "\n".join(["[Memory Start]"] + parts + ["[Memory End]"])


def format_memory_context_with_caption(context: list[dict]) -> str:
    """和 format_memory_context 一样（V3 分层 + [D/F] 前缀），但在 raw 层每条记忆
    的 text 后面加 caption。

    caption 格式：'  Image {img_id} caption: {cap_text}'（和 _format_history 一致）。
    供 _insert_images_inline 正则匹配用，把图片内联到 caption 位置。

    注意：必须和 format_memory_context 保持完全一致的文本结构（除了多出 caption 行），
    否则 inline vs append 对比会被文本格式差异混淆。
    """
    if not context:
        return "None"

    raw = [m for m in context if m.get("memory_type") != "fact"]
    fact = [m for m in context if m.get("memory_type") == "fact"]

    parts = []
    if raw:
        parts.append("## Original Dialogue")
        for i, m in enumerate(raw):
            text = m.get("text", "")
            ts = m.get("date") or m.get("timestamp", "")
            entry = f"[D{i}] timestamp: {ts}\n{text}" if ts else f"[D{i}] {text}"
            # 加 caption（和 prompts.py _format_history 格式一致）
            caps = m.get("caption", {})
            if caps and isinstance(caps, dict):
                for img_id, cap_text in caps.items():
                    entry += f"\n  Image {img_id} caption: {cap_text}"
            parts.append(entry)
    if fact:
        parts.append("## Atomic Facts")
        for i, m in enumerate(fact):
            text = m.get("text", "")
            ts = m.get("date") or m.get("timestamp", "")
            parts.append(f"[F{i}] timestamp: {ts}\n{text}" if ts else f"[F{i}] {text}")

    return "\n".join(["[Memory Start]"] + parts + ["[Memory End]"])


def _insert_images_inline(memory_context_text: str, memory_images: list[dict]) -> list[dict]:
    """用正则匹配 caption 行，在 caption 后插入对应的图片 image_url block。

    memory_images: [{"mem_id":..., "image_id":..., "base64":...}, ...]
    返回 content blocks 列表（text + image_url 交替），图片紧跟在 caption 行后面。
    """
    import re
    if not memory_images:
        return [{"type": "text", "text": memory_context_text}]

    # image_id → base64 映射
    img_map = {mi.get("image_id", ""): mi.get("base64", "") for mi in memory_images}

    # 匹配 caption 行：'  Image {img_id} caption: {cap_text}'
    pattern = re.compile(r'  Image\s+(\S+)\s+caption:[^\n]*', re.MULTILINE)

    blocks = []
    last_end = 0
    for match in pattern.finditer(memory_context_text):
        img_id = match.group(1)
        if img_id not in img_map:
            continue  # 没有对应图片，跳过
        # caption 行文本（含该行）
        blocks.append({"type": "text", "text": memory_context_text[last_end:match.end()]})
        # 图片（紧跟 caption 行后）
        blocks.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img_map[img_id]}"},
        })
        last_end = match.end()

    if last_end < len(memory_context_text):
        blocks.append({"type": "text", "text": memory_context_text[last_end:]})

    return blocks


# ── answer prompt（对齐 LTMemory fast_run_with_textual_memory）──────────────────

def answer_messages(query: str, context: list[dict],
                    character: str = "user", last_date: str = "unknown",
                    hint: str = "") -> list[dict]:
    """对齐 LTMemory 的 answer prompt 结构:
    system = SystemPrompt (sys_prompt.txt)
    user = [TextMsgPrompt(memory_context), DialogueAgentPrompt(question, speaker_a, speaker_b, format_constraint)]

    hint 参数 = format_constraint（AR/CD/VS 有，其他题型空）。
    """
    memory_context = format_memory_context(context)
    # 对齐 LTMemory LMTruncation: 4096 tokens, keep last
    memory_context = truncate_memory_context(memory_context)
    text_memory_prompt = TEXT_MSG_PROMPT.format(memory_context=memory_context)

    speaker_a = f"user ({character})" if character else "user"
    speaker_b = "assistant"
    format_constraint = hint or ""
    query_prompt = DIALOGUE_AGENT_PROMPT.format(
        observation=query,
        speaker_a=speaker_a,
        speaker_b=speaker_b,
        format_constraint=format_constraint,
    )

    # user content: text_memory_prompt + query_prompt（对齐 fast_run_with_textual_memory 无图分支）
    user_content = [
        {"type": "text", "text": text_memory_prompt},
        {"type": "text", "text": query_prompt},
    ]

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def answer_messages_with_image(query: str, context: list[dict],
                               image_base64: str,
                               character: str = "user", last_date: str = "unknown",
                               hint: str = "") -> list[dict]:
    """对齐 LTMemory fast_run_with_textual_memory 有图分支:
    user = [TextMsgPrompt(memory_context), DialogueAgentPrompt, DialogueAgentPromptImage, image]
    """
    memory_context = format_memory_context(context)
    # 对齐 LTMemory LMTruncation: 4096 tokens, keep last
    memory_context = truncate_memory_context(memory_context)
    text_memory_prompt = TEXT_MSG_PROMPT.format(memory_context=memory_context)

    speaker_a = f"user ({character})" if character else "user"
    speaker_b = "assistant"
    format_constraint = hint or ""
    query_prompt = DIALOGUE_AGENT_PROMPT.format(
        observation=query,
        speaker_a=speaker_a,
        speaker_b=speaker_b,
        format_constraint=format_constraint,
    )

    user_content = [
        {"type": "text", "text": text_memory_prompt},
        {"type": "text", "text": query_prompt},
        {"type": "text", "text": DIALOGUE_AGENT_IMAGE_PROMPT},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
    ]

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


# 记忆图片补充说明（with_visual 实验用）
MEMORY_IMAGES_PROMPT = (
    "\nThe following images are from the memories above, labeled by their memory ID. "
    "Use them as supplementary visual evidence alongside the caption text:\n"
)


def answer_messages_with_memory_images(query: str, context: list[dict],
                                       memory_images: list[dict],
                                       question_image_base64: str = None,
                                       character: str = "user",
                                       last_date: str = "unknown",
                                       hint: str = "") -> list[dict]:
    """with_visual 答题 prompt：文本上下文(含caption) + top-K 记忆原图 + 可选问题图。

    memory_images: [{"mem_id":..., "image_id":..., "base64":...}, ...]（已压缩、已限 K）。
    question_image_base64: 问题图片 base64（可选，有则附加在最后）。

    user content 顺序（对齐 LTMemory 结构 + 记忆图补充 + 问题图）：
      1. TextMsgPrompt(memory_context) —— 文本上下文（含 caption，和 answer_messages 一样）
      2. MEMORY_IMAGES_PROMPT + 每张记忆图 [Memory {mem_id}] image {image_id} + image_url
      3. DialogueAgentPrompt(question, ...) —— 问题
      4. 若有 question image：DialogueAgentImagePrompt + image_url
    """
    memory_context = format_memory_context(context)
    memory_context = truncate_memory_context(memory_context)
    text_memory_prompt = TEXT_MSG_PROMPT.format(memory_context=memory_context)

    speaker_a = f"user ({character})" if character else "user"
    speaker_b = "assistant"
    format_constraint = hint or ""
    query_prompt = DIALOGUE_AGENT_PROMPT.format(
        observation=query,
        speaker_a=speaker_a,
        speaker_b=speaker_b,
        format_constraint=format_constraint,
    )

    user_content = [
        {"type": "text", "text": text_memory_prompt},
    ]

    # 记忆原图（按 mem_id 标注，让模型能映射到文本上下文中的对应记忆）
    if memory_images:
        user_content.append({"type": "text", "text": MEMORY_IMAGES_PROMPT})
        for mi in memory_images:
            label = f"[Memory {mi.get('mem_id', '?')}] image {mi.get('image_id', '?')}"
            user_content.append({"type": "text", "text": label})
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{mi['base64']}"},
            })

    # 问题 prompt
    user_content.append({"type": "text", "text": query_prompt})

    # 问题图片（可选，附加在最后，和 answer_messages_with_image 一致）
    if question_image_base64:
        user_content.append({"type": "text", "text": DIALOGUE_AGENT_IMAGE_PROMPT})
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{question_image_base64}"},
        })

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def answer_messages_with_inline_memory_images(query: str, context: list[dict],
                                              memory_images: list[dict],
                                              question_image_base64: str = None,
                                              character: str = "user",
                                              last_date: str = "unknown",
                                              hint: str = "") -> list[dict]:
    """with_visual inline 版：图片内联到对应 caption 位置 + 可选问题图。

    和 answer_messages_with_memory_images 的区别：
    - 用 format_memory_context_with_caption 生成含 caption 的文本
    - 用 _insert_images_inline 把图片插入到对应 caption 行后面
    - 不用 MEMORY_IMAGES_PROMPT（图片已在对应位置）

    user content 顺序：
      1. TEXT_MSG_PROMPT 前缀
      2. inline blocks（text + image_url 交替，图片紧跟 caption）
      3. TEXT_MSG_PROMPT 后缀 + query_prompt
      4. 若有 question image：DialogueAgentImagePrompt + image_url
    """
    # 1. 生成含 caption 的文本上下文
    memory_context_text = format_memory_context_with_caption(context)

    # 2. 内联图片到 caption 位置
    inline_blocks = _insert_images_inline(memory_context_text, memory_images)

    # 3. TEXT_MSG_PROMPT 拆成前缀 + inline blocks + 后缀
    text_prefix = "\nThe retrieved memory contents are as follows:\n\n"
    text_suffix = "\n"

    speaker_a = f"user ({character})" if character else "user"
    speaker_b = "assistant"
    format_constraint = hint or ""
    query_prompt = DIALOGUE_AGENT_PROMPT.format(
        observation=query,
        speaker_a=speaker_a,
        speaker_b=speaker_b,
        format_constraint=format_constraint,
    )

    user_content = [{"type": "text", "text": text_prefix}]
    user_content.extend(inline_blocks)
    user_content.append({"type": "text", "text": text_suffix + query_prompt})

    # 问题图片（可选，附加在最后，和 answer_messages_with_image 一致）
    if question_image_base64:
        user_content.append({"type": "text", "text": DIALOGUE_AGENT_IMAGE_PROMPT})
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{question_image_base64}"},
        })

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


# ── 旧接口兼容（eval_all_mem_gallery.py 调用 answer_system_prompt / answer_user_prompt）──

def answer_system_prompt(character: str, last_date: str, hint: str = "") -> str:
    """返回 LTMemory 的 SystemPrompt（sys_prompt.txt 原文）。"""
    return SYSTEM_PROMPT


def answer_user_prompt(question: str, context: list[dict], hint: str = "") -> str:
    """旧接口：返回拼好的 user prompt 文本（无图）。
    新代码应直接用 answer_messages() 返回 messages list。
    """
    memory_context = format_memory_context(context)
    memory_context = truncate_memory_context(memory_context)
    text_memory_prompt = TEXT_MSG_PROMPT.format(memory_context=memory_context)
    # character 从 context 里拿不到，用通用值
    query_prompt = DIALOGUE_AGENT_PROMPT.format(
        observation=question,
        speaker_a="user",
        speaker_b="assistant",
        format_constraint=hint or "",
    )
    return text_memory_prompt + query_prompt


# ── LLM-Judge prompt（对齐官方 llm_judge.txt）──────────────────────────────────

JUDGE_PROMPT = (
    "You are an impartial judge evaluating the memory capabilities of an AI assistant "
    "with the question-answering task.\n"
    "Your task is to compare the Assistant's Answer against the Ground Truth and assign a score of "
    "0, 0.25, 0.5, 0.75, or 1.\n\n"
    "### Scoring Rubric\n\n"
    "**Score 0 (Incorrect / Miss):**\n\n"
    "- The answer contradicts the Ground Truth.\n"
    "- For Yes/No questions: The answer has the wrong polarity (e.g., says \"Yes\" when Ground Truth is \"No\").\n"
    "- For Open-ended questions: The answer provides factually wrong information or hallucinations.\n"
    "- The assistant fails to provide the required information.\n\n"
    "**Score 0.25 (Poor / Tangential):**\n\n"
    "- The answer touches on the topic but misses the **core entity** or key value required.\n"
    "- The answer contains a mix of minor correct details and **significant hallucinations** or wrong associations.\n"
    "- The answer is excessively vague to the point of being useless (e.g., answering \"a dog\" instead of \"a golden retriever\").\n\n"
    "**Score 0.5 (Partial / Vague):**\n\n"
    "- The answer is technically correct, but lacks confidence or is incomplete.\n"
    "- The answer captures the **main entity or concept** correctly but misses a part of the required supporting details.\n"
    "- For Yes/No questions: The polarity is correct, but the reasoning is flawed (if have), or the assistant is uncertain "
    "(e.g., \"I think it might be Yes\").\n"
    "- For Open-ended questions: The answer is too general or misses key adjectives/details present in the Ground Truth.\n\n"
    "**Score 0.75 (Good / Minor Imperfection):**\n\n"
    "- The answer is largely accurate and captures the core information confidently.\n"
    "- It misses only **minor details** (e.g., specific adjectives or secondary details) that do not alter the main truth.\n"
    "- The answer contains all the correct information but includes unnecessary \"fluff\" or slight conversational filler "
    "that reduces precision.\n\n"
    "**Score 1 (Correct / Exact):**\n\n"
    "- The answer is accurate, precise, and confident.\n"
    "- For Yes/No questions: The polarity matches the Ground Truth perfectly.\n"
    "- For Open-ended questions: The answer contains **all** the core information and necessary details required by the "
    "Ground Truth without hallucinations.\n\n"
    "### Input Data\n\n"
    "Question: {question}\n"
    "Ground Truth: {ground_truth}\n"
    "Assistant Answer: {model_output}\n\n"
    "### Output Format\n\n"
    "Output strictly in the following JSON format:\n"
    '{{"score": <0, 0.25, 0.5, 0.75, or 1>, "reasoning": "<short explanation>"}}'
)


def judge_messages(question: str, gt: str, prediction: str) -> list[dict]:
    """LLM-judge（对齐官方 llm_judge.txt，无 system prompt）。"""
    user = JUDGE_PROMPT.format(
        question=question,
        ground_truth=gt,
        model_output=prediction,
    )
    return [
        {"role": "user", "content": user},
    ]


# ── MemEye MCQ 答题 prompt + 评分────────────────────────────────────────

MCQ_SYSTEM_PROMPT = (
    "You are an AI assistant evaluated on multimodal long-term conversational memory.\n"
    "For the given multiple-choice question, you must select the BEST answer from the "
    "four options (A, B, C, D).\n\n"
    "STRICT RULES — follow exactly:\n"
    "1. Always use the retrieved memory to answer. No exceptions.\n"
    '2. Reply with ONLY the letter of the correct option: "A", "B", "C", or "D".\n'
    "   Do NOT include any explanation, preamble, or option text.\n"
    "3. You MUST choose one of A/B/C/D. Never say 'Not mentioned' or refuse to answer.\n"
    "4. Do NOT infer, guess, or hallucinate. Pick the option best supported by memory.\n"
    "5. If multiple pieces of information about the same event appear, rely on the most recent."
)

# MCQ 选项注入格式（放在 DialogueAgentPrompt 的 format_constraint 位置）
MCQ_OPTIONS_PROMPT = (
    "\n\nPlease select the best answer from the following options:\n"
    "{options_text}\n\n"
    "Reply with ONLY the letter (A, B, C, or D)."
)


def answer_messages_mcq(query: str, context: list[dict],
                        options_dict: dict,
                        character: str = "user",
                        last_date: str = "unknown") -> list[dict]:
    """MemEye MCQ 答题 prompt：文本上下文 + 问题 + 注入选项，要求回答 A/B/C/D。

    options_dict: {"A": "...", "B": "...", "C": "...", "D": "...", "answer": "A"}
    取 A/B/C/D 四个键的文本注入 prompt，'answer' 键忽略。
    """
    memory_context = format_memory_context(context)
    memory_context = truncate_memory_context(memory_context)
    text_memory_prompt = TEXT_MSG_PROMPT.format(memory_context=memory_context)

    # 选项文本
    options_text = "\n".join(
        f"{k}. {options_dict[k]}"
        for k in sorted(options_dict.keys()) if k in ("A", "B", "C", "D")
    )
    mcq_constraint = MCQ_OPTIONS_PROMPT.format(options_text=options_text)

    speaker_a = f"user ({character})" if character else "user"
    speaker_b = "assistant"
    query_prompt = DIALOGUE_AGENT_PROMPT.format(
        observation=query,
        speaker_a=speaker_a,
        speaker_b=speaker_b,
        format_constraint=mcq_constraint,
    )

    user_content = [
        {"type": "text", "text": text_memory_prompt},
        {"type": "text", "text": query_prompt},
    ]

    return [
        {"role": "system", "content": MCQ_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def answer_messages_mcq_with_images(query: str, context: list[dict],
                                    memory_images: list[dict],
                                    options_dict: dict,
                                    character: str = "user",
                                    last_date: str = "unknown") -> list[dict]:
    """MemEye MCQ + 记忆原图 prompt：文本上下文(含dense caption) + top-K 记忆原图 + 选项。

    用于视觉识别题（X4: UNO 游戏状态、健康仪表盘等），caption 描述不清的细粒度
    视觉信息通过原图补充。system prompt 用 MCQ_SYSTEM_PROMPT（要求回答 A/B/C/D）。

    memory_images: [{"mem_id":..., "image_id":..., "base64":...}, ...]（已压缩、已限 K）。
    """
    memory_context = format_memory_context(context)
    memory_context = truncate_memory_context(memory_context)
    text_memory_prompt = TEXT_MSG_PROMPT.format(memory_context=memory_context)

    # MCQ 选项注入
    options_text = "\n".join(
        f"{k}. {options_dict[k]}"
        for k in sorted(options_dict.keys()) if k in ("A", "B", "C", "D")
    )
    mcq_constraint = MCQ_OPTIONS_PROMPT.format(options_text=options_text)

    speaker_a = f"user ({character})" if character else "user"
    speaker_b = "assistant"
    query_prompt = DIALOGUE_AGENT_PROMPT.format(
        observation=query,
        speaker_a=speaker_a,
        speaker_b=speaker_b,
        format_constraint=mcq_constraint,
    )

    user_content = [
        {"type": "text", "text": text_memory_prompt},
    ]

    # 记忆原图（按 mem_id 标注，让模型能映射到文本上下文中的对应记忆）
    if memory_images:
        user_content.append({"type": "text", "text": MEMORY_IMAGES_PROMPT})
        for mi in memory_images:
            label = f"[Memory {mi.get('mem_id', '?')}] image {mi.get('image_id', '?')}"
            user_content.append({"type": "text", "text": label})
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{mi['base64']}"},
            })

    user_content.append({"type": "text", "text": query_prompt})

    return [
        {"role": "system", "content": MCQ_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def answer_messages_mcq_with_inline_images(query: str, context: list[dict],
                                            memory_images: list[dict],
                                            options_dict: dict,
                                            character: str = "user",
                                            last_date: str = "unknown") -> list[dict]:
    """MemEye MCQ + 记忆原图 inline 版：图片内联到对应 caption 位置，不放在末尾。

    和 answer_messages_mcq_with_images 的区别：
    - 用 format_memory_context_with_caption 生成含 caption 的文本（append 版不显示 caption）
    - 用 _insert_images_inline 把图片插入到对应 caption 行后面，模型看到 caption 即见原图
    - 不用 MEMORY_IMAGES_PROMPT（图片已在对应位置，不需额外引导语）

    memory_images: [{"mem_id":..., "image_id":..., "base64":...}, ...]（已压缩、已限 K）。
    """
    # 1. 生成含 caption 的文本上下文
    memory_context_text = format_memory_context_with_caption(context)

    # 2. 内联图片到 caption 位置（返回 text+image_url 交替的 blocks）
    inline_blocks = _insert_images_inline(memory_context_text, memory_images)

    # 3. TEXT_MSG_PROMPT 拆成前缀 + inline blocks + 后缀
    # TEXT_MSG_PROMPT = "\nThe retrieved memory contents are as follows:\n\n{memory_context}\n"
    text_prefix = "\nThe retrieved memory contents are as follows:\n\n"
    text_suffix = "\n"

    # MCQ 选项注入
    options_text = "\n".join(
        f"{k}. {options_dict[k]}"
        for k in sorted(options_dict.keys()) if k in ("A", "B", "C", "D")
    )
    mcq_constraint = MCQ_OPTIONS_PROMPT.format(options_text=options_text)

    speaker_a = f"user ({character})" if character else "user"
    speaker_b = "assistant"
    query_prompt = DIALOGUE_AGENT_PROMPT.format(
        observation=query,
        speaker_a=speaker_a,
        speaker_b=speaker_b,
        format_constraint=mcq_constraint,
    )

    user_content = [{"type": "text", "text": text_prefix}]
    user_content.extend(inline_blocks)
    user_content.append({"type": "text", "text": text_suffix + query_prompt})

    return [
        {"role": "system", "content": MCQ_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def score_mcq(prediction: str, gt_letter: str) -> float:
    """MCQ 评分：从预测文本提取首字母，和 GT 字母比较。

    gt_letter: 'A'/'B'/'C'/'D'
    模型可能回答 "A" / "A." / "The answer is A" / "A. White background" 等。
    """
    import re
    gt_letter = gt_letter.strip().upper()
    pred = prediction.strip()

    # 1. 精确匹配单个字母
    if pred.upper() in ("A", "B", "C", "D"):
        return 1.0 if pred.upper() == gt_letter else 0.0

    # 2. 找第一个独立的 A/B/C/D（词边界）
    m = re.search(r'\b([ABCD])\b', pred, re.IGNORECASE)
    if m:
        return 1.0 if m.group(1).upper() == gt_letter else 0.0

    # 3. 首字符
    if pred and pred[0].upper() in ("A", "B", "C", "D"):
        return 1.0 if pred[0].upper() == gt_letter else 0.0

    return 0.0
