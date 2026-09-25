"""
vLLM Client — 调本地 vllm 部署的 Qwen3-Omni（OpenAI 兼容接口）。

用途：检索期 LLM 后端（route / sufficiency / rewrite）。
答题仍走 utils.llm_client.call_llm（美团 gpt-4.1/4o 轮换）。

与 call_llm 对齐的鲁棒性：
- 连接失败 / 空回复：重试 LLM_MAX_RETRIES 次
- 返回纯文本（route/sufficiency 的 JSON 由调用方 extract_json 解析）
"""
import time
import config

# 复用全局 OpenAI client（按 base_url 缓存，可复用 HTTP 连接）
from openai import OpenAI

# base_url -> OpenAI client（支持检索/答题走不同 vllm 实例：检索8901, 答题8903）
_clients = {}


def _get_client(base_url: str = None):
    """获取 OpenAI client，按 base_url 缓存。
    base_url=None → 用 VLLM_BASE_URL_OVERRIDE 或 config.VLLM_BASE_URL（检索期默认 8901）
    base_url="http://..." → 指定实例（答题期走 8903 答题专用 vllm，与检索隔离）
    """
    import os
    if base_url is None:
        base_url = os.environ.get("VLLM_BASE_URL_OVERRIDE") or getattr(config, "VLLM_BASE_URL", "http://localhost:8901/v1")
    if base_url not in _clients:
        _clients[base_url] = OpenAI(
            api_key="EMPTY",
            base_url=base_url,
            timeout=180,  # 3分钟，避免偶发超时导致case中断（看图请求重，排队久）
        )
    return _clients[base_url]


def _content_text(c):
    """把 message content 转成纯文本用于 token 估算。
    支持 str 和 OpenAI 多模态 list 格式（image_url + text）。
    image 块用占位符 [image] 估算，不展开 base64（否则估算爆炸）。
    """
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts = []
        for b in c:
            if isinstance(b, dict):
                if b.get("type") == "text":
                    parts.append(b.get("text", ""))
                elif b.get("type") == "image_url":
                    parts.append("[image]")
        return " ".join(parts)
    return str(c) if c else ""


def call_vllm(messages: list[dict],
              temperature: float = None,
              max_tokens: int = None,
              response_format: dict = None,
              base_url: str = None,
              model: str = None) -> str:
    """
    调本地 vllm，返回文本。

    与 call_llm 接口对齐（messages + 返回 str），便于检索器无感切换。
    - 连接失败 / 空回复：重试 LLM_MAX_RETRIES 次
    - 不做 429 模型轮换（本地部署无 RPM 限制）
    - response_format: 传 {"type":"json_object"} 强制 JSON 输出（route/sufficiency 用）
    - base_url: 指定 vllm 实例（None=检索默认 8901；答题传 8903 走独立实例）
    - model: 指定模型名（None=用 VLLM_MODEL_OVERRIDE 环境变量或 config.VLLM_MODEL）
    - messages 的 content 支持 str 和 OpenAI 多模态 list 格式
      （[{"type":"image_url","image_url":{"url":"data:image/jpeg;base64,..."}},
        {"type":"text","text":"..."}]），vLLM Qwen3-VL 自动解析图片。
      截断逻辑只裁 text 块，不动 image 块。
    """
    # 支持环境变量覆盖（用于多实例并行）
    import os
    model = model or os.environ.get("VLLM_MODEL_OVERRIDE") or getattr(config, "VLLM_MODEL", "Qwen3-Omni-30B-A3B-Instruct")
    temperature = temperature if temperature is not None else config.LLM_TEMPERATURE
    max_tokens = max_tokens or getattr(config, "VLLM_MAX_TOKENS", config.LLM_MAX_TOKENS)

    # 估算 prompt 长度（rough estimate: 3 chars ≈ 1 token）
    # content 可能是 str 或多模态 list，用 _content_text 统一转文本估算
    prompt_text = "\n".join(_content_text(m.get("content", "")) for m in messages)
    estimated_input_tokens = len(prompt_text) // 3  # 保守估计

    # 根据模型 max_model_len 调整 max_tokens
    # 8901=Qwen3-VL-8B 启动 --max-model-len 16384；8902=30B 为 32768。
    # 这里用实际最小值 16384，避免 8B 实例 max_tokens 超限触发 400
    # （此前硬编码 32768 导致 Card 等长 context 题 sufficiency 报
    #  "max_tokens is too large: 4096 > 16384 - 13842"）。
    model_max_len = int(os.environ.get("VLLM_MAX_MODEL_LEN", "16384"))
    max_allowed_output = model_max_len - estimated_input_tokens - 512  # 留 512 buffer

    if max_allowed_output < 256:
        print(f"    [warn] prompt too long ({estimated_input_tokens} est. tokens), truncating...")
        # 截断最后一条消息：只裁 text 部分，image_url 块原样保留
        if messages:
            content = messages[-1].get("content", "")
            max_chars = (model_max_len - 2048) * 3  # 保留 2048 输出空间
            if isinstance(content, str):
                messages[-1]["content"] = content[:max_chars] + "\n[truncated]"
            elif isinstance(content, list):
                # 多模态：只截第一个超长 text 块，image_url 块不动
                for b in content:
                    if isinstance(b, dict) and b.get("type") == "text":
                        t = b.get("text", "")
                        if len(t) > max_chars:
                            b["text"] = t[:max_chars] + "\n[truncated]"
                            break
        max_allowed_output = 2048

    max_tokens = min(max_tokens, max_allowed_output)
    max_tokens = max(max_tokens, 256)  # 至少 256

    client = _get_client(base_url)
    last_err = None
    for attempt in range(1, config.LLM_MAX_RETRIES + 1):
        try:
            kwargs = dict(model=model, messages=messages, stream=False,
                          temperature=temperature, max_tokens=max_tokens)
            if response_format:
                kwargs["response_format"] = response_format
            result = client.chat.completions.create(**kwargs)
            content = result.choices[0].message.content
            if content and content.strip():
                return content
            print(f"    [vllm empty response, retry {attempt}/{config.LLM_MAX_RETRIES}] "
                  f"wait {config.LLM_RETRY_DELAY}s", flush=True)
        except Exception as e:
            last_err = e
            print(f"    [vllm retry {attempt}/{config.LLM_MAX_RETRIES}]: {e}", flush=True)
        time.sleep(config.LLM_RETRY_DELAY)

    # 重试用完仍失败：raise（让调用方知道，不静默返回空串污染结果）
    raise RuntimeError(f"vllm call failed after {config.LLM_MAX_RETRIES} retries: {last_err}")


# Thinking 模型输出分隔符：thinking 过程在 之前，最终回复在  之后
_THINK_END = "</think>"


def call_vllm_thinking(messages: list[dict],
                       temperature: float = None,
                       max_tokens: int = None,
                       base_url: str = None,
                       model: str = None) -> tuple[str, str]:
    """调 vLLM Thinking 模型（Qwen3-VL-8B-Thinking），返回 (thinking, response)。

    Thinking 模型输出格式：
      thinking过程...</think>
      最终回复

    - thinking:  推理过程（ 之前）
    - response: 最终回复（ 之后，strip）
    - 如果输出不含 （thinking 被截断），尝试从 content regex 提取 JSON 作 response

    和 call_vllm 的区别：
    - 不传 response_format（Thinking 模型的 JSON 在 thinking 之后，强制 JSON 会破坏 thinking）
    - 默认 max_tokens 更大（thinking 过程消耗大量 token）
    - 支持显式传 base_url + model（Thinking 实例与 Instruct 实例隔离）
    """
    import re
    # Thinking 模型需要更大的 max_tokens（thinking 可能占几千 token）
    # 4096 够 200-word thinking (~300 token) + JSON response (~200 token)
    if max_tokens is None:
        max_tokens = 4096
    # response_format=None：Thinking 模型不强制 JSON
    content = call_vllm(messages, temperature=temperature,
                        max_tokens=max_tokens, response_format=None,
                        base_url=base_url, model=model)
    # 分割 thinking 和最终回复
    idx = content.find(_THINK_END)
    if idx >= 0:
        thinking = content[:idx].strip()
        response = content[idx + len(_THINK_END):].strip()
    else:
        # thinking 被截断（max_tokens 不够，没到 ），尝试 regex 提取 JSON
        thinking = content.strip()
        m = re.search(r'\{[^{}]*\}', content, re.DOTALL)
        response = m.group(0) if m else ""
    return thinking, response
