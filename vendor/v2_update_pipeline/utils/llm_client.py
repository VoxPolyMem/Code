"""OpenAI-compatible LLM client with an optional private-provider adapter.

Credentials and provider endpoints are supplied through environment
variables. Retry, model-pinning, usage accounting, and response semantics
match the frozen experiment client.
"""
import time
import json
import os
import re
import random
import requests

from openai import OpenAI

import config

_clients = {}  # model -> OpenAI client（都同一 base_url/key，可复用）

# 进程级：记住上次成功的 model+key 索引，避免每次都从默认模型撞 429
_last_good = {"model_idx": 0, "key_idx": 0}

# 进程级 token 用量统计（成本计算用）。每次成功调用累加 prompt/completion tokens。
_usage_stats = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0, "by_model": {}}

# 定价（¥/1M token，输入/输出）。model 名前缀匹配。
_PRICE_PER_M = [
    ("gpt-4.1-mini", (2.93, 11.73)),
    ("gpt-4o-mini", (1.10, 4.40)),
    ("gpt-4.1", (14.66, 58.64)),
    ("gpt-4o", (18.33, 73.33)),
    ("gpt-5", (7.33, 61.05)),
]


def _price_per_m(model: str):
    m = (model or "").lower()
    for prefix, price in _PRICE_PER_M:
        if prefix in m:
            return price
    return (0.0, 0.0)


def _accumulate_usage(model: str, prompt_tokens: int, completion_tokens: int):
    """累计 token 用量（不改变调用返回值，纯 side-channel 统计）。"""
    pt = int(prompt_tokens or 0)
    ct = int(completion_tokens or 0)
    _usage_stats["prompt_tokens"] += pt
    _usage_stats["completion_tokens"] += ct
    _usage_stats["calls"] += 1
    bucket = _usage_stats["by_model"].setdefault(model or "unknown", {"prompt": 0, "completion": 0, "calls": 0})
    bucket["prompt"] += pt
    bucket["completion"] += ct
    bucket["calls"] += 1


def get_usage_stats() -> dict:
    """返回 token 用量统计 + 估算成本（¥）。"""
    total_cost = 0.0
    by_model = []
    for model, b in _usage_stats["by_model"].items():
        pin, pout = _price_per_m(model)
        cost = b["prompt"] / 1e6 * pin + b["completion"] / 1e6 * pout
        total_cost += cost
        by_model.append({
            "model": model, "calls": b["calls"],
            "prompt_tokens": b["prompt"], "completion_tokens": b["completion"],
            "cost_rmb": round(cost, 4),
        })
    return {
        "calls": _usage_stats["calls"],
        "prompt_tokens": _usage_stats["prompt_tokens"],
        "completion_tokens": _usage_stats["completion_tokens"],
        "total_cost_rmb": round(total_cost, 4),
        "by_model": by_model,
    }


def print_usage_summary():
    """打印 token 用量 + 成本汇总（eval 结束时调用）。"""
    s = get_usage_stats()
    if s["calls"] == 0:
        return
    print(f"\n[LLM usage] calls={s['calls']} prompt={s['prompt_tokens']} "
          f"completion={s['completion_tokens']} total_tokens={s['prompt_tokens'] + s['completion_tokens']}", flush=True)
    for b in s["by_model"]:
        print(f"  - {b['model']}: {b['calls']} calls, "
              f"{b['prompt_tokens']}+{b['completion_tokens']} tokens, ¥{b['cost_rmb']}", flush=True)
    print(f"  TOTAL cost ≈ ¥{s['total_cost_rmb']}", flush=True)



def _log_velen_use(model, note=""):
    """每次 velen 调用落一条日志 (用户指令 2026-08-29: 用 velen 必须有记录)。"""
    try:
        import datetime as _dt
        _d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
        os.makedirs(_d, exist_ok=True)
        line = f"{_dt.datetime.now().isoformat(timespec='seconds')} VELEN-CALL model={model} {note}\n"
        with open(os.path.join(_d, "velen_usage.log"), "a") as f:
            f.write(line)
    except Exception:
        pass

# velen 接口配置
_VELEN_URL = os.environ.get("VELEN_URL", "")
_VELEN_DEFAULTS = {
    "tenant": os.environ.get("VELEN_TENANT", ""),
    "velenOfflineChatFlag": False,
    "scene": "OfflineDianXiao",
    "stream": False,
}


def _call_velen(messages: list[dict], model: str = None,
                temperature: float = 0, max_tokens: int = None,
                response_format: dict = None) -> str:
    """Call the optional private provider using its response envelope."""
    if not _VELEN_URL:
        raise RuntimeError("VELEN_URL is required when FORCE_VELEN_MODEL is set")
    model = model or getattr(config, "LLM_MODEL", "gpt-4o-2024-11-20")
    data = {
        "model": model,
        "temperature": temperature,
        "messages": messages,
        **_VELEN_DEFAULTS,
    }
    if max_tokens:
        data["max_tokens"] = max_tokens
    if response_format:
        data["response_format"] = response_format

    resp = requests.post(_VELEN_URL,
                         headers={"Content-Type": "application/json"},
                         json=data, timeout=180)
    resp.raise_for_status()
    r = resp.json()
    if r.get("code") != 200:
        raise RuntimeError(f"velen error: code={r.get('code')}, msg={r.get('msg')}")
    content = r["data"]["choices"][0]["message"]["content"]
    if not content or not content.strip():
        raise RuntimeError("velen returned empty content")
    _usage = (r.get("data") or {}).get("usage") or {}
    _accumulate_usage(model, _usage.get("prompt_tokens"), _usage.get("completion_tokens"))
    return content


def get_client(api_key: str = None):
    """获取 OpenAI client，支持多 key（每个 key 一个 client）。timeout=60s 防止 API hang。"""
    api_key = api_key or config.LLM_API_KEY
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY or LLM_API_KEYS before a live run")
    if api_key not in _clients:
        _clients[api_key] = OpenAI(api_key=api_key, base_url=config.LLM_BASE_URL, timeout=60)
    return _clients[api_key]


def call_llm(messages: list[dict],
             model: str = None,
             temperature: float = None,
             max_tokens: int = None,
             response_format: dict = None,
             api_key: str = None) -> str:
    """
    调 LLM，返回文本。
    - 429/限流：轮换 model + api_key 重试，超过 4h 直接 raise（不跳过 turn）
    - 其他错误：重试 LLM_MAX_RETRIES 次
    - api_key: 指定固定 key（不轮换到其他 key，429 时只换 model）。用于按数据集分配 key 避免 RPM 限流。
    - 环境变量 FORCE_VELEN_MODEL: 设定后所有调用强制走 velen 接口 + 指定 model，
      不走 OpenAI（用于并行测试 gpt-4o-mini 等，不消耗 OpenAI Credit）。
    """
    # 环境变量强制走 velen（用于测试其他 model 如 gpt-4o-mini）
    import os as _os
    _force_velen = _os.environ.get("FORCE_VELEN_MODEL")
    if _force_velen:
        _v_model = _force_velen
        _v_temp = temperature if temperature is not None else config.LLM_TEMPERATURE
        for _attempt in range(1, config.LLM_MAX_RETRIES + 1):
            try:
                return _call_velen(messages, model=_v_model, temperature=_v_temp,
                                   max_tokens=max_tokens, response_format=response_format)
            except Exception as e:
                print(f"    [velen retry {_attempt}/{config.LLM_MAX_RETRIES}]: {str(e)[:80]}", flush=True)
                time.sleep(config.LLM_RETRY_DELAY)
        raise RuntimeError(f"velen call failed after {config.LLM_MAX_RETRIES} retries: model={_v_model}")

    # 模型列表
    models = list(config.LLM_MODELS) if hasattr(config, "LLM_MODELS") else [config.LLM_MODEL]
    if model:
        models = [model] + [m for m in config.LLM_MODELS if m != model]
    # 2026-08-29 测评矩阵: EVAL_PIN_MODEL 钉死单模型 (不轮换; 429 时仅换 key)
    _pin = _os.environ.get("EVAL_PIN_MODEL")
    if _pin:
        models = [_pin]

    temperature = temperature if temperature is not None else config.LLM_TEMPERATURE
    max_tokens = max_tokens or config.LLM_MAX_TOKENS
    client = get_client(api_key)

    rate_limit_start = None  # 首次 429 的时间
    rate_limit_count = 0     # 连续429次数（用于退避，避免thundering herd）
    rate_timeout = getattr(config, "LLM_RATE_LIMIT_TIMEOUT", 4 * 3600)
    # 从上次成功的 model+key 开始，不每次都从默认撞 429
    model_idx = _last_good["model_idx"]
    other_attempts = 0
    empty_rounds = 0    # 连续空回复轮换轮数（>=3 则 raise，避免死循环）

    # api key 列表：传了 api_key 就只用它（不跨数据集抢 key），否则用全部 key 轮换
    if api_key:
        api_keys = [api_key]
    else:
        api_keys = list(config.LLM_API_KEYS) if hasattr(config, "LLM_API_KEYS") else [config.LLM_API_KEY]
    key_idx = _last_good["key_idx"]

    while True:
        cur_model = models[model_idx % len(models)]
        cur_key = api_keys[key_idx % len(api_keys)]
        client = get_client(cur_key)
        try:
            kwargs = dict(model=cur_model, messages=messages, stream=False,
                          temperature=temperature, max_tokens=max_tokens)
            if response_format:
                kwargs["response_format"] = response_format
            result = client.chat.completions.create(**kwargs)
            choice = result.choices[0]
            # aigc gemini: 响应被 max_tokens 截断时返回 message=None (finish_reason=length)
            if choice.message is None:
                raise RuntimeError(
                    f"aigc returned message=None (finish={choice.finish_reason}) "
                    f"— response truncated, max_tokens={max_tokens} too small?")
            content = choice.message.content
            # 空回复视为失败，重试（429 不代表当轮结束，一直等到非空回复）
            if not content or not content.strip():
                other_attempts += 1
                if other_attempts > config.LLM_MAX_RETRIES:
                    # 空回复重试次数用完，换 model+key 继续等；连续 3 轮仍空则 raise
                    other_attempts = 0
                    empty_rounds += 1
                    if empty_rounds >= 3:
                        raise RuntimeError(
                            f"aigc empty response after {empty_rounds} model/key switches "
                            f"(deterministic empty? last model={cur_model})")
                    model_idx += 1
                    key_idx += 1
                    print(f"    [empty response] switch model, wait {config.LLM_RETRY_DELAY}s")
                    time.sleep(config.LLM_RETRY_DELAY)
                    continue
                print(f"    [empty response, retry {other_attempts}/{config.LLM_MAX_RETRIES} "
                      f"model={cur_model}, wait {config.LLM_RETRY_DELAY}s]")
                time.sleep(config.LLM_RETRY_DELAY)
                continue
            # 成功！记住当前 model+key，下次直接从它开始
            _last_good["model_idx"] = model_idx % len(models)
            _last_good["key_idx"] = key_idx % len(api_keys)
            _u = getattr(result, "usage", None)
            _accumulate_usage(cur_model,
                              getattr(_u, "prompt_tokens", 0),
                              getattr(_u, "completion_tokens", 0))
            return content
        except Exception as e:
            err = str(e)
            # 402=Credit耗尽 也要轮换（某个 key+model 组合耗尽，换下一个）
            is_rate = ("429" in err or "rate" in err.lower() or "quota" in err.lower()
                       or "402" in err or "credit" in err.lower())
            is_timeout = "timeout" in err.lower() or "timed out" in err.lower() or "apitimeout" in err.lower()

            if is_rate or is_timeout:
                # 429/RPM/timeout: 不 fallback velen（用户要求优先 openai 接口、velen 花钱）
                # 直接走下方 rate_limit 轮换 model+key + 退避重试 aigc，连续 16 轮仍失败则 raise
                # 2026-08-29 用户指令 v2: aigc 连续限流 >= 1h 才 velen 兜底
                # (尽量全走 aigc; 用了 velen 必须落日志)。
                # BUGFIX: rate_limit_start 首个 429 时还是 None (赋值在下方),
                # 不判 None 会 float-NoneType TypeError 把检索打成机械零 (v21 教训)。
                # NO_VELEN_FALLBACK=1: 永不 velen 兜底 (2026-08-30 用户: 都走 aigc)
                if (not _os.environ.get("NO_VELEN_FALLBACK")
                        and rate_limit_start is not None
                        and time.time() - rate_limit_start >= 3600):
                    print(f"    [VELEN-USED] aigc 429 persisted "
                          f"{(time.time()-rate_limit_start)/60:.0f}min >= 1h "
                          f"-> velen (same model {cur_model})", flush=True)
                    _log_velen_use(cur_model, "reason=aigc_429_ge_1h")
                    for _va in range(1, config.LLM_MAX_RETRIES + 1):
                        try:
                            return _call_velen(messages, model=cur_model,
                                               temperature=temperature,
                                               max_tokens=max_tokens,
                                               response_format=response_format)
                        except Exception as _ve:
                            print(f"    [velen retry {_va}/{config.LLM_MAX_RETRIES}]: "
                                  f"{_ve}", flush=True)
                            time.sleep(config.LLM_RETRY_DELAY)
                    raise RuntimeError(
                        f"aigc 429 exhausted AND velen failed "
                        f"(model={cur_model}). Last aigc error: {e}") from e

                if rate_limit_start is None:
                    rate_limit_start = time.time()
                elapsed = time.time() - rate_limit_start
                # 超过 4h 直接 raise（不跳过 turn）
                if elapsed >= rate_timeout:
                    raise RuntimeError(
                        f"Rate limit/timeout persists {elapsed/3600:.1f}h (>={rate_timeout/3600:.0f}h), "
                        f"aborting (no skip). Last error: {e}"
                    ) from e
                # 轮换 model + api_key
                model_idx += 1
                key_idx += 1
                rate_limit_count += 1
                next_model = models[model_idx % len(models)]
                next_key = api_keys[key_idx % len(api_keys)]
                # 增大 jitter + 连续重试退避，避免 thundering herd
                base_wait = 10 + random.randint(0, 50)   # 10-60s 基础
                backoff = min(rate_limit_count * 3, 30)  # 每次连续重试多等3s，上限+30s
                wait = min(base_wait + backoff, 120)     # 总上限 120s
                err_tag = "429" if is_rate else "timeout"
                print(f"    [{err_tag} retry, elapsed {elapsed/60:.1f}min #{rate_limit_count}] "
                      f"switch {cur_model}/{cur_key[:8]}.. -> {next_model}/{next_key[:8]}.., wait {wait}s")
                time.sleep(wait)
            else:
                # 非限流错误：重试
                other_attempts += 1
                if other_attempts > config.LLM_MAX_RETRIES:
                    raise
                print(f"    [LLM retry {other_attempts}/{config.LLM_MAX_RETRIES} "
                      f"model={cur_model}, wait {config.LLM_RETRY_DELAY}s]: {e}")
                time.sleep(config.LLM_RETRY_DELAY)


def extract_json(text: str):
    """从 LLM 回复鲁棒提取 JSON。

    GPT judge 偶尔返回畸形 JSON（如 reasoning 字段值未加引号），
    此时用 regex fallback 提取 score，避免训练崩溃。
    """
    m = re.search(r'```(?:json)?\s*([\s\S]+)', text)
    if m:
        candidate = m.group(1).strip()
        if '```' in candidate:
            candidate = candidate[:candidate.index('```')].strip()
        r = _try_parse(candidate)
        if r is not None:
            return r
        # truncation recovery on the fence-stripped candidate. gemini often
        # cuts mid-value under 429/timeout (e.g. {"addressee": "group with no
        # closing quote/brace). The raw-based suffix loop below can't see
        # past the leading ```json fence, so recover here first.
        for suffix in ['"}', '"]}', '"]', '}']:
            try:
                return json.loads(candidate + suffix)
            except json.JSONDecodeError:
                continue
    r = _try_parse(text)
    if r is not None:
        return r
    raw = text.strip()
    for ch in ['{', '[']:
        idx = raw.find(ch)
        if idx == -1:
            continue
        r = _try_parse(raw[idx:])
        if r is not None:
            return r
    # 截断 JSON 容错：LLM 输出被 max_tokens 截断（如 reason 字段太长没闭合），
    # 尝试补全末尾的引号+闭合括号后重新解析
    for suffix in ['"}', '"}', '"]}', '"]', '}']:
        try:
            return json.loads(raw + suffix)
        except json.JSONDecodeError:
            continue

    # Regex fallback: 从畸形 JSON 中提取 score（GPT 偶尔不加引号到 reasoning 值）
    score_match = re.search(r'"score"\s*:\s*([0-9.]+)', text)
    if score_match:
        return {"score": float(score_match.group(1))}
    # Regex fallback: 从畸形/截断 JSON 中提取 sufficient（orchestrator sufficiency check）
    suff_match = re.search(r'"sufficient"\s*:\s*(true|false)', text, re.IGNORECASE)
    if suff_match:
        return {"sufficient": suff_match.group(1).lower() == "true"}
    raise ValueError(f"无法提取 JSON:\n{text[:500]}")


def _try_parse(s: str):
    s = s.strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    cleaned = re.sub(r',\s*([\}\]])', r'\1', s)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None
