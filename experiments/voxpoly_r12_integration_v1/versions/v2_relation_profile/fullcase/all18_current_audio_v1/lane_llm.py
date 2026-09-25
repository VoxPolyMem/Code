"""Provider-lane LLM wrapper for reproducible, rate-limited evaluation.

The normal client keeps its existing model and retry behaviour.  When an
AIGC evaluation worker is assigned ``AIGC_EVAL_KEY_INDEX``, this wrapper pins
that worker to one configured key so two independent workers do not rotate the
same small RPM pool.  Credentials never enter output files or process args.
"""

from __future__ import annotations

import os
from typing import Any

from utils.llm_client import call_llm as _call_llm


def call_llm(messages: list[dict[str, Any]], **kwargs: Any) -> str:
    index_text = os.environ.get("AIGC_EVAL_KEY_INDEX")
    if index_text is None:
        return _call_llm(messages, **kwargs)
    if os.environ.get("FORCE_VELEN_MODEL"):
        raise RuntimeError("AIGC key pin cannot be combined with a Velen lane")
    try:
        index = int(index_text)
    except ValueError as exc:
        raise RuntimeError("AIGC_EVAL_KEY_INDEX must be an integer") from exc
    from config import LLM_API_KEYS

    keys = list(LLM_API_KEYS)
    if not 0 <= index < len(keys):
        raise RuntimeError(f"AIGC key index {index} is unavailable")
    if kwargs.get("api_key") is not None:
        raise RuntimeError("lane wrapper received a second api_key")
    kwargs["api_key"] = keys[index]
    return _call_llm(messages, **kwargs)
