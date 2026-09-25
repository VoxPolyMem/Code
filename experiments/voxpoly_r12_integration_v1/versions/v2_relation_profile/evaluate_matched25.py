#!/usr/bin/env python3
"""Default-off matched gpt-4.1-mini evaluation for AudioMem v2.

The control and v2 arms reuse one frozen R12 route/context source and receive
the same answer prompt, judge prompt, model, Top30 budget, and audio-derived
runtime character.  Retrieval is completely frozen before canonical answers
or gold evidence are opened.  Exact-message caching prevents repeated calls,
and each completed QA is checkpointed atomically.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


HERE = Path(__file__).resolve().parent
WORK_ROOT = HERE.parents[3]
V2_ROOT = Path(os.environ.get(
    "V2_PIPELINE_ROOT",
    WORK_ROOT / "vendor/v2_update_pipeline",
))
for candidate in (HERE, WORK_ROOT, V2_ROOT):
    while str(candidate) in sys.path:
        sys.path.remove(str(candidate))
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(WORK_ROOT))
sys.path.insert(2, str(V2_ROOT))

from audit_retrieval import AuditContractError, freeze_retrieval, read_object  # noqa: E402


SCHEMA_VERSION = "voxpoly-audio-v2-matched25.v1"
MODEL = "gpt-4.1-mini"
TOP_K = 30


class EvaluationContractError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as output:
            json.dump(payload, output, ensure_ascii=False, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class ExactMessageCache:
    def __init__(
        self,
        path: Path,
        caller: Callable[..., str],
        *,
        max_attempts: int,
        retry_delay_seconds: float,
    ) -> None:
        self.path = path
        self.caller = caller
        self.max_attempts = max_attempts
        self.retry_delay_seconds = retry_delay_seconds
        if path.exists():
            document = read_object(path, "exact-message cache")
            if document.get("schema_version") != "exact-message-cache.v1":
                raise EvaluationContractError("unsupported call cache")
            self.rows = document.get("rows")
            if not isinstance(self.rows, dict):
                raise EvaluationContractError("call cache rows must be an object")
        else:
            self.rows: dict[str, Any] = {}

    @staticmethod
    def _contract(messages: list[dict[str, Any]]) -> tuple[dict[str, Any], str]:
        contract = {"model": MODEL, "messages": messages}
        return contract, sha256_json(contract)

    def invalidate(self, messages: list[dict[str, Any]]) -> None:
        _, key = self._contract(messages)
        if self.rows.pop(key, None) is not None:
            write_json_atomic(self.path, {
                "schema_version": "exact-message-cache.v1", "rows": self.rows,
            })

    def call(self, messages: list[dict[str, Any]]) -> str:
        contract, key = self._contract(messages)
        if key in self.rows:
            return str(self.rows[key]["response"])
        error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                response = str(self.caller(messages, model=MODEL)).strip()
                if not response:
                    raise EvaluationContractError("LLM returned an empty response")
                self.rows[key] = {
                    "contract_sha256": key,
                    "contract": contract,
                    "response": response,
                    "attempts_for_this_call": attempt,
                }
                write_json_atomic(self.path, {
                    "schema_version": "exact-message-cache.v1", "rows": self.rows,
                })
                return response
            except Exception as exc:
                error = exc
                if attempt < self.max_attempts:
                    time.sleep(self.retry_delay_seconds * attempt)
        raise RuntimeError(
            f"LLM call failed after {self.max_attempts} bounded attempts: {error}"
        ) from error


def _answer_context(arm: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in arm.get("context") or []:
        rows.append({
            **row,
            "mem_id": row["node_id"],
            "layer": "raw",
            "text": f"[{row.get('speaker', 'unknown')}; {row.get('date', '')}]: {row.get('text', '')}",
        })
    return rows


def _canonical_qa(config: Mapping[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    from adapters.voxpoly import qa_pairs

    result = {}
    for case_config in config.get("cases") or []:
        case_id = str(case_config["case_id"])
        case = read_object(Path(case_config["canonical_case"]).resolve(strict=True), "canonical case")
        for qa in qa_pairs(case):
            qa_id = str(qa.get("qa_id") or "")
            if qa_id:
                result[(case_id, qa_id)] = qa
    return result


def _judge(
    cache: ExactMessageCache,
    extract_json: Callable[[str], dict[str, Any]],
    messages: list[dict[str, Any]],
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, cache.max_attempts + 1):
        try:
            value = extract_json(cache.call(messages))
            score = float(value.get("score"))
            if not 0.0 <= score <= 1.0:
                raise EvaluationContractError("judge score is outside [0,1]")
            value["score"] = score
            return value
        except RuntimeError:
            raise
        except Exception as exc:
            last_error = exc
            cache.invalidate(messages)
            if attempt < cache.max_attempts:
                time.sleep(cache.retry_delay_seconds * attempt)
    raise EvaluationContractError(
        f"judge parse failed after bounded attempts: {last_error}"
    ) from last_error


def run(
    *,
    config_path: Path,
    out_dir: Path,
    max_attempts: int,
    retry_delay_seconds: float,
    call_llm: Callable[..., str],
    extract_json: Callable[[str], dict[str, Any]],
) -> dict[str, Any]:
    from adapters.voxpoly import gold_evidence
    from eval.bench_prompts import answer_messages, judge_messages

    config = read_object(config_path, "matched25 config")
    if int(config.get("top_k", -1)) != TOP_K:
        raise EvaluationContractError("matched evaluation is frozen to Top30")

    # No answers or canonical gold have been opened at this point.
    frozen = freeze_retrieval(config, config_path)
    if int(frozen.get("qa_count", -1)) != 25:
        raise EvaluationContractError("matched panel must contain exactly 25 QAs")
    out_dir.mkdir(parents=True, exist_ok=True)
    frozen_path = out_dir / "frozen_contexts_before_gold.json"
    write_json_atomic(frozen_path, frozen)

    prompt_module = V2_ROOT / "eval" / "bench_prompts.py"
    run_config = {
        "config_id": "voxpoly_audio_relation_profile_v2_matched25_gpt41mini_top30",
        "model": MODEL,
        "top_k": TOP_K,
        "control": "saved_current_r12_context",
        "treatment": "audio_relation_profile_v2",
        "route_source": "one_previously_frozen_r12_route_per_qa_shared_by_both_arms",
        "new_router_calls": 0,
        "answer_prompt_sha256": sha256_file(prompt_module),
        "same_answer_prompt": True,
        "same_judge_prompt": True,
        "same_audio_derived_character": True,
        "uses_benchmark_label_for_retrieval": False,
        "uses_answer_for_retrieval": False,
        "uses_gold_for_retrieval": False,
        "retrieval_freeze_sha256": frozen["retrieval_freeze_sha256"],
    }
    partial_path = out_dir / "matched_pairs.partial.json"
    final_path = out_dir / "matched_pairs.json"
    if final_path.exists():
        final = read_object(final_path, "complete matched result")
        if final.get("run_config") != run_config or final.get("status") != "complete":
            raise EvaluationContractError("existing final output has a different contract")
        return final
    if partial_path.exists():
        document = read_object(partial_path, "partial matched result")
        if document.get("run_config") != run_config or document.get("status") != "partial":
            raise EvaluationContractError("existing partial output has a different contract")
    else:
        document = {
            "schema_version": SCHEMA_VERSION,
            "status": "partial",
            "run_config": run_config,
            "pairs": [],
        }
    done = {(row["case_id"], row["qa_id"]) for row in document["pairs"]}
    if len(done) != len(document["pairs"]):
        raise EvaluationContractError("partial output contains duplicate QAs")

    # Canonical QA is opened only after every candidate context was frozen and
    # durably written above.
    canonical = _canonical_qa(config)
    cache = ExactMessageCache(
        out_dir / "llm_call_cache.json",
        call_llm,
        max_attempts=max_attempts,
        retry_delay_seconds=retry_delay_seconds,
    )
    for position, row in enumerate(frozen["rows"], 1):
        key = (row["case_id"], row["qa_id"])
        if key in done:
            continue
        if key not in canonical:
            raise EvaluationContractError(f"canonical QA missing: {key}")
        qa = canonical[key]
        reference = str(qa.get("answer") or "")
        identity = row.get("query_identity") or {}
        character = (
            str(identity.get("profile_display"))
            if identity.get("confidence") == "high" and identity.get("profile_display")
            else "user"
        )
        case_dates = [
            str(item.get("date") or "")
            for item in row["arms"]["current_r12"]["context"]
            if item.get("date")
        ]
        last_date = max(case_dates) if case_dates else ""

        arms = {}
        for arm_name in ("current_r12", "audio_relation_profile_v2"):
            arm = row["arms"][arm_name]
            answer_prompt = answer_messages(
                row["question"], _answer_context(arm), character=character, last_date=last_date
            )
            prediction = cache.call(answer_prompt)
            judge_prompt = judge_messages(row["question"], reference, prediction)
            judged = _judge(cache, extract_json, judge_prompt)
            arms[arm_name] = {
                "prediction": prediction,
                "score": judged["score"],
                "judge": judged,
                "context_node_ids": arm["context_node_ids"],
                "retrieved_refer_ids": arm["retrieved_refer_ids"],
                "context_sha256": sha256_json(arm["context"]),
            }
        document["pairs"].append({
            "case_id": row["case_id"],
            "qa_id": row["qa_id"],
            "question": row["question"],
            "reference_answer": reference,
            "gold_evidence": gold_evidence(qa),
            "runtime_character": character,
            "query_identity_confidence": identity.get("confidence", "unresolved"),
            "arms": arms,
        })
        write_json_atomic(partial_path, document)
        print(
            f"[{position}/25] {row['case_id']}/{row['qa_id']} "
            f"r12={arms['current_r12']['score']:.2f} "
            f"v2={arms['audio_relation_profile_v2']['score']:.2f}",
            flush=True,
        )
    if len(document["pairs"]) != 25:
        raise EvaluationContractError("run ended without all 25 QA pairs")
    document["status"] = "complete"
    document["summary"] = {
        arm: sum(row["arms"][arm]["score"] for row in document["pairs"]) / 25
        for arm in ("current_r12", "audio_relation_profile_v2")
    }
    write_json_atomic(final_path, document)
    return document


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enable-paid-evaluation", action="store_true")
    parser.add_argument("--config", type=Path, default=HERE / "configs" / "matched25.json")
    parser.add_argument("--out-dir", type=Path, default=HERE / "artifacts" / "matched25_eval")
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--retry-delay-seconds", type=float, default=30.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.enable_paid_evaluation:
        print("paid matched evaluation is default-off; pass --enable-paid-evaluation", file=sys.stderr)
        return 2
    if not 1 <= args.max_attempts <= 8:
        print("--max-attempts must be in [1,8]", file=sys.stderr)
        return 2
    if not 0 <= args.retry_delay_seconds <= 300:
        print("--retry-delay-seconds must be in [0,300]", file=sys.stderr)
        return 2
    from utils.llm_client import call_llm, extract_json, print_usage_summary

    try:
        result = run(
            config_path=args.config.resolve(strict=True),
            out_dir=args.out_dir.resolve(),
            max_attempts=args.max_attempts,
            retry_delay_seconds=args.retry_delay_seconds,
            call_llm=call_llm,
            extract_json=extract_json,
        )
    except (AuditContractError, EvaluationContractError, OSError, RuntimeError, ValueError) as exc:
        print(f"matched25 failed closed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print_usage_summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
