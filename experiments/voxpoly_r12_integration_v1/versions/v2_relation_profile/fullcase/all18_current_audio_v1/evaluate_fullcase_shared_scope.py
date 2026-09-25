#!/usr/bin/env python3
"""Evaluate a shared planner's optional audio asker scope on one full case.

The planner and its answer/judge prompts are shared with the text and image
benchmarks. Only a high-confidence waveform identity is appended to the
planner input for AudioMem. The planner may choose ``global`` (the frozen v2
relation/profile context is reused verbatim) or ``asker_only`` (raw dense/BM25
is restricted through that profile's grounded ``refer_ids``). No question
type, answer, or gold evidence is read during retrieval freezing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping


HERE = Path(__file__).resolve().parent
V2_EXPERIMENT = HERE.parents[1]
WORK_ROOT = V2_EXPERIMENT.parents[3]
V2_ROOT = Path(os.environ.get(
    "V2_PIPELINE_ROOT",
    WORK_ROOT / "vendor/v2_update_pipeline",
))
for candidate in (V2_EXPERIMENT, WORK_ROOT, V2_ROOT):
    while str(candidate) in sys.path:
        sys.path.remove(str(candidate))
sys.path.insert(0, str(V2_EXPERIMENT))
sys.path.insert(1, str(WORK_ROOT))
sys.path.insert(2, str(V2_ROOT))

from audit_retrieval import (  # noqa: E402
    AuditContractError,
    _context_record,
    freeze_retrieval,
    read_object,
)
from audio_persona_adapter import (  # noqa: E402
    annotate_raw_relation_context,
    has_high_confidence_identity,
    speaker_profile_prior_raw_context,
    speaker_owned_raw_context,
)
from evaluate_matched25 import (  # noqa: E402
    ExactMessageCache,
    EvaluationContractError,
    MODEL,
    TOP_K,
    _answer_context,
    _canonical_qa,
    _judge,
    sha256_file,
    sha256_json,
    write_json_atomic,
)


SCHEMA_VERSION = "voxpoly-audio-shared-scope-fullcase.v2"


def _base_v2_results(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """Read a completed v2 result only after retrieval contexts are frozen."""

    document = read_object(path, "completed audio v2 result")
    if document.get("status") != "complete":
        raise EvaluationContractError("base v2 result is not complete")
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for pair in document.get("pairs") or []:
        key = (str(pair.get("case_id") or ""), str(pair.get("qa_id") or ""))
        arm = (pair.get("arms") or {}).get("audio_relation_profile_v2")
        if not all(key) or not isinstance(arm, dict) or key in rows:
            raise EvaluationContractError("base v2 result is malformed")
        rows[key] = dict(pair)
    return rows


def _build_scope_freeze(
    *,
    config: Mapping[str, Any],
    config_path: Path,
    planner_cache: ExactMessageCache,
    asker_context_strategy: str,
    enable_iterative_refinement: bool,
    iterative_max_rounds: int,
    iterative_round_budgets: str,
) -> dict[str, Any]:
    """Finish all QA-blind routing before canonical QA is opened."""

    from core.iterative_retrieval import retrieve_iteratively
    from core.planner import unified_hybrid_plan
    from evaluation.voxpoly import LayerIndex

    expected_count = len((config.get("cases") or [{}])[0].get("panel_qa_ids") or [])
    base_frozen = freeze_retrieval(config, config_path)
    if not expected_count or int(base_frozen.get("qa_count", -1)) != expected_count:
        raise EvaluationContractError("shared scope evaluation requires complete requested QA coverage")
    cases = config.get("cases") or []
    if len(cases) != 1 or int(config.get("top_k", -1)) != TOP_K:
        raise EvaluationContractError("shared scope evaluation requires Top30 one-case config")
    if asker_context_strategy not in {"speaker_only", "soft_profile_prior"}:
        raise EvaluationContractError("unsupported asker context strategy")
    memory_path = Path(cases[0]["memory"]).resolve(strict=True)
    memory = read_object(memory_path, "audio memory")
    index = LayerIndex(memory_path)
    raw_count = len(index.layers["raw"])
    rows = []
    for position, base in enumerate(base_frozen["rows"], 1):
        question = str(base["question"])
        identity = dict(base.get("query_identity") or {})
        scope = "global"
        route_plan: dict[str, Any] | None = None
        arm = base["arms"]["audio_relation_profile_v2"]
        scope_trace: dict[str, Any] = {
            "scope_source": "no_high_confidence_audio_identity",
            "uses_benchmark_label": False,
            "uses_answer": False,
            "uses_gold": False,
        }
        if has_high_confidence_identity(identity):
            display = str(identity.get("profile_display") or "").strip()
            if not display:
                raise EvaluationContractError("high-confidence identity lacks profile_display")

            def router(messages: list[dict[str, Any]]) -> str:
                return planner_cache.call(messages)

            route_plan = unified_hybrid_plan(
                question, False, router, known_asker=display,
            )
            if route_plan.get("router_parse_ok") is not True:
                raise EvaluationContractError(
                    f"shared planner parse failed closed: {base['case_id']}/{base['qa_id']}"
                )
            scope = str(route_plan.get("speaker_scope") or "global")
            if scope not in {"global", "asker_only"}:
                raise EvaluationContractError("planner emitted an invalid speaker scope")
            if scope == "asker_only":
                rewritten = str(route_plan.get("rewritten_query") or question)
                # The planner's resolution is used to select the grounded
                # speaker scope, rather than replacing the retrieval query.
                # In particular, a name-substitution can erase useful
                # lexical cues in an otherwise well-formed first-person
                # question.  Once the profile projection has made the scope
                # explicit, the original query is the least transformed and
                # modality-shared retrieval signal.
                dense = index.rank("raw", "dense", question, top_k=raw_count)
                bm25 = index.rank("raw", "bm25", question, top_k=raw_count)
                context_builder = (
                    speaker_owned_raw_context
                    if asker_context_strategy == "speaker_only"
                    else speaker_profile_prior_raw_context
                )
                context, trace = context_builder(
                    question=question,
                    query_identity=identity,
                    memory=memory,
                    dense_raw=dense,
                    bm25_raw=bm25,
                    top_k=TOP_K,
                )
                context = annotate_raw_relation_context(memory, context)
                if len(context) != TOP_K:
                    raise EvaluationContractError(
                        f"speaker-owned route underfilled Top30: {base['case_id']}/{base['qa_id']}"
                    )
                if enable_iterative_refinement:
                    # Round 0 is the direct, profile-grounded asker lookup.
                    # Only a later, model-requested action may broaden into
                    # the common hierarchical executor, which preserves a
                    # needed response or delegation by another speaker.
                    initial_plan = dict(route_plan)

                    def execute_round(round_plan: Mapping[str, Any]) -> dict[str, Any]:
                        if round_plan is initial_plan:
                            return {
                                "context": context,
                                "trace": [{"channel": "audio.profile_round0", **trace}],
                                "memory_image_candidates": [],
                                "need_memory_images": False,
                                "selected_collection": None,
                                "expanded_refer_count": 0,
                            }
                        round_context, round_trace, selected = index.execute(
                            dict(round_plan), top_k=TOP_K,
                        )
                        round_context = annotate_raw_relation_context(
                            memory, round_context,
                        )
                        return {
                            "context": round_context,
                            "trace": round_trace,
                            "memory_image_candidates": [],
                            "need_memory_images": False,
                            "selected_collection": (
                                selected.get("node_id") if selected else None
                            ),
                            "expanded_refer_count": len(
                                selected.get("refer_ids", []) if selected else []
                            ),
                        }

                    iterative = retrieve_iteratively(
                        question=question,
                        has_question_image=False,
                        initial_plan=initial_plan,
                        execute=execute_round,
                        call_router=router,
                        top_k=TOP_K,
                        environ={
                            "MPMEM_ITERATIVE_MAX_ROUNDS": str(iterative_max_rounds),
                            "MPMEM_ITERATIVE_ROUND_BUDGETS": iterative_round_budgets,
                        },
                        known_asker=display,
                    )
                    context = annotate_raw_relation_context(memory, iterative["context"])
                    route_plan = dict(iterative["route_plan"])
                    route_plan["speaker_scope"] = "asker_only"
                    route_plan["known_asker"] = display
                    trace = {
                        "round0": trace,
                        "iterative_retrieval": iterative["iterative_retrieval"],
                        "round_execution": iterative["trace"],
                    }
                arm = _context_record(context, trace)
                scope_trace = {
                    "scope_source": "shared_planner_asker_only",
                    "retrieval_query": question,
                    "planner_resolved_query": rewritten,
                    "query_resolution_used_for": "speaker_scope_selection",
                    "asker_context_strategy": asker_context_strategy,
                    "iterative_refinement": enable_iterative_refinement,
                    "uses_benchmark_label": False,
                    "uses_answer": False,
                    "uses_gold": False,
                }
            else:
                scope_trace = {
                    "scope_source": "shared_planner_global_reused_audio_v2",
                    "uses_benchmark_label": False,
                    "uses_answer": False,
                    "uses_gold": False,
                }
        rows.append({
            "case_id": str(base["case_id"]),
            "qa_id": str(base["qa_id"]),
            "question": question,
            "query_identity": identity,
            "speaker_scope": scope,
            "route_plan": route_plan,
            "scope_trace": scope_trace,
            "context": arm["context"],
            "context_node_ids": arm["context_node_ids"],
            "retrieved_refer_ids": arm["retrieved_refer_ids"],
            "base_current_r12_context": base["arms"]["current_r12"]["context"],
        })
        print(
            f"[freeze {position}/{expected_count}] {base['case_id']}/{base['qa_id']} scope={scope}",
            flush=True,
        )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "top_k": TOP_K,
        "model": MODEL,
        "qa_count": len(rows),
        "planner": "unified_hybrid_plan(optional_known_asker)",
        "asker_context_strategy": asker_context_strategy,
        "iterative_refinement": {
            "enabled": enable_iterative_refinement,
            "max_rounds": iterative_max_rounds,
            "round_budgets": iterative_round_budgets,
            "only_after_profile_round0": True,
        },
        "external_llm_calls": len(planner_cache.rows),
        "uses_benchmark_label_for_retrieval": False,
        "uses_answer_for_retrieval": False,
        "uses_gold_for_retrieval": False,
        "rows": rows,
    }
    payload["retrieval_freeze_sha256"] = sha256_json(payload)
    return payload


def _same_context(base_pair: Mapping[str, Any], row: Mapping[str, Any]) -> bool:
    arm = (base_pair.get("arms") or {}).get("audio_relation_profile_v2") or {}
    return (
        list(arm.get("context_node_ids") or []) == list(row.get("context_node_ids") or [])
        and len(row.get("context_node_ids") or []) == TOP_K
    )


def _answer_query(row: Mapping[str, Any], source: str) -> str:
    """Choose a QA-blind question string for the unchanged answer template."""

    original = str(row.get("question") or "")
    if source == "original":
        return original
    if source != "planner_resolved_for_asker_only":
        raise EvaluationContractError(f"unsupported answer query source: {source}")
    if row.get("speaker_scope") != "asker_only":
        return original
    plan = row.get("route_plan") or {}
    resolved = str(plan.get("rewritten_query") or "").strip()
    return resolved or original


def _answer_messages_with_known_asker(
    messages: list[dict[str, Any]], *, known_asker: str
) -> list[dict[str, Any]]:
    """Append one generic identity-resolution sentence to a shared prompt.

    The interface is modality-neutral: any front end may supply a
    high-confidence asker identity.  It neither identifies a benchmark type
    nor constrains the answer to utterances by that person.
    """

    if not known_asker.strip() or len(messages) < 2:
        raise EvaluationContractError("known asker prompt extension lacks an identity")
    user = messages[1]
    content = user.get("content")
    if not isinstance(content, list):
        raise EvaluationContractError("shared answer prompt has an unexpected content shape")
    extension = {
        "type": "text",
        "text": (
            f"Identity context: the person asking this question is {known_asker}. "
            "Resolve I/me/my as this person. Each memory may identify its speaker "
            "and addressee: use those relations to distinguish the asker's own "
            "statement, a request made to the asker, and a statement by someone else "
            "about the asker.\n"
        ),
    }
    copied = [dict(message) for message in messages]
    copied_user = dict(copied[1])
    copied_user["content"] = [content[0], extension, *content[1:]]
    copied[1] = copied_user
    return copied


def _relation_aware_answer_context(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Render raw evidence with speaker/addressee fields already in memory."""

    output = []
    for item in row.get("context") or []:
        speaker = str(item.get("speaker") or item.get("speaker_ref") or "unknown")
        addressee = str(item.get("addressee") or "unknown")
        date = str(item.get("date") or item.get("timestamp") or "")
        output.append({
            **dict(item),
            "mem_id": str(item.get("node_id") or ""),
            "layer": "raw",
            "text": (
                f"[speaker={speaker}; addressee={addressee}; date={date}]: "
                f"{str(item.get('text') or '')}"
            ),
        })
    return output


def run(
    *,
    config_path: Path,
    base_result_path: Path,
    out_dir: Path,
    answer_query_source: str = "original",
    asker_context_strategy: str = "speaker_only",
    include_known_asker_hint: bool = False,
    include_relation_labels: bool = False,
    enable_iterative_refinement: bool = False,
    iterative_max_rounds: int = 3,
    iterative_round_budgets: str = "30,8,5",
    max_attempts: int,
    retry_delay_seconds: float,
    call_llm: Callable[..., str],
    extract_json: Callable[[str], dict[str, Any]],
) -> dict[str, Any]:
    from adapters.voxpoly import gold_evidence
    from eval.bench_prompts import answer_messages, judge_messages

    config = read_object(config_path, "full-case config")
    if answer_query_source not in {"original", "planner_resolved_for_asker_only"}:
        raise EvaluationContractError("unsupported answer query source")
    if asker_context_strategy not in {"speaker_only", "soft_profile_prior"}:
        raise EvaluationContractError("unsupported asker context strategy")
    if not 1 <= iterative_max_rounds <= 5:
        raise EvaluationContractError("iterative max rounds must be in [1, 5]")
    out_dir.mkdir(parents=True, exist_ok=True)
    planner_cache = ExactMessageCache(
        out_dir / "router_call_cache.json",
        call_llm,
        max_attempts=max_attempts,
        retry_delay_seconds=retry_delay_seconds,
    )
    frozen_path = out_dir / "frozen_contexts_before_gold.json"
    if frozen_path.exists():
        frozen = read_object(frozen_path, "shared-scope frozen context")
        if (
            frozen.get("status") != "complete"
            or frozen.get("schema_version") != SCHEMA_VERSION
            or int(frozen.get("qa_count", -1)) != len((config.get("cases") or [{}])[0].get("panel_qa_ids") or [])
        ):
            raise EvaluationContractError("existing shared-scope freeze has a different contract")
    else:
        frozen = _build_scope_freeze(
            config=config,
            config_path=config_path,
            planner_cache=planner_cache,
            asker_context_strategy=asker_context_strategy,
            enable_iterative_refinement=enable_iterative_refinement,
            iterative_max_rounds=iterative_max_rounds,
            iterative_round_budgets=iterative_round_budgets,
        )
        write_json_atomic(frozen_path, frozen)

    prompt_module = V2_ROOT / "eval" / "bench_prompts.py"
    run_config = {
        "schema_version": SCHEMA_VERSION,
        "config_id": (
            f"{str((config.get('cases') or [{}])[0].get('case_id')).lower()}"
            f"_shared_scope_{answer_query_source}_gpt41mini_top30"
        ),
        "model": MODEL,
        "top_k": TOP_K,
        "planner": "shared_unified_hybrid_optional_known_asker",
        "global_scope_context": "frozen_audio_relation_profile_v2_reused_verbatim",
        "asker_only_context": (
            "profile_grounded_raw_dense_bm25_rrf"
            if asker_context_strategy == "speaker_only"
            else "raw_dense_bm25_rrf_with_profile_prior"
        ),
        "asker_context_strategy": asker_context_strategy,
        "known_asker_answer_hint": include_known_asker_hint,
        "relation_aware_answer_context": include_relation_labels,
        "iterative_refinement": {
            "enabled": enable_iterative_refinement,
            "max_rounds": iterative_max_rounds,
            "round_budgets": iterative_round_budgets,
            "only_after_profile_round0": True,
        },
        "same_shared_answer_prompt": True,
        "same_shared_judge_prompt": True,
        "answer_query_source": answer_query_source,
        "uses_oracle_asker": False,
        "uses_benchmark_label_for_retrieval": False,
        "uses_answer_for_retrieval": False,
        "uses_gold_for_retrieval": False,
        "answer_and_judge_prompt_sha256": sha256_file(prompt_module),
        "retrieval_freeze_sha256": frozen["retrieval_freeze_sha256"],
        "base_result_sha256": sha256_file(base_result_path),
    }
    partial_path = out_dir / "matched_pairs.partial.json"
    final_path = out_dir / "matched_pairs.json"
    if final_path.exists():
        final = read_object(final_path, "complete shared-scope result")
        if final.get("status") != "complete" or final.get("run_config") != run_config:
            raise EvaluationContractError("existing shared-scope output has a different contract")
        return final
    if partial_path.exists():
        document = read_object(partial_path, "partial shared-scope result")
        if document.get("status") != "partial" or document.get("run_config") != run_config:
            raise EvaluationContractError("existing shared-scope partial has a different contract")
    else:
        document = {
            "schema_version": SCHEMA_VERSION,
            "status": "partial",
            "run_config": run_config,
            "pairs": [],
        }
    done = {(row["case_id"], row["qa_id"]) for row in document["pairs"]}
    if len(done) != len(document["pairs"]):
        raise EvaluationContractError("shared-scope partial has duplicate QA IDs")

    # Answers and gold are opened only after the complete context freeze above.
    base_results = _base_v2_results(base_result_path)
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
        qa = canonical.get(key)
        base_pair = base_results.get(key)
        if qa is None or base_pair is None:
            raise EvaluationContractError(f"canonical/base result missing: {key}")
        identity = row["query_identity"]
        answer_query = _answer_query(row, answer_query_source)
        character = (
            str(identity.get("profile_display"))
            if identity.get("confidence") == "high" and identity.get("profile_display")
            else "user"
        )
        reused = row["speaker_scope"] == "global" and _same_context(base_pair, row)
        if reused:
            arm = dict(base_pair["arms"]["audio_relation_profile_v2"])
        else:
            dates = [
                str(item.get("date") or "")
                for item in row["base_current_r12_context"] if item.get("date")
            ]
            answer_context = (
                _relation_aware_answer_context(row)
                if include_relation_labels and row["speaker_scope"] == "asker_only"
                else _answer_context(row)
            )
            answer_prompt = answer_messages(
                answer_query, answer_context, character=character,
                last_date=max(dates) if dates else "",
            )
            if include_known_asker_hint and row["speaker_scope"] == "asker_only":
                answer_prompt = _answer_messages_with_known_asker(
                    answer_prompt, known_asker=character,
                )
            prediction = cache.call(answer_prompt)
            judged = _judge(cache, extract_json, judge_messages(
                row["question"], str(qa.get("answer") or ""), prediction,
            ))
            arm = {
                "prediction": prediction,
                "score": judged["score"],
                "judge": judged,
                "context_node_ids": row["context_node_ids"],
                "retrieved_refer_ids": row["retrieved_refer_ids"],
                "context_sha256": sha256_json(row["context"]),
            }
        document["pairs"].append({
            "case_id": key[0],
            "qa_id": key[1],
            "fine_category": qa.get("category"),
            "four_way_category": qa.get("four_way_category"),
            "question": row["question"],
            "answer_query": answer_query,
            "reference_answer": str(qa.get("answer") or ""),
            "gold_evidence": gold_evidence(qa),
            "runtime_character": character,
            "query_identity_confidence": identity.get("confidence", "unresolved"),
            "speaker_scope": row["speaker_scope"],
            "scope_trace": row["scope_trace"],
            "route_plan": row["route_plan"],
            "reused_global_v2_result": reused,
            "arm": arm,
        })
        write_json_atomic(partial_path, document)
        print(
            f"[eval {position}/{len(frozen['rows'])}] {key[0]}/{key[1]} scope={row['speaker_scope']} "
            f"score={arm['score']:.2f} reused={reused}",
            flush=True,
        )
    if len(document["pairs"]) != len(frozen["rows"]):
        raise EvaluationContractError("shared-scope evaluation ended before every requested QA")
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in document["pairs"]:
        by_category[str(pair.get("four_way_category"))].append(pair)
    document["status"] = "complete"
    document["summary"] = {
        "overall_llm_score": sum(pair["arm"]["score"] for pair in document["pairs"]) / len(document["pairs"]),
        "scope_counts": {
            scope: sum(pair["speaker_scope"] == scope for pair in document["pairs"])
            for scope in ("global", "asker_only")
        },
        "reused_global_results": sum(
            bool(pair["reused_global_v2_result"]) for pair in document["pairs"]
        ),
        "by_four_way_category": {
            category: {
                "count": len(rows),
                "llm_score": sum(row["arm"]["score"] for row in rows) / len(rows),
                "asker_only": sum(row["speaker_scope"] == "asker_only" for row in rows),
            }
            for category, rows in sorted(by_category.items())
        },
    }
    write_json_atomic(final_path, document)
    return document


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enable-paid-evaluation", action="store_true")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--base-result", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--answer-query-source",
        choices=("original", "planner_resolved_for_asker_only"),
        default="original",
    )
    parser.add_argument(
        "--asker-context-strategy",
        choices=("speaker_only", "soft_profile_prior"),
        default="speaker_only",
    )
    parser.add_argument("--include-known-asker-hint", action="store_true")
    parser.add_argument("--include-relation-labels", action="store_true")
    parser.add_argument("--enable-iterative-refinement", action="store_true")
    parser.add_argument("--iterative-max-rounds", type=int, default=3)
    parser.add_argument("--iterative-round-budgets", default="30,8,5")
    parser.add_argument("--max-attempts", type=int, default=8)
    parser.add_argument("--retry-delay-seconds", type=float, default=300.0)
    args = parser.parse_args()
    if not args.enable_paid_evaluation:
        print("paid shared-scope evaluation is default-off; pass --enable-paid-evaluation")
        return 2
    if not 1 <= args.max_attempts <= 8:
        print("--max-attempts must be in [1,8]")
        return 2
    from lane_llm import call_llm
    from utils.llm_client import extract_json, print_usage_summary

    try:
        result = run(
            config_path=args.config.resolve(strict=True),
            base_result_path=args.base_result.resolve(strict=True),
            out_dir=args.out_dir.resolve(),
            answer_query_source=args.answer_query_source,
            asker_context_strategy=args.asker_context_strategy,
            include_known_asker_hint=args.include_known_asker_hint,
            include_relation_labels=args.include_relation_labels,
            enable_iterative_refinement=args.enable_iterative_refinement,
            iterative_max_rounds=args.iterative_max_rounds,
            iterative_round_budgets=args.iterative_round_budgets,
            max_attempts=args.max_attempts,
            retry_delay_seconds=args.retry_delay_seconds,
            call_llm=call_llm,
            extract_json=extract_json,
        )
    except (AuditContractError, EvaluationContractError, OSError, RuntimeError, ValueError) as exc:
        print(f"shared-scope full-case evaluation failed closed: {exc}")
        return 2
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print_usage_summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
