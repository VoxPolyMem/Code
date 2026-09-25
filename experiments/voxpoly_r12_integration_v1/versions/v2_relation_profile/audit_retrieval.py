#!/usr/bin/env python3
"""Zero-paid-API retrieval audit for AudioMem adapter v2.

The script first freezes every retrieval context using only saved R12
contexts, question text, QA-free memory, and optional audio-derived identity.
Only after a freeze digest is computed are canonical cases opened to score
gold-evidence recall.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


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

from retrieval_adapter import raw_hybrid_topk, relation_profile_context  # noqa: E402
from audio_persona_adapter import (  # noqa: E402
    is_audio_persona_query,
    speaker_owned_raw_context,
)


SCHEMA_VERSION = "voxpoly-audio-v2-retrieval-audit.v1"
TOP_K = 30


class AuditContractError(ValueError):
    pass


def read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AuditContractError(f"cannot read {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AuditContractError(f"{label} must be a JSON object")
    return value


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


def _node_id(row: Mapping[str, Any]) -> str:
    return str(row.get("node_id") or row.get("mem_id") or "")


def _refs(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    return list(dict.fromkeys(
        str(ref) for row in rows for ref in row.get("refer_ids") or [] if str(ref)
    ))


def _context_record(rows: Sequence[Mapping[str, Any]], trace: Any) -> dict[str, Any]:
    context = []
    for rank, row in enumerate(rows, 1):
        context.append({
            "rank": rank,
            "node_id": _node_id(row),
            "text": str(row.get("text") or row.get("retrieval_text") or ""),
            "speaker": row.get("speaker"),
            "speaker_ref": row.get("speaker_ref"),
            "addressee": row.get("relation_addressee", row.get("addressee")),
            "addressee_refs": list(
                row.get("relation_addressee_refs", row.get("addressee_refs")) or []
            ),
            "relation_label_source": row.get("relation_label_source"),
            "session_id": row.get("session_id"),
            "ordinal": row.get("ordinal"),
            "date": row.get("date"),
            "timestamp": row.get("timestamp"),
            "reply_to_turn_id": row.get("reply_to_turn_id"),
            "image_ids": list(row.get("image_ids") or []),
            "image_captions": dict(row.get("image_captions") or {}),
            "audio_ref": row.get("audio_ref"),
            "refer_ids": list(row.get("refer_ids") or []),
            "selection_trace": row.get("audio_retrieval_v2"),
        })
    return {
        "context_node_ids": [_node_id(row) for row in rows],
        "retrieved_refer_ids": _refs(rows),
        "n_context": len(rows),
        "context": context,
        "trace": trace,
    }


def _rrf_rank(
    dense: Sequence[Mapping[str, Any]], bm25: Sequence[Mapping[str, Any]], top_k: int
) -> list[dict[str, Any]]:
    scores: dict[str, float] = defaultdict(float)
    rows: dict[str, Mapping[str, Any]] = {}
    for ranked in (dense, bm25):
        seen: set[str] = set()
        for rank, row in enumerate(ranked, 1):
            identity = _node_id(row)
            if not identity or identity in seen:
                continue
            seen.add(identity)
            rows.setdefault(identity, row)
            scores[identity] += 1.0 / (60 + rank)
    return [
        dict(rows[identity])
        for identity in sorted(scores, key=lambda key: (-scores[key], key))[:top_k]
    ]


def _load_identity_sidecar(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    if not path.exists():
        return {}
    sidecar = read_object(path, "query identity sidecar")
    if sidecar.get("schema_version") != "voxpoly.audio-query-identity.v2":
        raise AuditContractError("unsupported identity sidecar")
    forbidden_true = [
        key for key in ("uses_filename_identity", "uses_question_text", "uses_answer", "uses_gold")
        if sidecar.get(key) is not False
    ]
    if forbidden_true:
        raise AuditContractError(f"identity sidecar violates contract: {forbidden_true}")
    result = {}
    for row in sidecar.get("observations") or []:
        key = (str(row.get("case_id")), str(row.get("qa_id")))
        if key in result:
            raise AuditContractError(f"duplicate identity observation: {key}")
        result[key] = dict(row)
    return result


def _saved_inputs(path: Path) -> dict[str, dict[str, Any]]:
    """Immediately project a saved result onto retrieval-safe fields."""

    document = read_object(path, "saved R12 result")
    safe: dict[str, dict[str, Any]] = {}
    for row in document.get("pairs") or []:
        qa_id = str(row.get("qa_id") or "")
        arm = row.get("content_only") or {}
        if not qa_id or qa_id in safe:
            raise AuditContractError("saved R12 result has missing/duplicate QA ID")
        safe[qa_id] = {
            "question": str(row.get("question") or ""),
            "context_node_ids": [str(value) for value in arm.get("context_node_ids") or []],
        }
    del document
    return safe


def freeze_retrieval(config: Mapping[str, Any], config_path: Path) -> dict[str, Any]:
    if int(config.get("top_k", -1)) != TOP_K:
        raise AuditContractError("v2 audit is frozen to Top30")
    os.environ.setdefault("EMBEDDING_SERVER_URL", str(config.get("embedding_server")))
    from evaluation.voxpoly import LayerIndex

    identity_path = (config_path.parent.parent / str(config.get("query_identity_sidecar"))).resolve()
    identities = _load_identity_sidecar(identity_path)
    frozen_rows = []
    local_dense_queries = 0
    for case_config in config.get("cases") or []:
        case_id = str(case_config["case_id"])
        memory_path = Path(case_config["memory"]).resolve(strict=True)
        memory = read_object(memory_path, f"{case_id} memory")
        raw_by_id = {_node_id(row): row for row in memory.get("raw") or []}
        saved = _saved_inputs(Path(case_config["saved_r12"]).resolve(strict=True))
        index = LayerIndex(memory_path)
        requested = list(case_config.get("panel_qa_ids") or [])
        if not requested or len(requested) != len(set(requested)):
            raise AuditContractError(f"{case_id} panel QA IDs are missing or duplicated")
        for qa_id in requested:
            if qa_id not in saved:
                raise AuditContractError(f"{case_id}/{qa_id} absent from saved R12")
            safe = saved[qa_id]
            question = safe["question"]
            current = []
            for node_id in safe["context_node_ids"]:
                if node_id not in raw_by_id:
                    raise AuditContractError(f"saved R12 node is not raw memory: {node_id}")
                current.append(dict(raw_by_id[node_id]))
            if len(current) != TOP_K or len({_node_id(row) for row in current}) != TOP_K:
                raise AuditContractError(f"saved R12 context is not unique Top30: {case_id}/{qa_id}")
            dense_raw = index.rank("raw", "dense", question, top_k=60)
            local_dense_queries += 1
            bm25_raw = index.rank("raw", "bm25", question, top_k=60)
            raw_global = raw_hybrid_topk(dense_raw, bm25_raw, top_k=TOP_K)
            if len(raw_global) != TOP_K:
                raise AuditContractError(f"raw global retrieval underfilled: {case_id}/{qa_id}")

            row = {
                "case_id": case_id,
                "qa_id": qa_id,
                "question": question,
                "is_attribution_audit": qa_id in set(case_config.get("attribution_audit_qa_ids") or []),
                "is_regression_audit": qa_id in set(case_config.get("regression_audit_qa_ids") or []),
                "arms": {
                    "current_r12": _context_record(current, {"source": "saved_matched_r12"}),
                    "raw_only_global_top30": _context_record(
                        raw_global,
                        {"source": "original_question_raw_dense_bm25_rrf", "top_k": TOP_K},
                    ),
                },
            }
            if row["is_attribution_audit"]:
                relation, relation_trace = relation_profile_context(
                    raw_global,
                    raw_global,
                    memory,
                    question,
                    query_identity=None,
                    top_k=TOP_K,
                    safety_prefix=26,
                    relation_budget=4,
                    profile_state_budget=0,
                )
                row["arms"]["raw_anchor_relation_top30"] = _context_record(
                    relation, relation_trace
                )
            identity = identities.get((case_id, qa_id))
            ranked_facts = []
            if identity and identity.get("confidence") == "high":
                dense_fact = index.rank("fact", "dense", question, top_k=30)
                local_dense_queries += 1
                bm25_fact = index.rank("fact", "bm25", question, top_k=30)
                ranked_facts = _rrf_rank(dense_fact, bm25_fact, TOP_K)
            v2, v2_trace = relation_profile_context(
                current,
                raw_global,
                memory,
                question,
                query_identity=identity,
                ranked_facts=ranked_facts,
                top_k=TOP_K,
                safety_prefix=24,
                relation_budget=4,
                profile_state_budget=2,
            )
            row["arms"]["audio_relation_profile_v2"] = _context_record(v2, v2_trace)
            # Persona adapter: the runtime gate sees only high-confidence
            # waveform identity and observable first-person wording.  The
            # complete raw ranks are needed because a speaker-owned fact may
            # be below the generic global Top-60 list.
            if is_audio_persona_query(question, identity):
                dense_persona = index.rank(
                    "raw", "dense", question, top_k=len(index.layers["raw"])
                )
                local_dense_queries += 1
                bm25_persona = index.rank(
                    "raw", "bm25", question, top_k=len(index.layers["raw"])
                )
                persona, persona_trace = speaker_owned_raw_context(
                    question=question,
                    query_identity=identity,
                    memory=memory,
                    dense_raw=dense_persona,
                    bm25_raw=bm25_persona,
                )
                row["arms"]["audio_unified_persona_v2_2"] = _context_record(
                    persona, persona_trace
                )
            else:
                row["arms"]["audio_unified_persona_v2_2"] = _context_record(
                    v2,
                    {
                        **v2_trace,
                        "persona_adapter_active": False,
                        "fallback": "audio_relation_profile_v2",
                    },
                )
            row["query_identity"] = identity or {
                "stable_asker_ref": None,
                "confidence": "unresolved",
                "source": "no_query_audio_for_this_panel_item",
            }
            frozen_rows.append(row)
    freeze_payload = {
        "schema_version": SCHEMA_VERSION,
        "top_k": TOP_K,
        "qa_count": len(frozen_rows),
        "local_embedding_dense_queries": local_dense_queries,
        "external_llm_calls": 0,
        "uses_benchmark_label_for_retrieval": False,
        "uses_answer_for_retrieval": False,
        "uses_gold_for_retrieval": False,
        "rows": frozen_rows,
    }
    freeze_payload["retrieval_freeze_sha256"] = sha256_json(freeze_payload)
    return freeze_payload


def score_frozen(
    frozen: dict[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    # Canonical QA/gold is deliberately imported and opened only after every
    # context has been frozen and hashed above.
    from adapters.voxpoly import gold_evidence, qa_pairs

    canonical_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for case_config in config.get("cases") or []:
        case_id = str(case_config["case_id"])
        case = read_object(Path(case_config["canonical_case"]).resolve(strict=True), "canonical case")
        for qa in qa_pairs(case):
            qa_id = str(qa.get("qa_id") or "")
            if qa_id:
                canonical_by_key[(case_id, qa_id)] = qa

    aggregates: dict[str, list[float]] = defaultdict(list)
    cohort_aggregates: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in frozen["rows"]:
        key = (row["case_id"], row["qa_id"])
        if key not in canonical_by_key:
            raise AuditContractError(f"canonical QA missing: {key}")
        gold = list(dict.fromkeys(map(str, gold_evidence(canonical_by_key[key]))))
        row["gold_evidence"] = gold
        for arm_name, arm in row["arms"].items():
            retrieved = set(arm["retrieved_refer_ids"])
            matched = [ref for ref in gold if ref in retrieved]
            recall = len(matched) / len(gold) if gold else None
            arm["matched_gold_evidence"] = matched
            arm["evidence_recall_at_30"] = recall
            if recall is not None:
                aggregates[arm_name].append(recall)
                cohort_aggregates["panel25"][arm_name].append(recall)
                if row["is_attribution_audit"]:
                    cohort_aggregates["attribution12"][arm_name].append(recall)
                if row["is_regression_audit"]:
                    cohort_aggregates["regression6"][arm_name].append(recall)
    frozen["summary"] = {
        name: {
            "qa_with_gold": len(values),
            "mean_evidence_recall_at_30": sum(values) / len(values) if values else None,
        }
        for name, values in sorted(aggregates.items())
    }
    frozen["cohort_summary"] = {
        cohort: {
            name: {
                "qa_with_gold": len(values),
                "mean_evidence_recall_at_30": sum(values) / len(values) if values else None,
            }
            for name, values in sorted(arms.items())
        }
        for cohort, arms in sorted(cohort_aggregates.items())
    }
    frozen["gold_opened_after_retrieval_freeze"] = True
    frozen["scored_document_sha256"] = sha256_json(frozen)
    return frozen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "configs" / "matched25.json")
    parser.add_argument("--output", type=Path, default=HERE / "artifacts" / "retrieval_audit.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve(strict=True)
    config = read_object(config_path, "audit config")
    frozen = freeze_retrieval(config, config_path)
    scored = score_frozen(frozen, config)
    write_json_atomic(args.output, scored)
    print(json.dumps({
        "output": str(args.output),
        "retrieval_freeze_sha256": scored["retrieval_freeze_sha256"],
        "qa_count": scored["qa_count"],
        "local_embedding_dense_queries": scored["local_embedding_dense_queries"],
        "external_llm_calls": 0,
        "summary": scored["summary"],
        "cohort_summary": scored["cohort_summary"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
