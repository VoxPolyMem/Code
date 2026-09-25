"""Build grounded raw/fact/collection memories for VoxPolyBench.

The builder is resumable per 12-turn extraction window. It never consumes QA,
answers, categories, or gold evidence. Every node is grounded to bottom turn IDs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
WORK_ROOT = HERE.parent
V2_ROOT = Path(os.environ.get(
    "V2_PIPELINE_ROOT",
    WORK_ROOT / "vendor/v2_update_pipeline",
))
sys.path.insert(0, str(V2_ROOT))

from adapters.voxpoly import iter_bottom_nodes, load_case  # noqa: E402
from utils.llm_client import call_llm, extract_json, print_usage_summary  # noqa: E402


PROMPT_VERSION = "audio-grounded-facts-v1"
WINDOW = 12
DEFAULT_CASE_ROOT = Path(os.environ.get(
    "VOXPOLYBENCH_ROOT", WORK_ROOT / "data/VoxPolyBench/cases"
))
DEFAULT_OUT = WORK_ROOT / "artifacts" / "memory" / "voxpoly"


def messages(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    transcript = "\n".join(
        f"[{i}] {row['speaker']}: {row['text']}" for i, row in enumerate(rows)
    )
    return [
        {
            "role": "system",
            "content": (
                "Extract grounded, self-contained atomic memories from a multi-speaker "
                "conversation. Resolve pronouns and relative time when the dialogue "
                "provides enough information. Do not infer unsupported information. "
                "Return strict JSON only."
            ),
        },
        {
            "role": "user",
            "content": f"""Dialogue window:
{transcript}

Return {{"facts":[{{"text":"self-contained fact","speaker":"name",
"addressee":"name, group, or unknown","turn_refs":[0],
"fact_type":"fact|event|preference|relationship|plan|status_update|opinion"}}]}}.
turn_refs must be nonempty local integer indices and must include every source
turn used by the fact. Skip greetings and questions that carry no durable fact.""",
        },
    ]


def _save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def build_case(case_root: Path, case_name: str, out_dir: Path) -> dict[str, Any]:
    source_path, data = load_case(case_root, case_name)
    raw_nodes = list(iter_bottom_nodes(data))
    by_session: dict[str, list[dict[str, Any]]] = {}
    for node in raw_nodes:
        by_session.setdefault(node["session_id"], []).append(node)

    out_path = out_dir / f"{case_name}.json"
    if out_path.exists():
        payload = json.loads(out_path.read_text(encoding="utf-8"))
        if payload.get("prompt_version") != PROMPT_VERSION:
            raise RuntimeError(f"cache prompt mismatch: {out_path}")
    else:
        payload = {
            "prompt_version": PROMPT_VERSION,
            "case": case_name,
            "source_path": str(source_path),
            "raw": raw_nodes,
            "windows": {},
            "facts": [],
            "collections": [],
        }

    for sid, rows in by_session.items():
        for start in range(0, len(rows), WINDOW):
            key = f"{sid}@{start}"
            if payload["windows"].get(key, {}).get("status") == "complete":
                continue
            window = rows[start:start + WINDOW]
            response = call_llm(
                messages(window), response_format={"type": "json_object"}, max_tokens=4096
            )
            parsed = extract_json(response)
            facts = []
            for item in parsed.get("facts", []):
                local_refs = item.get("turn_refs") or []
                refs = list(dict.fromkeys(
                    window[index]["refer_ids"][0]
                    for index in local_refs
                    if isinstance(index, int) and 0 <= index < len(window)
                ))
                text = str(item.get("text") or "").strip()
                if not text or not refs:
                    continue
                facts.append({
                    "node_id": f"fact:{case_name}:{key}:{len(facts)}",
                    "layer": "fact",
                    "text": text,
                    "speaker": str(item.get("speaker") or "unknown"),
                    "addressee": str(item.get("addressee") or "unknown"),
                    "fact_type": str(item.get("fact_type") or "fact"),
                    "session_id": sid,
                    "date": window[0].get("date", ""),
                    "refer_ids": refs,
                })
            payload["windows"][key] = {
                "status": "complete",
                "refer_ids": [row["refer_ids"][0] for row in window],
                "facts": facts,
            }
            _save(out_path, payload)
            print(f"[{case_name}] {key}: {len(window)} turns -> {len(facts)} facts", flush=True)

    facts = [
        fact
        for key in sorted(payload["windows"])
        for fact in payload["windows"][key].get("facts", [])
    ]
    collections = []
    for sid, rows in by_session.items():
        members = [fact for fact in facts if fact["session_id"] == sid]
        refs = list(dict.fromkeys(
            ref for fact in members for ref in fact.get("refer_ids", [])
        ))
        if not members or not refs:
            continue
        collections.append({
            "node_id": f"collection:{case_name}:{sid}",
            "layer": "collection",
            "text": f"Grounded event collection for session {sid}.",
            "retrieval_text": " ".join(fact["text"] for fact in members),
            "session_id": sid,
            "date": rows[0].get("date", ""),
            "member_fact_ids": [fact["node_id"] for fact in members],
            "refer_ids": refs,
        })
    payload["raw"] = raw_nodes
    payload["facts"] = facts
    payload["collections"] = collections
    payload["meta"] = {
        "turns": len(raw_nodes),
        "facts": len(facts),
        "collections": len(collections),
        "complete_windows": len(payload["windows"]),
        "empty_refer_ids": sum(
            not node.get("refer_ids")
            for layer in (raw_nodes, facts, collections)
            for node in layer
        ),
    }
    if payload["meta"]["empty_refer_ids"]:
        raise RuntimeError("memory integrity failure: empty refer_ids")
    _save(out_path, payload)
    return payload["meta"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", action="append", required=True)
    parser.add_argument("--case-root", type=Path, default=DEFAULT_CASE_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    for case_name in args.case:
        print(json.dumps(
            {"case": case_name, **build_case(args.case_root, case_name, args.out)},
            ensure_ascii=False,
        ), flush=True)
    print_usage_summary()


if __name__ == "__main__":
    main()
