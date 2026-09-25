"""Evaluate the three independent routing strategies on VoxPolyBench."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import re
import sys
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
WORK_ROOT = HERE.parent
V2_ROOT = Path(os.environ.get(
    "V2_PIPELINE_ROOT",
    WORK_ROOT / "vendor/v2_update_pipeline",
))
# Keep this workspace ahead of the legacy v2 package. Both trees expose an
# ``adapters`` package, but VoxPoly's adapter belongs to the unified workspace.
sys.path.insert(0, str(WORK_ROOT))
sys.path.insert(1, str(V2_ROOT))

from adapters.voxpoly import gold_evidence, load_case, qa_pairs  # noqa: E402
from eval.bench_prompts import answer_messages, judge_messages  # noqa: E402
from core.planner import (  # noqa: E402
    heuristic_plan,
    mg_keyword_r1_plan,
    model_type_fixed_plan,
    two_level_dynamic_plan,
    unified_hybrid_plan,
)
from core.provenance import select_context  # noqa: E402
from core.query_fusion import normalize_query_variants  # noqa: E402
from core.temporal_contract import (  # noqa: E402
    apply_frozen_time_range,
    stable_sort_answer_visible_raw,
    temporal_contract_enabled,
)
from utils.llm_client import call_llm, extract_json, print_usage_summary  # noqa: E402


TOKEN_RE = re.compile(r"[a-z0-9]+")
CASE_ROOT = Path(os.environ.get(
    "VOXPOLYBENCH_ROOT", WORK_ROOT / "data/VoxPolyBench/cases"
))


def encode_texts(texts: list[str]) -> np.ndarray:
    url = os.environ.get("EMBEDDING_SERVER_URL", "http://localhost:9981")
    batch = int(os.environ.get("EMBEDDING_BATCH_SIZE", "32"))
    parts = []
    for start in range(0, len(texts), batch):
        payload = {
            "type": "batch_text",
            "texts": texts[start:start + batch],
            "instr": "Represent the text for retrieval.",
        }
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=600) as response:
            out = json.loads(response.read())
        array = np.frombuffer(base64.b64decode(out["emb"]), dtype=np.float32)
        parts.append(array.reshape(tuple(out["shape"])))
    result = np.vstack(parts)
    return result / np.maximum(np.linalg.norm(result, axis=1, keepdims=True), 1e-12)


class LayerIndex:
    def __init__(self, memory_path: Path):
        self.memory_path = memory_path
        self.doc = json.loads(memory_path.read_text(encoding="utf-8"))
        self.layers = {
            "raw": self.doc["raw"],
            "fact": self.doc["facts"],
            "collection": self.doc["collections"],
        }
        self.raw_by_ref = {
            node["refer_ids"][0]: node for node in self.layers["raw"]
        }
        self.fact_by_id = {node["node_id"]: node for node in self.layers["fact"]}
        self.embeddings = {}
        self.bm25 = {}
        for layer, nodes in self.layers.items():
            texts = [str(node.get("retrieval_text") or node.get("text") or "") for node in nodes]
            signature = hashlib.sha256("\n".join(texts).encode()).hexdigest()
            cache = memory_path.with_name(f"{memory_path.stem}.{layer}.{signature[:12]}.npy")
            if cache.exists():
                array = np.load(cache)
            else:
                array = encode_texts(texts)
                np.save(cache, array)
            self.embeddings[layer] = array
            self.bm25[layer] = self._build_bm25(texts)

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return TOKEN_RE.findall(text.lower())

    def _build_bm25(self, texts: list[str]):
        docs = [self._tokens(text) for text in texts]
        counts = [Counter(doc) for doc in docs]
        lengths = np.asarray([len(doc) for doc in docs], dtype=np.float32)
        avg = float(lengths.mean()) if len(lengths) else 1.0
        df = Counter(token for doc in docs for token in set(doc))
        n = max(len(docs), 1)
        idf = {token: math.log(1 + (n-c+0.5)/(c+0.5)) for token, c in df.items()}
        return counts, lengths, avg, idf

    def _bm25_scores(self, layer: str, query: str) -> np.ndarray:
        counts, lengths, avg, idf = self.bm25[layer]
        scores = np.zeros(len(counts), dtype=np.float32)
        for token in set(self._tokens(query)):
            if token not in idf:
                continue
            for i, row in enumerate(counts):
                freq = row.get(token, 0)
                if not freq:
                    continue
                denominator = freq + 1.5 * (0.25 + 0.75 * lengths[i] / max(avg, 1e-6))
                scores[i] += idf[token] * freq * 2.5 / denominator
        return scores

    def rank(self, layer: str, retriever: str, query: str, top_k: int = 30):
        if layer not in self.layers or retriever not in {"dense", "bm25"}:
            return []
        if retriever == "dense":
            values = self.embeddings[layer] @ encode_texts([query])[0]
        else:
            values = self._bm25_scores(layer, query)
        order = np.argsort(-values)
        rows = []
        for index in order:
            if retriever == "bm25" and float(values[index]) <= 0:
                continue
            row = dict(self.layers[layer][int(index)])
            row["score"] = float(values[index])
            rows.append(row)
            if len(rows) >= top_k:
                break
        return rows

    def _project(self, ranked: list[dict[str, Any]]) -> list[dict[str, Any]]:
        scores = defaultdict(float)
        for rank, node in enumerate(ranked, 1):
            for ref in node.get("refer_ids", []):
                if ref in self.raw_by_ref:
                    scores[ref] += 1.0 / (60 + rank)
        rows = []
        for ref in sorted(scores, key=scores.get, reverse=True):
            row = dict(self.raw_by_ref[ref])
            row["score"] = scores[ref]
            rows.append(row)
        return rows

    @staticmethod
    def _rrf(channels):
        scores = defaultdict(float)
        rows = {}
        sources = defaultdict(list)
        for name, weight, ranked in channels:
            for rank, row in enumerate(ranked, 1):
                key = row["node_id"]
                rows.setdefault(key, row)
                scores[key] += weight / (60 + rank)
                sources[key].append(name)
        out = []
        for key in sorted(scores, key=scores.get, reverse=True):
            row = dict(rows[key])
            row["score"] = scores[key]
            row["retrieval_channels"] = sources[key]
            out.append(row)
        return out

    def execute(self, plan: dict[str, Any], top_k: int = 30):
        channels = []
        collection_channels = []
        trace = []
        time_range = plan.get("time_range")
        temporal_contract = temporal_contract_enabled()
        for route_index, route in enumerate(plan.get("routes", [])):
            layer = route.get("layer")
            query = str(route.get("query") or plan.get("rewritten_query") or "")
            for retriever in route.get("retrievers", []):
                ranked = self.rank(layer, retriever, query, top_k=top_k)
                ranked = apply_frozen_time_range(ranked, time_range)
                name = f"{layer}.{retriever}"
                trace.append({
                    "route_index": route_index,
                    "channel": name,
                    "n": len(ranked),
                })
                if layer == "collection":
                    collection_channels.append((name, 1.0, ranked))
                else:
                    projected = ranked if layer == "raw" else self._project(ranked)
                    channels.append((
                        name, 1.0 if layer == "raw" else 0.75, projected,
                    ))

        channels, query_fusion = normalize_query_variants(channels)
        collection_channels, collection_query_fusion = normalize_query_variants(
            collection_channels
        )
        trace.append({"channel": "query_variant_fusion", **query_fusion})
        trace.append({
            "channel": "collection_query_variant_fusion", **collection_query_fusion
        })
        fused = self._rrf(channels)
        collections = self._rrf(collection_channels)
        selected = collections[0] if collections else None
        promoted = []
        collection_node = None
        if selected:
            member_ids = set(selected.get("member_fact_ids", []))
            query = str(plan.get("rewritten_query") or "")
            dense_members = [
                row for row in self.rank("fact", "dense", query, top_k=len(self.layers["fact"]))
                if row["node_id"] in member_ids
            ]
            bm25_members = [
                row for row in self.rank("fact", "bm25", query, top_k=len(self.layers["fact"]))
                if row["node_id"] in member_ids
            ]
            member_rank = self._rrf([
                ("collection.inner.fact_dense", 1.0, dense_members),
                ("collection.inner.fact_bm25", 1.0, bm25_members),
            ])
            chosen_facts = member_rank[:8]
            promoted = self._project(chosen_facts)
            refs = list(dict.fromkeys(
                ref for fact in chosen_facts for ref in fact.get("refer_ids", [])
            ))
            collection_node = {
                "node_id": selected["node_id"],
                "mem_id": selected["node_id"],
                "text": "[Retrieved event collection]\n" + "\n".join(
                    fact["text"] for fact in chosen_facts
                ),
                "speaker": "multiple speakers",
                "date": selected.get("date", ""),
                "timestamp": 0,
                "refer_ids": refs,
                "score": 1.0,
                "retrieval_channels": ["collection.expand.fact_hybrid"],
                "memory_layer": "collection",
                "source_speaker_refs": list(dict.fromkeys(
                    str(value)
                    for fact in chosen_facts
                    for value in ([fact.get("source_speaker_ref")]
                                  if fact.get("source_speaker_ref") else [])
                )),
                "addressee_refs": list(dict.fromkeys(
                    str(value)
                    for fact in chosen_facts
                    for value in fact.get("addressee_refs", [])
                    if value
                )),
            }
            trace.append({"channel": "collection.expand.fact_hybrid", "n": len(chosen_facts)})

        if not fused and not collection_node:
            fallback = self.rank("raw", "dense", plan.get("rewritten_query") or "", top_k)
            fused = fallback
            trace.append({"channel": "fallback.raw.dense", "n": len(fallback)})
        if collection_node:
            promoted_ids = {row["node_id"] for row in promoted}
            context = [collection_node] + promoted + [
                row for row in fused if row["node_id"] not in promoted_ids
            ]
        else:
            context = fused
        if plan.get("strategy") == "unified_hybrid_route":
            context, selection = select_context(
                context, top_k=top_k, support_quota=2, expose_upper=False
            )
            trace.append({"channel": "provenance_selector", **selection})
        else:
            context = context[:top_k]
        if temporal_contract:
            context = stable_sort_answer_visible_raw(
                context, sort_by=plan.get("sort_by"), enabled=True
            )
        elif plan.get("sort_by") == "time" and not collection_node:
            context.sort(key=lambda row: (row.get("date", ""), row.get("ordinal", 0)))
        return context, trace, selected


def question_text(qa: dict[str, Any]) -> str:
    return str(qa.get("question") or qa.get("question_text") or qa.get("query") or "")


def evaluate(case_name: str, strategy: str, memory_dir: Path, out_dir: Path, limit: int | None):
    _, case = load_case(CASE_ROOT, case_name)
    memory_path = memory_dir / f"{case_name}.json"
    index = LayerIndex(memory_path)
    rows = qa_pairs(case)
    if limit is not None:
        rows = rows[:limit]
    result_dir = out_dir / strategy
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / f"{case_name}.json"
    results = json.loads(result_path.read_text()) if result_path.exists() else []
    done = {row["qa_id"] for row in results}
    dates = [node.get("date", "") for node in index.layers["raw"] if node.get("date")]
    last_date = max(dates) if dates else ""

    def router(msgs):
        return call_llm(msgs, max_tokens=512).strip()

    for position, qa in enumerate(rows, 1):
        qa_id = str(qa.get("qa_id") or f"Q{position}")
        if qa_id in done:
            continue
        question = question_text(qa)
        if not question:
            raise RuntimeError(f"empty question: {case_name}/{qa_id}")
        if strategy == "heuristic_gate":
            plan = heuristic_plan(question, False)
        elif strategy == "mg_keyword_r1_port":
            plan = mg_keyword_r1_plan(question, False)
        elif strategy == "model_type_fixed_route":
            plan = model_type_fixed_plan(question, False, router)
        elif strategy == "two_level_dynamic_route":
            plan = two_level_dynamic_plan(question, False, router)
        elif strategy == "unified_hybrid_route":
            plan = unified_hybrid_plan(question, False, router)
        else:
            raise ValueError(strategy)
        context, trace, selected = index.execute(plan, top_k=30)
        answer_context = [
            {
                **row,
                "mem_id": row.get("mem_id") or row["node_id"],
                "text": f"[{row.get('speaker','unknown')}; {row.get('date','')}]: {row.get('text','')}",
            }
            for row in context
        ]
        prediction = call_llm(answer_messages(
            question, answer_context, character="user", last_date=last_date
        )).strip()
        try:
            judged = extract_json(call_llm(judge_messages(
                question, str(qa.get("answer") or ""), prediction
            )))
            score = float(judged.get("score", 0.0))
        except Exception as exc:
            judged = {"score": 0.0, "reasoning": f"judge parse failed: {exc}"}
            score = 0.0
        retrieved_refs = list(dict.fromkeys(
            ref for row in context for ref in row.get("refer_ids", [])
        ))
        gold = gold_evidence(qa)
        recall = (len(set(gold) & set(retrieved_refs)) / len(set(gold))) if gold else None
        results.append({
            "qa_id": qa_id,
            "question": question,
            "answer": qa.get("answer"),
            "prediction": prediction,
            "score": score,
            "gold_evidence": gold,
            "retrieved_refer_ids": retrieved_refs,
            "evidence_recall_at_30": recall,
            "n_context": len(context),
            "route_plan": plan,
            "route_trace": trace,
            "selected_collection": selected.get("node_id") if selected else None,
            "judge": judged,
        })
        result_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[{case_name} {strategy}] {position}/{len(rows)} score={score:.2f} recall={recall}", flush=True)
    avg = sum(row["score"] for row in results) / max(len(results), 1)
    recalls = [row["evidence_recall_at_30"] for row in results if row["evidence_recall_at_30"] is not None]
    summary = {
        "case": case_name,
        "strategy": strategy,
        "n": len(results),
        "llm_score": avg,
        "evidence_recall_at_30": sum(recalls) / max(len(recalls), 1),
    }
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", action="append", required=True)
    parser.add_argument("--strategy", action="append", choices=[
        "heuristic_gate", "model_type_fixed_route", "two_level_dynamic_route",
        "mg_keyword_r1_port", "unified_hybrid_route",
    ], required=True)
    parser.add_argument(
        "--memory-dir", type=Path,
        default=WORK_ROOT / "artifacts" / "memory" / "voxpoly",
    )
    parser.add_argument(
        "--out", type=Path,
        default=WORK_ROOT / "artifacts" / "evaluation" / "voxpoly",
    )
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    summaries = []
    for case_name in args.case:
        for strategy in args.strategy:
            summaries.append(evaluate(
                case_name, strategy, args.memory_dir, args.out, args.limit
            ))
    print(json.dumps(summaries, ensure_ascii=False, indent=2), flush=True)
    print_usage_summary()


if __name__ == "__main__":
    main()
