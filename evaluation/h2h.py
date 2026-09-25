"""Run three independent route strategies on frozen H2HMem memories."""
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
V3_ROOT = Path(os.environ.get(
    "V3_PIPELINE_ROOT", WORK_ROOT / "vendor/v3_update_pipeline"
))
sys.path.insert(0, str(V3_ROOT))

from eval.eval_all_mem_gallery import _compress_image_to_b64  # noqa: E402
from eval.h2hmem_prompts import (  # noqa: E402
    answer_messages, answer_messages_with_image, answer_messages_with_memory_images,
    get_format_constraint, judge_messages,
)
from core.planner import (  # noqa: E402
    heuristic_plan, mg_keyword_r1_plan, model_type_fixed_plan,
    two_level_dynamic_plan, unified_hybrid_plan,
)
from core.evidence_projection import project_grounding  # noqa: E402
from core.provenance import select_context  # noqa: E402
from core.query_fusion import normalize_query_variants  # noqa: E402
from core.temporal_contract import (  # noqa: E402
    apply_frozen_time_range,
    stable_sort_answer_visible_raw,
    temporal_contract_enabled,
)
from core.visual_budget import prioritize_visual_rows  # noqa: E402
from core.collection_consensus import rerank_collections  # noqa: E402
from core.evidence_bundle import (  # noqa: E402
    answer_view,
    evidence_bundle_metadata_enabled,
    make_evidence_group,
    merge_evidence_groups,
)
from core.iterative_retrieval import (  # noqa: E402
    configured_top_k,
    iterative_retrieval_enabled,
    retrieve_iteratively,
)
from utils.llm_client import call_llm, extract_json, print_usage_summary  # noqa: E402


DATA_ROOT = Path(os.environ.get(
    "H2HMEM_ROOT", WORK_ROOT / "data/H2HMEM"
))
TOKEN_RE = re.compile(r"[a-z0-9]+")
SUB_TYPE_TO_POINT = {
    "Unimodal Precise Recall": "FR",
    "Cross-modal Related Retrieval": "VR",
    "Knowledge Resolution": "KR",
    "Temporal Reasoning": "TR",
    "Multimodal Causal Inference": "MR",
    "Reference & Evolution Tracking": "MR",
    "Test-Time Learning": "TTL",
    "Conflict Detection": "CD",
    "Answer Refusal": "AR",
}


def _embedding_request(kind: str, values: list[str]) -> np.ndarray:
    url = os.environ.get("EMBEDDING_SERVER_URL", "http://localhost:9981")
    payload = {
        "type": "batch_text" if kind == "text" else "batch_image",
        "texts" if kind == "text" else "images": values,
        "instr": "Represent the text for retrieval." if kind == "text" else "Represent the image for retrieval.",
    }
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        out = json.loads(response.read())
    array = np.frombuffer(base64.b64decode(out["emb"]), dtype=np.float32)
    result = array.reshape(tuple(out["shape"]))
    return result / np.maximum(np.linalg.norm(result, axis=1, keepdims=True), 1e-12)


def encode_texts(values: list[str]) -> np.ndarray:
    parts = [_embedding_request("text", values[i:i+32]) for i in range(0, len(values), 32)]
    return np.vstack(parts)


def encode_images(values: list[str]) -> np.ndarray:
    parts = [_embedding_request("image", values[i:i+8]) for i in range(0, len(values), 8)]
    return np.vstack(parts)


class H2HIndex:
    def __init__(self, memory_path: Path):
        self.path = memory_path
        self.doc = json.loads(memory_path.read_text(encoding="utf-8"))
        self.layers = {
            "raw": self.doc["raw"], "fact": self.doc["facts"],
            "collection": self.doc["collections"],
        }
        self.raw_by_ref = {row["refer_ids"][0]: row for row in self.layers["raw"]}
        self.raw_by_image = defaultdict(list)
        self.raw_by_session_ordinal = {}
        for row in self.layers["raw"]:
            self.raw_by_session_ordinal[(
                str(row.get("session_id") or ""),
                int(row.get("ordinal") or 0),
            )] = row
            for image_id in (row.get("image_captions") or {}):
                self.raw_by_image[str(image_id)].append(row)
        self.embeddings = {}
        self.bm25 = {}
        for layer, rows in self.layers.items():
            texts = [str(row.get("retrieval_text") or row.get("text") or "") for row in rows]
            self.embeddings[layer] = self._cached_array(layer, texts, encode_texts)
            self.bm25[layer] = self._build_bm25(texts)

        self.caption_rows = [row for row in self.layers["raw"] if row.get("image_captions")]
        caption_texts = [" ".join(row["image_captions"].values()) for row in self.caption_rows]
        self.caption_embeddings = self._cached_array("caption", caption_texts, encode_texts)
        self.image_rows = []
        image_paths = []
        for row in self.layers["raw"]:
            for path in row.get("image_paths", []):
                if Path(path).exists():
                    self.image_rows.append(row)
                    image_paths.append(path)
        self.image_embeddings = self._cached_array("image", image_paths, encode_images)

    def _cached_array(self, label: str, values: list[str], encoder):
        if not values:
            return np.zeros((0, 1), dtype=np.float32)
        signature = hashlib.sha256("\n".join(values).encode()).hexdigest()[:12]
        cache = self.path.with_name(f"{self.path.stem}.{label}.{signature}.npy")
        if cache.exists():
            return np.load(cache)
        result = encoder(values)
        np.save(cache, result)
        return result

    @staticmethod
    def _tokens(text):
        return TOKEN_RE.findall(text.lower())

    def _build_bm25(self, texts):
        docs = [self._tokens(text) for text in texts]
        counts = [Counter(doc) for doc in docs]
        lengths = np.asarray([len(doc) for doc in docs], dtype=np.float32)
        avg = float(lengths.mean()) if len(lengths) else 1.0
        df = Counter(token for doc in docs for token in set(doc))
        n = max(len(docs), 1)
        idf = {token: math.log(1 + (n-c+0.5)/(c+0.5)) for token, c in df.items()}
        return counts, lengths, avg, idf

    def _bm25_values(self, layer, query):
        counts, lengths, avg, idf = self.bm25[layer]
        values = np.zeros(len(counts), dtype=np.float32)
        for token in set(self._tokens(query)):
            if token not in idf:
                continue
            for i, row in enumerate(counts):
                freq = row.get(token, 0)
                if freq:
                    denominator = freq + 1.5 * (0.25 + 0.75 * lengths[i] / max(avg, 1e-6))
                    values[i] += idf[token] * freq * 2.5 / denominator
        return values

    @staticmethod
    def _top_rows(rows, values, top_k, sparse=False):
        result = []
        for index in np.argsort(-values):
            if sparse and float(values[index]) <= 0:
                continue
            row = dict(rows[int(index)])
            row["score"] = float(values[index])
            result.append(row)
            if len(result) >= top_k:
                break
        return result

    def _aggregate_visual_to_collection(self, raw_ranked):
        score = defaultdict(float)
        for rank, row in enumerate(raw_ranked, 1):
            score[row["session_id"]] += 1.0 / (60 + rank)
        collections = {row["session_id"]: row for row in self.layers["collection"]}
        result = []
        for sid in sorted(score, key=score.get, reverse=True):
            if sid in collections:
                row = dict(collections[sid]); row["score"] = score[sid]; result.append(row)
        return result

    def rank(self, layer, retriever, query, question_image, top_k=30):
        if retriever in {"dense", "bm25"}:
            if retriever == "dense":
                values = self.embeddings[layer] @ encode_texts([query])[0]
            else:
                values = self._bm25_values(layer, query)
            return self._top_rows(
                self.layers[layer], values, top_k, sparse=retriever == "bm25"
            )
        if retriever == "caption":
            if not len(self.caption_rows):
                return []
            values = self.caption_embeddings @ encode_texts([query])[0]
            raw = self._top_rows(self.caption_rows, values, top_k)
            return raw if layer == "raw" else self._aggregate_visual_to_collection(raw)
        if retriever == "image":
            if not len(self.image_rows):
                return []
            query_vector = (
                encode_images([question_image])[0]
                if question_image and Path(question_image).exists()
                else encode_texts([query])[0]
            )
            values = self.image_embeddings @ query_vector
            raw = self._top_rows(self.image_rows, values, top_k)
            return raw if layer == "raw" else self._aggregate_visual_to_collection(raw)
        return []

    def project(self, ranked):
        scores = defaultdict(float)
        relations = defaultdict(list)
        evidence_groups = defaultdict(list)
        for rank, row in enumerate(ranked, 1):
            direct_refs = [
                str(ref) for ref in row.get("refer_ids", [])
                if str(ref) in self.raw_by_ref
            ]
            group = (
                make_evidence_group(
                    str(row.get("node_id") or f"fact:{rank}"),
                    direct_refs,
                    source_rank=rank,
                )
                if evidence_bundle_metadata_enabled() else None
            )
            for raw, weight, relation in project_grounding(
                row,
                self.raw_by_ref,
                self.raw_by_image,
                self.raw_by_session_ordinal,
                neighbor_radius=1,
            ):
                ref = raw["refer_ids"][0]
                scores[ref] += weight / (60 + rank)
                relations[ref].append(relation)
                if group is not None and ref in direct_refs:
                    evidence_groups[ref].append(group)
        return [dict(
                    self.raw_by_ref[ref],
                    score=scores[ref],
                    projection_relations=list(dict.fromkeys(relations[ref])),
                    **({"evidence_groups": evidence_groups[ref]}
                       if evidence_groups[ref] else {}),
                )
                for ref in sorted(scores, key=scores.get, reverse=True)]

    @staticmethod
    def rrf(channels):
        scores = defaultdict(float); rows = {}; sources = defaultdict(list)
        for name, weight, ranked in channels:
            for rank, row in enumerate(ranked, 1):
                key = row["node_id"]
                if key not in rows:
                    rows[key] = row
                elif evidence_bundle_metadata_enabled():
                    rows[key] = dict(
                        rows[key],
                        evidence_groups=merge_evidence_groups(
                            rows[key].get("evidence_groups"),
                            row.get("evidence_groups"),
                        ),
                    )
                scores[key] += weight / (60 + rank); sources[key].append(name)
        result = []
        for key in sorted(scores, key=scores.get, reverse=True):
            row = dict(rows[key], score=scores[key], retrieval_channels=sources[key])
            result.append(row)
        return result

    def execute(self, plan, question_image, top_k=30, candidate_k=30):
        channels = []; collection_channels = []; trace = []; visual_ranked = []
        time_range = plan.get("time_range")
        temporal_contract = temporal_contract_enabled()
        for ri, route in enumerate(plan.get("routes", [])):
            layer = route["layer"]
            query = str(route.get("query") or plan.get("rewritten_query") or "")
            for retriever in route.get("retrievers", []):
                ranked = self.rank(
                    layer, retriever, query, question_image, candidate_k
                )
                ranked = apply_frozen_time_range(ranked, time_range)
                name = f"{layer}.{retriever}"
                trace.append({
                    "route_index": ri,
                    "channel": name, "n": len(ranked),
                })
                if layer == "collection":
                    collection_channels.append((name, 1.0, ranked))
                else:
                    if layer == "raw" and retriever == "image":
                        visual_ranked.extend(ranked)
                    projected = ranked if layer == "raw" else self.project(ranked)
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
        fused = self.rrf(channels)
        collections = rerank_collections(self.rrf(collection_channels), fused)
        selected = collections[0] if collections else None
        node = None; promoted = []
        if selected:
            member_ids = set(selected.get("member_fact_ids", []))
            dense_members = [row for row in self.rank(
                "fact", "dense", plan.get("rewritten_query") or "", None,
                len(self.layers["fact"])
            ) if row["node_id"] in member_ids]
            bm25_members = [row for row in self.rank(
                "fact", "bm25", plan.get("rewritten_query") or "", None,
                len(self.layers["fact"])
            ) if row["node_id"] in member_ids]
            facts = self.rrf([
                ("collection.inner.fact_dense", 1.0, dense_members),
                ("collection.inner.fact_bm25", 1.0, bm25_members),
            ])[:8]
            promoted = self.project(facts)
            refs = list(dict.fromkeys(ref for fact in facts for ref in fact["refer_ids"]))
            node = {
                "node_id": selected["node_id"], "text": "[Retrieved event collection]\n" +
                "\n".join(fact["text"] for fact in facts),
                "speaker": "multiple speakers", "date": selected.get("date", ""),
                "refer_ids": refs, "score": 1.0, "memory_layer": "collection",
                "source_speaker_refs": list(dict.fromkeys(
                    str(value)
                    for fact in facts
                    for value in ([fact.get("source_speaker_ref")]
                                  if fact.get("source_speaker_ref") else [])
                )),
                "addressee_refs": list(dict.fromkeys(
                    str(value)
                    for fact in facts
                    for value in fact.get("addressee_refs", [])
                    if value
                )),
            }
            trace.append({"channel": "collection.expand.fact_hybrid", "n": len(facts)})
        if not fused and not node:
            fused = self.rank(
                "raw", "dense", plan.get("rewritten_query") or "", None,
                candidate_k,
            )
            trace.append({"channel": "fallback.raw.dense", "n": len(fused)})
        if node:
            promoted_ids = {row["node_id"] for row in promoted}
            context = [node] + promoted + [row for row in fused if row["node_id"] not in promoted_ids]
        else:
            context = fused
        if plan.get("strategy") == "unified_hybrid_route":
            support_quota = min(
                8, max(2, len(node.get("refer_ids", []))) if node else 2
            )
            context, selection = select_context(
                context, top_k=top_k, support_quota=support_quota,
                expose_upper=False,
            )
            trace.append({"channel": "provenance_selector", **selection})
        else:
            context = context[:top_k]
        # Freeze the visual candidate order before an answer-only temporal
        # reorder.  Otherwise sorting raw rows can silently change which five
        # images are sent, confounding the sort-only treatment.
        visual_context = list(context)
        if temporal_contract:
            context = stable_sort_answer_visible_raw(
                context, sort_by=plan.get("sort_by"), enabled=True
            )
        elif plan.get("sort_by") == "time" and not node:
            context.sort(key=lambda row: (row.get("date", ""), row.get("ordinal", 0)))
        image_candidates = prioritize_visual_rows(
            visual_ranked, visual_context,
            lambda row: str(row.get("node_id") or ""),
        )
        return context, trace, selected, image_candidates


def collect_questions(dialogue_path):
    rows = []
    for path in sorted(dialogue_path.glob("scenes/session*/questions.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for i, question in enumerate(data.get("questions", [])):
            qtype = question.get("question_type") or {}
            rows.append({
                "qa_id": f"{path.parent.name}_Q{i}",
                "session": path.parent.name,
                "question": (question.get("question") or {}).get("text", ""),
                "question_image": (question.get("question") or {}).get("image", ""),
                "answer": str(question.get("original_answer") or ""),
                "sub_type": qtype.get("sub_type", ""),
                "answer_sessions": [
                    str(value) for value in question.get("answer_session", [])
                ],
                "answer_dialogue": str(question.get("answer_dialogue") or ""),
                "validation_notes": str(question.get("validation_notes") or ""),
                "original_question_id": str(question.get("original_question_id") or ""),
            })
    return rows


def resolve_question_image(dialogue_path, session, value):
    if not value:
        return None
    parts = value.split("/")
    path = (dialogue_path / "scenes" / parts[0] / "image" / parts[1]) if len(parts) == 2 else (
        dialogue_path / "scenes" / session / "image" / value
    )
    return str(path) if path.exists() else None


def collect_memory_images(context, max_k=5):
    """Expose only images selected into top-k context, preserving image ids."""
    collected = []
    for row in context:
        if len(collected) >= max_k:
            break
        paths = row.get("image_paths") or []
        captions = row.get("image_captions") or {}
        image_ids = list(captions) if isinstance(captions, dict) else []
        for index, path in enumerate(paths):
            if len(collected) >= max_k:
                break
            if not Path(path).exists():
                continue
            try:
                # Several retrieved images share one request. Keep each image
                # readable while bounding the aggregate Velen gateway payload.
                encoded = _compress_image_to_b64(
                    path, max_long_edge=768, quality=75
                )
            except Exception:
                continue
            collected.append({
                "mem_id": row.get("node_id", "?"),
                "image_id": (
                    image_ids[index] if index < len(image_ids) else Path(path).name
                ),
                "base64": encoded,
            })
    return collected


def evaluate(
    dialogue, strategy, split, memory_dir, out_dir, limit, qa_ids=None,
    per_type: int | None = None, route_plan_dir: Path | None = None,
):
    dialogue_path = DATA_ROOT / split / dialogue
    questions = collect_questions(dialogue_path)
    if qa_ids:
        questions = [row for row in questions if row["qa_id"] in qa_ids]
    if per_type:
        kept = []
        counts = defaultdict(int)
        for row in questions:
            key = row.get("sub_type") or "unknown"
            if counts[key] >= per_type:
                continue
            counts[key] += 1
            kept.append(row)
        questions = kept
    if limit is not None:
        questions = questions[:limit]
    index = H2HIndex(memory_dir / f"{split}_{dialogue}.json")
    result_dir = out_dir / strategy; result_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / f"{split}_{dialogue}.json"
    results = json.loads(result_path.read_text()) if result_path.exists() else []
    done = {row["qa_id"] for row in results}
    frozen_plans = {}
    if route_plan_dir is not None:
        frozen_path = route_plan_dir / strategy / f"{split}_{dialogue}.json"
        if not frozen_path.exists():
            raise FileNotFoundError(f"missing frozen route-plan file: {frozen_path}")
        frozen_rows = json.loads(frozen_path.read_text(encoding="utf-8"))
        frozen_plans = {
            str(row["qa_id"]): row["route_plan"]
            for row in frozen_rows
            if row.get("qa_id") and row.get("route_plan")
        }

    def router(messages):
        return call_llm(messages, max_tokens=512).strip()

    top_k = configured_top_k()

    for position, qa in enumerate(questions, 1):
        if qa["qa_id"] in done:
            continue
        image_path = resolve_question_image(
            dialogue_path, qa["session"], qa["question_image"]
        )
        if route_plan_dir is not None:
            if qa["qa_id"] not in frozen_plans:
                raise KeyError(
                    f"missing frozen route plan for {dialogue}/{qa['qa_id']}"
                )
            # Keep the archived plan immutable so this diagnostic changes only
            # the executor treatment selected by its opt-in environment flag.
            plan = json.loads(json.dumps(frozen_plans[qa["qa_id"]]))
        elif strategy == "heuristic_gate":
            plan = heuristic_plan(qa["question"], bool(image_path))
        elif strategy == "mg_keyword_r1_port":
            plan = mg_keyword_r1_plan(qa["question"], bool(image_path))
        elif strategy == "model_type_fixed_route":
            plan = model_type_fixed_plan(qa["question"], bool(image_path), router)
        elif strategy == "two_level_dynamic_route":
            plan = two_level_dynamic_plan(qa["question"], bool(image_path), router)
        elif strategy == "unified_hybrid_route":
            plan = unified_hybrid_plan(qa["question"], bool(image_path), router)
        else:
            raise ValueError(strategy)
        iterative_diagnostics = {"enabled": False, "rounds_executed": 1}
        selected_collection = None
        if strategy == "unified_hybrid_route" and iterative_retrieval_enabled():
            def execute_round(round_plan):
                round_context, round_trace, round_selected, round_images = index.execute(
                    round_plan, image_path, top_k=top_k, candidate_k=30
                )
                return {
                    "context": round_context,
                    "trace": round_trace,
                    "memory_image_candidates": round_images,
                    "need_memory_images": any(
                        "image" in route.get("retrievers", [])
                        for route in round_plan.get("routes", [])
                    ),
                    "selected_collection": (
                        round_selected.get("node_id") if round_selected else None
                    ),
                    "expanded_refer_count": len(
                        round_selected.get("refer_ids", []) if round_selected else []
                    ),
                }

            iterative = retrieve_iteratively(
                question=qa["question"],
                has_question_image=bool(image_path),
                initial_plan=plan,
                execute=execute_round,
                call_router=router,
                top_k=top_k,
            )
            context = iterative["context"]
            trace = iterative["trace"]
            image_candidates = iterative["memory_image_candidates"]
            selected_collection = iterative["selected_collection"]
            iterative_diagnostics = iterative["iterative_retrieval"]
            plan = iterative["route_plan"]
        else:
            context, trace, selected, image_candidates = index.execute(
                plan, image_path, top_k=top_k, candidate_k=30
            )
            selected_collection = selected.get("node_id") if selected else None
        bundled_context, bundle_diagnostics = answer_view(context, plan=plan)
        answer_context = [{
            **row, "mem_id": row["node_id"],
            "caption": row.get("image_captions", {}),
            "text": f"[{row.get('speaker','unknown')}; {row.get('date','')}]: {row.get('text','')}",
        } for row in bundled_context]
        point = SUB_TYPE_TO_POINT.get(qa["sub_type"], "FR")
        hint = get_format_constraint(point)
        hint += (
            " If the question requests an image identifier, copy the exact "
            "<session>:<image_file> id from memory; do not rewrite img3.jpg as "
            "Figure 3 or omit the file extension."
        )
        try:
            uses_image_route = any(
                "image" in route.get("retrievers", [])
                for route in plan.get("routes", [])
            )
            memory_images = (
                collect_memory_images(image_candidates)
                if uses_image_route else []
            )
            question_image_b64 = (
                _compress_image_to_b64(
                    image_path, max_long_edge=768, quality=75
                ) if image_path else None
            )
            if memory_images:
                messages = answer_messages_with_memory_images(
                    qa["question"], answer_context, memory_images,
                    question_image_base64=question_image_b64, hint=hint,
                )
            elif image_path:
                messages = answer_messages_with_image(
                    qa["question"], answer_context, question_image_b64, hint=hint
                )
            else:
                messages = answer_messages(qa["question"], answer_context, hint=hint)
            prediction = call_llm(messages).strip()
        except Exception:
            prediction = call_llm(answer_messages(qa["question"], answer_context, hint=hint)).strip()
        try:
            judged = extract_json(call_llm(judge_messages(
                qa["question"], qa["answer"], prediction
            )))
            score = float(judged.get("score", 0.0))
        except Exception as exc:
            judged = {"score": 0.0, "reasoning": f"judge parse failed: {exc}"}; score = 0.0
        results.append({
            **qa, "point": point, "prediction": prediction, "score": score,
            "n_context": len(context),
            "retrieval_top_k": top_k,
            "retrieved_refer_ids": list(dict.fromkeys(
                ref for row in context for ref in row.get("refer_ids", [])
            )),
            "route_plan": plan, "route_trace": trace,
            "context_node_ids": [row.get("node_id") for row in context],
            "context_image_ids": list(dict.fromkeys(
                image_id
                for row in context
                for image_id in (row.get("image_captions") or {})
            )),
            "answer_memory_image_ids": [
                row["image_id"] for row in memory_images
            ],
            "selected_collection": selected_collection,
            "evidence_bundle": bundle_diagnostics,
            "iterative_retrieval": iterative_diagnostics,
            "judge": judged,
        })
        result_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[{dialogue} {strategy}] {position}/{len(questions)} {point} score={score:.2f}", flush=True)
    avg = sum(row["score"] for row in results) / max(len(results), 1)
    summary = {"dialogue": dialogue, "strategy": strategy, "n": len(results), "llm_score": avg}
    print(json.dumps(summary), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="multi-party")
    parser.add_argument("--dialogue", action="append", required=True)
    parser.add_argument("--strategy", action="append", required=True, choices=[
        "heuristic_gate", "model_type_fixed_route", "two_level_dynamic_route",
        "mg_keyword_r1_port", "unified_hybrid_route",
    ])
    parser.add_argument(
        "--memory-dir", type=Path,
        default=WORK_ROOT / "artifacts" / "memory" / "h2h",
    )
    parser.add_argument(
        "--out", type=Path,
        default=WORK_ROOT / "artifacts" / "evaluation" / "h2h",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--per-type", type=int,
        help="deterministically keep the first N QA of every subtype per dialogue",
    )
    parser.add_argument(
        "--qa-ids",
        help="comma-separated QA ids for integrity diagnostics, e.g. session1_Q0,session1_Q1",
    )
    parser.add_argument(
        "--route-plan-dir", type=Path,
        help=(
            "reuse archived route plans from <dir>/<strategy>/<split>_<dialogue>.json; "
            "this suppresses router calls for controlled executor diagnostics"
        ),
    )
    args = parser.parse_args()
    qa_ids = {value.strip() for value in (args.qa_ids or "").split(",") if value.strip()}
    summaries = [evaluate(
        d, s, args.split, args.memory_dir, args.out, args.limit, qa_ids,
        args.per_type, args.route_plan_dir,
    )
                 for d in args.dialogue for s in args.strategy]
    print(json.dumps(summaries, indent=2)); print_usage_summary()


if __name__ == "__main__":
    main()
