#!/usr/bin/env python3
"""Low-latency online speaker identity tracking for VoxPolyBench.

The inference path consumes each waveform and its clean transcript in temporal
order. It never loads a participant roster, target speaker count, TTS voice
map, or turn-level speaker label. Hidden labels are opened only by --score,
after prediction artifacts have been written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from ecapa_encoder import EcapaEncoder
import identity_graph
from speaker_identity_pipeline import POLICY as OFFLINE_ACOUSTIC_POLICY
from speaker_identity_pipeline import reconcile_fragments


@dataclass(frozen=True)
class OnlinePolicy:
    match_cosine_threshold: float = 0.35
    ema_update_cosine_threshold: float = 0.40
    ema_alpha: float = 0.05


POLICY = OnlinePolicy()
BOUNDARY_ACOUSTIC_POLICY = replace(
    OFFLINE_ACOUSTIC_POLICY,
    min_relative_separation=1.25,
    min_cohesion_ratio=0.50,
)

MID_VOCATIVE_RE = re.compile(
    r",\s*((?:(?:Ms|Mr|Mrs|Dr)\.?\s+)?[A-Z][A-Za-z'’-]+"
    r"(?:\s+[A-Z][A-Za-z'’-]+){0,2})\s*[!?:.]"
)


def session_key(value: str) -> tuple[int, str]:
    match = re.fullmatch(r"S(\d+)", value)
    return (int(match.group(1)) if match else 10**9, value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_stream(case_root: Path) -> list[dict[str, Any]]:
    """Load only audio coordinates/path and clean text; discard identity fields."""

    source = json.loads((case_root / "full_case.json").read_text())
    dialogues = source.get("dialogues")
    if not isinstance(dialogues, dict):
        raise ValueError("full_case.json must contain a dialogues mapping")
    text_by_key = {
        (str(session), index): str(turn.get("text") or "")
        for session, turns in dialogues.items()
        for index, turn in enumerate(turns)
    }

    stream = []
    audio_root = case_root / "audio"
    for session_dir in sorted(
        (path for path in audio_root.iterdir() if path.is_dir()),
        key=lambda path: session_key(path.name),
    ):
        manifest = json.loads(
            (session_dir / f"{session_dir.name}_manifest.json").read_text()
        )
        session = str(manifest.get("session_id", session_dir.name))
        for item in manifest.get("turns", []):
            turn_index = int(item["turn_idx"])
            audio_path = Path(str(item.get("audio_path") or ""))
            key = (session, turn_index)
            if key not in text_by_key:
                raise ValueError(f"missing clean text for {session}:{turn_index}")
            if not audio_path.is_file():
                raise ValueError(f"missing waveform for {session}:{turn_index}: {audio_path}")
            stream.append(
                {
                    "session": session,
                    "turn_idx": turn_index,
                    "audio_path": str(audio_path),
                    "text": text_by_key[key],
                }
            )
    if set(text_by_key) != {
        (row["session"], row["turn_idx"]) for row in stream
    }:
        raise ValueError("audio manifests and clean transcript coordinates differ")
    return stream


def load_cache(path: Path, stream: list[dict[str, Any]]) -> np.ndarray:
    expected = [f"{row['session']}:{row['turn_idx']}" for row in stream]
    cached = np.load(path, allow_pickle=False)
    if cached["keys"].tolist() != expected:
        raise ValueError(f"embedding cache coordinates differ: {path}")
    vectors = np.asarray(cached["vectors"], dtype=np.float32)
    if len(vectors) != len(stream):
        raise ValueError(f"embedding cache length differs: {path}")
    return vectors


class OnlineSpeakerMemory:
    """One-pass nearest-prototype assignment with guarded EMA adaptation."""

    def __init__(self, policy: OnlinePolicy = POLICY) -> None:
        self.policy = policy
        self.profiles: list[dict[str, Any]] = []

    def ingest(self, vector: np.ndarray) -> dict[str, Any]:
        if not self.profiles:
            return self._create(vector, None, None)

        similarities = np.asarray(
            [float(vector @ profile["prototype"]) for profile in self.profiles]
        )
        order = similarities.argsort()[::-1]
        best_index = int(order[0])
        best_similarity = float(similarities[best_index])
        runner_up = float(similarities[order[1]]) if len(order) > 1 else None
        if best_similarity < self.policy.match_cosine_threshold:
            return self._create(vector, best_similarity, runner_up)

        profile = self.profiles[best_index]
        updated = best_similarity >= self.policy.ema_update_cosine_threshold
        if updated:
            prototype = (
                (1.0 - self.policy.ema_alpha) * profile["prototype"]
                + self.policy.ema_alpha * vector
            )
            profile["prototype"] = prototype / (np.linalg.norm(prototype) + 1e-12)
            profile["updates"] += 1
        profile["assigned_turns"] += 1
        return {
            "acoustic_speaker_id": profile["speaker_id"],
            "created_new_identity": False,
            "top_cosine": best_similarity,
            "runner_up_cosine": runner_up,
            "ema_updated": updated,
        }

    def _create(
        self,
        vector: np.ndarray,
        best_similarity: float | None,
        runner_up: float | None,
    ) -> dict[str, Any]:
        speaker_id = f"ONLINE_SPK_{len(self.profiles) + 1:03d}"
        self.profiles.append(
            {
                "speaker_id": speaker_id,
                "prototype": vector.copy(),
                "assigned_turns": 1,
                "updates": 0,
            }
        )
        return {
            "acoustic_speaker_id": speaker_id,
            "created_new_identity": True,
            "top_cosine": best_similarity,
            "runner_up_cosine": runner_up,
            "ema_updated": False,
        }


def mid_vocative_events(session: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Link recurrent clause-internal vocatives to the immediate response."""

    adjacency_edges = {
        frozenset((left["cluster"], right["cluster"]))
        for left, right in zip(rows, rows[1:])
        if left["cluster"] != right["cluster"]
    }
    observed_clusters = {row["cluster"] for row in rows}
    hub_clusters = {
        cluster
        for cluster in observed_clusters
        if adjacency_edges and all(cluster in edge for edge in adjacency_edges)
    }
    if not hub_clusters:
        return []
    events = []
    for source, response in zip(rows, rows[1:]):
        if source["cluster"] == response["cluster"]:
            continue
        for match in MID_VOCATIVE_RE.finditer(source["text"]):
            name = match.group(1).strip()
            if not identity_graph.plausible_name(name):
                continue
            events.append(
                {
                    "cluster": response["cluster"],
                    "name": name,
                    "event_type": "direct_response",
                    "evidence_pattern": "mid_vocative",
                    "source_turn": source["turn_idx"],
                    "speaker_turn": response["turn_idx"],
                    "session": session,
                }
            )
    return events

def resolve_seen_identities(
    records: list[dict[str, Any]], vectors: list[np.ndarray]
) -> dict[str, Any]:
    """Resolve aliases from evidence observed up to the current session only."""

    base_labels = [record["acoustic_speaker_id"] for record in records]
    turns = [
        {"session": record["session"], "turn_idx": record["turn_idx"]}
        for record in records
    ]
    reconciled_labels, acoustic_audit = reconcile_fragments(
        base_labels,
        np.asarray(vectors, dtype=np.float32),
        turns,
        BOUNDARY_ACOUSTIC_POLICY,
    )
    raw_by_component: dict[str, set[str]] = defaultdict(set)
    for component, raw_label in zip(reconciled_labels, base_labels):
        raw_by_component[component].add(raw_label)
    component_to_canonical = {
        component: sorted(raw_labels)[0]
        for component, raw_labels in raw_by_component.items()
    }
    reconciled_labels = [
        component_to_canonical[component] for component in reconciled_labels
    ]
    acoustic_aliases = {
        raw_label: component_to_canonical[component]
        for component, raw_labels in raw_by_component.items()
        for raw_label in raw_labels
    }

    sessions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record, reconciled_label in zip(records, reconciled_labels):
        sessions[record["session"]].append(
            {
                "session": record["session"],
                "turn_idx": record["turn_idx"],
                "cluster": reconciled_label,
                "text": record["text"],
            }
        )

    accepted, rejected, seen_events = [], [], set()
    for session in sorted(sessions, key=session_key):
        rows = sorted(sessions[session], key=lambda row: row["turn_idx"])
        proposed = identity_graph.self_identification_events(session, rows)
        proposed += identity_graph.direct_response_events(session, rows)
        proposed += mid_vocative_events(session, rows)
        for event in proposed:
            evidence_family = (
                "self_identification"
                if event["event_type"] == "self_identification"
                else "direct_response"
            )
            signature = (
                event["cluster"],
                identity_graph.norm(event["name"]),
                evidence_family,
                event["source_turn"],
                event["speaker_turn"],
                event["session"],
            )
            if signature in seen_events:
                continue
            seen_events.add(signature)
            if event.get("evidence_pattern") == "mid_vocative":
                valid, reasons = True, []
            else:
                valid, reasons = identity_graph.validate_event(event, rows)
            audited = {**event, "valid": valid, "rejection_reasons": reasons}
            (accepted if valid else rejected).append(audited)

    strict_direct_name_counts = Counter(
        identity_graph.norm(event["name"])
        for event in accepted
        if event["event_type"] == "direct_response"
        and event.get("evidence_pattern") != "mid_vocative"
    )
    mid_vocative_counts = Counter(
        (event["cluster"], identity_graph.norm(event["name"]))
        for event in accepted
        if event.get("evidence_pattern") == "mid_vocative"
    )
    recurrent_accepted = []
    for event in accepted:
        is_self_identification = event["event_type"] == "self_identification"
        is_full_name = len(identity_graph.core_name(event["name"]).split()) >= 2
        if event.get("evidence_pattern") == "mid_vocative":
            repeated = mid_vocative_counts[
                (event["cluster"], identity_graph.norm(event["name"]))
            ] >= 2
        else:
            repeated = strict_direct_name_counts[
                identity_graph.norm(event["name"])
            ] >= 2
        if is_self_identification or is_full_name or repeated:
            recurrent_accepted.append(event)
        else:
            rejected.append(
                {
                    **event,
                    "valid": False,
                    "rejection_reasons": ["single_word_name_not_repeated"],
                }
            )
    accepted = recurrent_accepted

    clusters = sorted(set(reconciled_labels))
    resolved, conflicts = identity_graph.resolve_cluster_names(accepted, clusters)
    identities, _ = identity_graph.build_stable_identities(resolved)

    reconciled_aliases: dict[str, str] = {}
    canonical_identities = []
    for identity in identities:
        clusters_in_identity = sorted(identity["acoustic_clusters"])
        stable_id = clusters_in_identity[0]
        for cluster in clusters_in_identity:
            reconciled_aliases[cluster] = stable_id
        canonical_identities.append(
            {
                "stable_speaker_id": stable_id,
                "speaker_name": identity.get("canonical_name"),
                "acoustic_speaker_ids": clusters_in_identity,
            }
        )
    aliases = {
        raw_label: reconciled_aliases[reconciled_label]
        for raw_label, reconciled_label in acoustic_aliases.items()
    }
    names = {
        identity["stable_speaker_id"]: identity.get("speaker_name")
        for identity in canonical_identities
    }
    return {
        "accepted_events": accepted,
        "rejected_events": rejected,
        "resolved_clusters": resolved,
        "conflicts": conflicts,
        "stable_identities": canonical_identities,
        "acoustic_reconciliation": acoustic_audit,
        "acoustic_aliases": acoustic_aliases,
        "aliases": aliases,
        "names": names,
    }


def prediction_payload(
    case_root: Path,
    records: list[dict[str, Any]],
    identity_state: dict[str, Any],
    boundary_updates: list[dict[str, Any]],
    timings: dict[str, float],
) -> dict[str, Any]:
    aliases = identity_state["aliases"]
    names = identity_state["names"]
    predictions = []
    for record in records:
        raw_id = record["acoustic_speaker_id"]
        reconciled_id = identity_state["acoustic_aliases"][raw_id]
        stable_id = aliases[raw_id]
        predictions.append(
            {
                "session": record["session"],
                "turn_idx": record["turn_idx"],
                "cluster": reconciled_id,
                "acoustic_speaker_id": raw_id,
                "reconciled_speaker_id": reconciled_id,
                "stable_identity_id": stable_id,
                "speaker_name": names.get(stable_id),
                "created_new_identity": record["created_new_identity"],
                "top_cosine": record["top_cosine"],
                "runner_up_cosine": record["runner_up_cosine"],
                "ema_updated": record["ema_updated"],
            }
        )
    return {
        "case_id": case_root.name,
        "protocol": (
            "online ECAPA nearest-prototype matching with guarded EMA; "
            "session-boundary transcript-grounded name aliasing; no roster, "
            "target speaker count, speaker labels, or language model"
        ),
        "policy": asdict(POLICY),
        "boundary_acoustic_policy": asdict(BOUNDARY_ACOUSTIC_POLICY),
        "language_model_calls": 0,
        "acoustic_profile_count": len({row["acoustic_speaker_id"] for row in records}),
        "stable_identity_count": len(identity_state["stable_identities"]),
        "boundary_updates": boundary_updates,
        "accepted_identity_events": identity_state["accepted_events"],
        "rejected_identity_events": identity_state["rejected_events"],
        "resolved_clusters": identity_state["resolved_clusters"],
        "acoustic_reconciliation": identity_state["acoustic_reconciliation"],
        "identity_conflicts": identity_state["conflicts"],
        "stable_identities": identity_state["stable_identities"],
        "predictions_frozen": predictions,
        "timing_seconds": timings,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    case_root = args.case_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stream = load_stream(case_root)
    print(f"loaded {len(stream)} turns", flush=True)

    vectors = None
    cache_source = None
    if args.embedding_cache is not None:
        cache_source = args.embedding_cache.resolve()
        vectors = load_cache(cache_source, stream)
    elif args.reuse_embeddings and (output_dir / "online_embeddings.npz").is_file():
        cache_source = output_dir / "online_embeddings.npz"
        vectors = load_cache(cache_source, stream)

    encoder = None if vectors is not None else EcapaEncoder(device=args.device)
    memory = OnlineSpeakerMemory(POLICY)
    records, collected_vectors, boundary_updates = [], [], []
    embedding_seconds = matching_seconds = identity_seconds = 0.0
    last_session = None
    identity_state = None

    for index, row in enumerate(stream):
        if last_session is not None and row["session"] != last_session:
            started = time.perf_counter()
            identity_state = resolve_seen_identities(records, collected_vectors)
            identity_seconds += time.perf_counter() - started
            boundary_updates.append(
                {
                    "after_session": last_session,
                    "turns_seen": len(records),
                    "acoustic_profiles": len(memory.profiles),
                    "reconciled_acoustic_profiles": identity_state["acoustic_reconciliation"]["final_cluster_count"],
                    "stable_identities": len(identity_state["stable_identities"]),
                }
            )

        started = time.perf_counter()
        vector = vectors[index] if vectors is not None else encoder.encode(row["audio_path"])
        embedding_seconds += time.perf_counter() - started
        if vector is None:
            raise RuntimeError(f"empty waveform: {row['audio_path']}")
        vector = np.asarray(vector, dtype=np.float32)
        vector /= np.linalg.norm(vector) + 1e-12
        collected_vectors.append(vector)

        started = time.perf_counter()
        decision = memory.ingest(vector)
        matching_seconds += time.perf_counter() - started
        records.append({**row, **decision})
        last_session = row["session"]

    started = time.perf_counter()
    identity_state = resolve_seen_identities(records, collected_vectors)
    identity_seconds += time.perf_counter() - started
    boundary_updates.append(
        {
            "after_session": last_session,
            "turns_seen": len(records),
            "acoustic_profiles": len(memory.profiles),
            "reconciled_acoustic_profiles": identity_state["acoustic_reconciliation"]["final_cluster_count"],
            "stable_identities": len(identity_state["stable_identities"]),
        }
    )

    output_cache = output_dir / "online_embeddings.npz"
    keys = np.asarray([f"{row['session']}:{row['turn_idx']}" for row in stream])
    np.savez_compressed(
        output_cache,
        keys=keys,
        vectors=np.asarray(collected_vectors, dtype=np.float32),
    )
    timings = {
        "embedding": round(embedding_seconds, 6),
        "online_matching_and_ema": round(matching_seconds, 6),
        "session_boundary_identity": round(identity_seconds, 6),
        "total_inference": round(embedding_seconds + matching_seconds + identity_seconds, 6),
    }
    result = prediction_payload(
        case_root, records, identity_state, boundary_updates, timings
    )
    prediction_path = output_dir / "online_identity_predictions.json"
    prediction_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")

    score = None
    if args.score:
        score = identity_graph.score_frozen_predictions(case_root, result)
        (output_dir / "scoring_only.json").write_text(
            json.dumps(score, ensure_ascii=False, indent=2) + "\n"
        )

    profiles = np.asarray([profile["prototype"] for profile in memory.profiles])
    np.savez_compressed(
        output_dir / "online_registry_state.npz",
        speaker_ids=np.asarray([profile["speaker_id"] for profile in memory.profiles]),
        assigned_turns=np.asarray([profile["assigned_turns"] for profile in memory.profiles]),
        updates=np.asarray([profile["updates"] for profile in memory.profiles]),
        prototypes=profiles,
    )
    script_path = Path(__file__).resolve()
    manifest = {
        "case_id": case_root.name,
        "inference_protocol": "online_audio_plus_seen_clean_text_roster_free",
        "turns": len(records),
        "parameters": asdict(POLICY),
        "boundary_acoustic_parameters": asdict(BOUNDARY_ACOUSTIC_POLICY),
        "language_model_calls": 0,
        "embedding_cache_source": str(cache_source) if cache_source else None,
        "timing_seconds": timings,
        "code_sha256": {
            path.name: sha256_file(path)
            for path in [script_path, script_path.with_name("ecapa_encoder.py"), script_path.with_name("identity_graph.py")]
        },
        "artifacts": {
            "predictions": str(prediction_path),
            "embeddings": str(output_cache),
            "registry_state": str(output_dir / "online_registry_state.npz"),
            "scoring_only": str(output_dir / "scoring_only.json") if score else None,
        },
        "scoring_only": score,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(manifest, ensure_ascii=False), flush=True)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--embedding-cache",
        type=Path,
        help="Optional coordinate-checked ECAPA cache for fast reproducibility tests.",
    )
    parser.add_argument(
        "--reuse-embeddings",
        action="store_true",
        help="Reuse output-dir/online_embeddings.npz when coordinates match.",
    )
    parser.add_argument(
        "--score",
        action="store_true",
        help="Open hidden labels only after writing predictions, for QA only.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    try:
        run(parse_args())
    except Exception as exc:
        print(f"online speaker identity pipeline failed: {exc}", file=sys.stderr)
        raise
