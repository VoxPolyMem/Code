#!/usr/bin/env python3
"""Reproducible roster-free speaker identity resolution for VoxPolyBench.

The inference path reads waveform files, session/turn coordinates, and the
clean source transcript.  It never reads a participant roster, a target
speaker count, TTS voice assignments, or turn-level speaker labels.  Hidden
labels are opened only by the optional scoring pass after predictions have
been written to disk.

The final pipeline has two conceptual modules:

1. acoustic identity induction with sample-size-aware fragment reconciliation;
2. deterministic transcript-grounded name resolution.

All method parameters are frozen below.  They are intentionally not exposed as
command-line flags, which prevents per-case threshold tuning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ecapa_encoder import EcapaEncoder
from identity_graph import infer_identity_graph, score_frozen_predictions


@dataclass(frozen=True)
class AcousticPolicy:
    """Frozen development-time policy for anonymous acoustic reconciliation."""

    graph_cosine_threshold: float = 0.55
    max_fragment_size_ratio: float = 1.0 / 3.0
    top_k: int = 5

    # Multi-utterance fragment evidence.
    min_relative_separation: float = 1.50
    min_cohesion_ratio: float = 0.70
    min_session_overlap: float = 0.50

    # A singleton has no reliable within-cluster distribution, so the same
    # absolute-evidence/margin principle uses local-neighbour statistics.
    min_singleton_target_size: int = 10
    min_singleton_top_k_mean: float = 0.35
    min_singleton_margin: float = 0.075
    min_singleton_relative_separation: float = 1.25


POLICY = AcousticPolicy()


def session_key(value: str) -> tuple[int, str]:
    match = re.fullmatch(r"S(\d+)", value)
    return (int(match.group(1)) if match else 10**9, value)


def load_anonymous_turns(case_root: Path) -> list[dict]:
    """Read only waveform paths and coordinates from TTS manifests."""

    audio_root = case_root / "audio"
    if not audio_root.is_dir():
        return []
    turns = []
    session_dirs = sorted(
        (path for path in audio_root.iterdir() if path.is_dir()),
        key=lambda path: session_key(path.name),
    )
    for session_dir in session_dirs:
        manifest_path = session_dir / f"{session_dir.name}_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        session = str(manifest.get("session_id", session_dir.name))
        for item in manifest.get("turns", []):
            audio_path = item.get("audio_path")
            if audio_path and Path(audio_path).is_file():
                turns.append(
                    {
                        "path": str(audio_path),
                        "session": session,
                        "turn_idx": int(item["turn_idx"]),
                    }
                )
    return turns


def normalized_centroid(vectors: np.ndarray) -> np.ndarray:
    centroid = vectors.mean(axis=0)
    return centroid / (np.linalg.norm(centroid) + 1e-12)


def connected_components(vectors: np.ndarray, threshold: float) -> tuple[list[str], dict[str, Any]]:
    """Build high-precision anonymous components from a cosine graph."""

    similarities = vectors @ vectors.T
    parent = list(range(len(vectors)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    edge_count = 0
    for left in range(len(vectors)):
        for right in range(left + 1, len(vectors)):
            if float(similarities[left, right]) < threshold:
                continue
            edge_count += 1
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                parent[right_root] = left_root

    root_to_id: dict[int, str] = {}
    labels = []
    for index in range(len(vectors)):
        root = find(index)
        if root not in root_to_id:
            root_to_id[root] = f"SPK_{len(root_to_id) + 1:03d}"
        labels.append(root_to_id[root])
    sizes = sorted((labels.count(label) for label in set(labels)), reverse=True)
    return labels, {
        "algorithm": "cosine_connected_components",
        "similarity_threshold": threshold,
        "edges": edge_count,
        "components": len(root_to_id),
        "component_sizes": sizes,
    }


def members_from_labels(labels: list[str]) -> dict[str, list[int]]:
    members: dict[str, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        members[label].append(index)
    return dict(members)


def component_summary(indices: list[int], vectors: np.ndarray, turns: list[dict]) -> dict[str, Any]:
    component_vectors = vectors[indices]
    within_median = None
    if len(indices) > 1:
        within = component_vectors @ component_vectors.T
        within_median = float(np.median(within[np.triu_indices(len(indices), k=1)]))
    sessions: dict[str, int] = defaultdict(int)
    for index in indices:
        sessions[str(turns[index]["session"])] += 1
    return {
        "indices": list(indices),
        "size": len(indices),
        "centroid": normalized_centroid(component_vectors),
        "within_median": within_median,
        "sessions": dict(sessions),
    }


def pair_evidence(
    source: dict[str, Any],
    target: dict[str, Any],
    vectors: np.ndarray,
    top_k: int,
) -> dict[str, Any]:
    """Compute the common evidence record used by every merge decision."""

    cross = vectors[source["indices"]] @ vectors[target["indices"]].T
    flat = cross.ravel()
    top = np.sort(flat)[-min(top_k, len(flat)):]
    shared = set(source["sessions"]) & set(target["sessions"])
    smaller_session_count = min(len(source["sessions"]), len(target["sessions"]))
    within = [
        value
        for value in (source["within_median"], target["within_median"])
        if value is not None
    ]
    cross_median = float(np.median(flat))
    cohesion = cross_median / max(1e-6, min(within)) if within else None
    nearest_flat = int(np.argmax(flat))
    nearest_source, nearest_target = np.unravel_index(nearest_flat, cross.shape)
    return {
        "source_size": int(source["size"]),
        "target_size": int(target["size"]),
        "size_ratio": min(source["size"], target["size"]) / max(source["size"], target["size"]),
        "centroid_cosine": float(source["centroid"] @ target["centroid"]),
        "max_pair_cosine": float(flat.max()),
        "top_k_pair_mean": float(top.mean()),
        "cross_median": cross_median,
        "cohesion_ratio": cohesion,
        "session_overlap": len(shared) / max(1, smaller_session_count),
        "nearest_pair_indices": [
            int(source["indices"][nearest_source]),
            int(target["indices"][nearest_target]),
        ],
    }


def _candidate_score(evidence: dict[str, Any]) -> float:
    # Cross-median is robust when a fragment has repeated observations.  For a
    # singleton, top-k support is a lower-variance estimator than one centroid
    # or one nearest-neighbour comparison.
    if int(evidence["source_size"]) == 1:
        return float(evidence["top_k_pair_mean"])
    return float(evidence["cross_median"])


def evaluate_fragment_link(
    evidence: dict[str, Any],
    runner_up: dict[str, Any] | None,
    reciprocal_best: bool,
    policy: AcousticPolicy,
) -> tuple[bool, list[str]]:
    """Apply one evidence/margin/compatibility principle to every fragment."""

    reasons = []
    if float(evidence["size_ratio"]) > policy.max_fragment_size_ratio:
        reasons.append("balanced_components")

    score = _candidate_score(evidence)
    runner_score = _candidate_score(runner_up) if runner_up is not None else None
    if int(evidence["source_size"]) == 1:
        if int(evidence["target_size"]) < policy.min_singleton_target_size:
            reasons.append("target_too_small")
        if float(evidence["top_k_pair_mean"]) < policy.min_singleton_top_k_mean:
            reasons.append("top_k_mean_below_floor")
        if runner_score is not None:
            if score - runner_score < policy.min_singleton_margin:
                reasons.append("margin_below_floor")
            if score / max(1e-8, runner_score) < policy.min_singleton_relative_separation:
                reasons.append("relative_separation_below_floor")
    else:
        if not reciprocal_best:
            reasons.append("not_reciprocal_best")
        if runner_score is not None and score < policy.min_relative_separation * max(runner_score, 0.02):
            reasons.append("relative_separation_below_floor")
        if evidence["cohesion_ratio"] is None or float(evidence["cohesion_ratio"]) < policy.min_cohesion_ratio:
            reasons.append("cohesion_below_floor")
        if float(evidence["session_overlap"]) < policy.min_session_overlap:
            reasons.append("session_support_below_floor")
    return not reasons, reasons


def reconcile_fragments(
    base_labels: list[str],
    vectors: np.ndarray,
    turns: list[dict],
    policy: AcousticPolicy,
) -> tuple[list[str], dict[str, Any]]:
    """Reconcile small fragments without a roster or a target cluster count."""

    members = members_from_labels(base_labels)
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    merge_index = 0

    # Recompute candidates after each contraction because component evidence
    # changes when a fragment joins a larger identity.
    while len(members) > 1:
        summaries = {
            label: component_summary(indices, vectors, turns)
            for label, indices in members.items()
        }
        directed: dict[tuple[str, str], dict[str, Any]] = {}
        rankings: dict[str, list[tuple[float, str]]] = defaultdict(list)
        labels = sorted(members)
        for source_id in labels:
            source = summaries[source_id]
            for target_id in labels:
                if source_id == target_id:
                    continue
                evidence = pair_evidence(source, summaries[target_id], vectors, policy.top_k)
                evidence.update({"source_cluster": source_id, "target_cluster": target_id})
                directed[(source_id, target_id)] = evidence
                rankings[source_id].append((_candidate_score(evidence), target_id))
        for values in rankings.values():
            values.sort(reverse=True)

        proposals = []
        for source_id, ranked in rankings.items():
            if not ranked:
                continue
            target_id = ranked[0][1]
            evidence = directed[(source_id, target_id)]
            # Fragment reconciliation is directional.  Only the smaller side
            # can propose a contraction, while the complete bidirectional
            # ranking above is retained to test reciprocal-best support.
            if evidence["source_size"] > evidence["target_size"]:
                continue
            runner_up = directed.get((source_id, ranked[1][1])) if len(ranked) > 1 else None
            target_ranked = rankings.get(target_id, [])
            reciprocal = bool(target_ranked and target_ranked[0][1] == source_id)
            ok, reasons = evaluate_fragment_link(evidence, runner_up, reciprocal, policy)
            event = {
                **evidence,
                "score": _candidate_score(evidence),
                "runner_up": None if runner_up is None else {
                    "cluster": runner_up["target_cluster"],
                    "score": _candidate_score(runner_up),
                },
                "accepted": ok,
                "reasons": reasons,
                "evidence_profile": "singleton" if evidence["source_size"] == 1 else "multi_utterance",
            }
            if ok:
                proposals.append(event)
            else:
                rejected.append(event)

        if not proposals:
            break
        proposals.sort(key=lambda row: (row["score"], -row["size_ratio"]), reverse=True)
        chosen = proposals[0]
        source_id, target_id = chosen["source_cluster"], chosen["target_cluster"]
        if source_id not in members or target_id not in members:
            continue
        merge_index += 1
        merged_id = f"MERGED_{merge_index:03d}"
        members[merged_id] = members.pop(source_id) + members.pop(target_id)
        accepted.append({**chosen, "merged_component": merged_id})

    label_by_index: dict[int, str] = {}
    for label, indices in members.items():
        for index in indices:
            label_by_index[index] = label
    final_labels = [label_by_index[index] for index in range(len(base_labels))]
    return final_labels, {
        "policy": asdict(policy),
        "accepted_edges": accepted,
        "rejected_candidate_audit": rejected,
        "base_cluster_count": len(set(base_labels)),
        "final_cluster_count": len(set(final_labels)),
    }


def extract_or_load_embeddings(
    turns: list[dict],
    device: str,
    cache_path: Path,
    reuse: bool,
) -> np.ndarray:
    keys = [f"{turn['session']}:{turn['turn_idx']}" for turn in turns]
    if reuse and cache_path.is_file():
        cached = np.load(cache_path, allow_pickle=False)
        if cached["keys"].tolist() == keys:
            print(f"reusing anonymous embedding cache: {cache_path}", flush=True)
            return np.asarray(cached["vectors"], dtype=np.float32)
    extractor = EcapaEncoder(device=device)
    vectors = []
    for index, turn in enumerate(turns, start=1):
        vector = extractor.encode(turn["path"])
        if vector is None:
            raise RuntimeError(f"empty audio at {turn['session']} turn {turn['turn_idx']}")
        vectors.append(vector)
        if index % 80 == 0 or index == len(turns):
            print(f"embedded {index}/{len(turns)}", flush=True)
    result = np.asarray(vectors, dtype=np.float32)
    np.savez_compressed(cache_path, keys=np.asarray(keys), vectors=result)
    return result


def write_frozen_acoustic_predictions(
    path: Path,
    case_id: str,
    turns: list[dict],
    base_labels: list[str],
    final_labels: list[str],
    initial_graph: dict[str, Any],
    reconciliation: dict[str, Any],
) -> None:
    payload = {
        "case_id": case_id,
        "protocol": (
            "ECAPA waveform graph plus sample-size-aware fragment reconciliation; "
            "no transcript, roster, labels, TTS voice map, or target speaker count"
        ),
        "frozen": True,
        "initial_graph": initial_graph,
        "fragment_reconciliation": reconciliation,
        "predictions": [
            {
                "session": str(turn["session"]),
                "turn_idx": int(turn["turn_idx"]),
                "anonymous_base_cluster": base_labels[index],
                "anonymous_merged_cluster": final_labels[index],
            }
            for index, turn in enumerate(turns)
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(args: argparse.Namespace) -> dict[str, Any]:
    case_root = args.case_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    turns = load_anonymous_turns(case_root)
    if not turns:
        raise ValueError(f"no dialogue audio found below {case_root / 'audio'}")
    print(f"loaded {len(turns)} anonymous turns", flush=True)

    embedding_path = output_dir / "anonymous_embeddings.npz"
    vectors = extract_or_load_embeddings(turns, args.device, embedding_path, args.reuse_embeddings)
    base_labels, initial_graph = connected_components(vectors, POLICY.graph_cosine_threshold)
    final_labels, reconciliation = reconcile_fragments(base_labels, vectors, turns, POLICY)

    acoustic_path = output_dir / "acoustic_predictions.frozen.json"
    write_frozen_acoustic_predictions(
        acoustic_path,
        case_root.name,
        turns,
        base_labels,
        final_labels,
        initial_graph,
        reconciliation,
    )

    identity_path = output_dir / "identity_predictions.frozen.json"
    identity_result = infer_identity_graph(case_root, acoustic_path, identity_path)

    score = None
    if args.score:
        # The identity artifact has already been frozen on disk.  Only this
        # explicitly requested QA pass opens hidden speaker labels.
        score = score_frozen_predictions(case_root, identity_result)
        (output_dir / "scoring_only.json").write_text(
            json.dumps(score, ensure_ascii=False, indent=2) + "\n"
        )

    script_path = Path(__file__).resolve()
    code_files = [
        script_path,
        script_path.with_name("ecapa_encoder.py"),
        script_path.with_name("identity_graph.py"),
    ]
    manifest = {
        "case_id": case_root.name,
        "inference_protocol": "audio_plus_clean_transcript_roster_free",
        "language_model_calls": 0,
        "turns": len(turns),
        "base_clusters": len(set(base_labels)),
        "final_acoustic_clusters": len(set(final_labels)),
        "stable_identities": len(identity_result["stable_identities"]),
        "accepted_acoustic_fragment_edges": len(reconciliation["accepted_edges"]),
        "parameters": asdict(POLICY),
        "code_sha256": {path.name: sha256_file(path) for path in code_files},
        "artifacts": {
            "embeddings": str(embedding_path),
            "acoustic_predictions": str(acoustic_path),
            "identity_predictions": str(identity_path),
            "scoring_only": str(output_dir / "scoring_only.json") if score is not None else None,
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
        "--reuse-embeddings",
        action="store_true",
        help="Reuse the anonymous embedding cache only when turn coordinates match exactly.",
    )
    parser.add_argument(
        "--score",
        action="store_true",
        help="After freezing predictions, open hidden labels in a separate QA-only scoring pass.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    try:
        run(parse_args())
    except Exception as exc:
        print(f"speaker identity pipeline failed: {exc}", file=sys.stderr)
        raise
