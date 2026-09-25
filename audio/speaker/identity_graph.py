"""Deterministic transcript-grounded identity graph for VoxPolyBench.

Only session, turn index, anonymous acoustic cluster, and clean transcript text
are retained on the inference side.  Participant rosters and speaker labels are
never loaded.  The optional scoring function opens manifest labels only after
the prediction artifact has been frozen.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment


UNKNOWN = {"", "unknown", "none", "null", "n/a", "na"}
HONORIFICS = {"mr", "mrs", "ms", "dr", "doctor"}
NON_NAME_VOCATIVES = {
    "absolutely", "actually", "agreed", "ah", "alright", "amazing", "and",
    "anyhow", "anyway", "anytime", "awesome", "aw", "coming", "congrats",
    "cool", "definitely", "everyone", "everybody", "exactly", "excellent",
    "father", "fine", "first", "folks", "good", "gosh", "grandma", "grandpa",
    "gotcha", "great", "hello", "here", "hey", "hi", "hmm", "haha", "honestly", "man",
    "mhm", "mom", "mother", "morning", "nah", "next", "nice", "no", "nope",
    "noted", "now", "oh", "okay", "perfect", "phew", "remember", "right",
    "saved", "seriously", "so", "sure", "team", "thanks", "ugh", "um",
    "understood", "wait", "welcome", "well", "wonderful", "wow", "yeah",
    "yep", "yes",
}

SELF_NAME_RE = re.compile(
    r"\b(?i:I(?:'|’)m|I am|my name is|this is)\s+"
    r"((?:(?:Ms|Mr|Mrs|Dr)\.?\s+)?[A-Z][A-Za-z'’-]+"
    r"(?:\s+[A-Z][A-Za-z'’-]+){1,2})\b"
)
LEADING_VOCATIVE_RE = re.compile(
    r"^\s*(?:(?i:oh)\s*[,!-]?\s+)?"
    r"(?:(?i:hey|hi|hello|thanks|okay|alright|so)\s*[,!-]?\s+)?"
    r"((?:(?:Ms|Mr|Mrs|Dr)\.?\s+)?[A-Z][A-Za-z'’-]+"
    r"(?:\s+[A-Z][A-Za-z'’-]+){0,2})\s*[,!:]"
)
TRAILING_VOCATIVE_RE = re.compile(
    r",\s*((?:(?:Ms|Mr|Mrs|Dr)\.?\s+)?[A-Z][A-Za-z'’-]+"
    r"(?:\s+[A-Z][A-Za-z'’-]+){0,2})\s*[?!.]\s*$"
)


def session_key(value: str) -> tuple[int, str]:
    match = re.fullmatch(r"S(\d+)", value)
    return (int(match.group(1)) if match else 10**9, value)


def norm(value: str | None) -> str:
    return " ".join(re.sub(r"[^a-z]+", " ", str(value or "").casefold()).split())


def core_name(value: str | None) -> str:
    tokens = norm(value).split()
    while tokens and tokens[0] in HONORIFICS:
        tokens.pop(0)
    return " ".join(tokens)


def plausible_name(value: str | None) -> bool:
    tokens = core_name(value).split()
    return 1 <= len(tokens) <= 3 and all(
        token not in NON_NAME_VOCATIVES for token in tokens
    )


def same_name_family(left: str | None, right: str | None) -> bool:
    left_name, right_name = core_name(left), core_name(right)
    if not left_name or not right_name:
        return False
    if left_name == right_name:
        return True
    left_tokens, right_tokens = left_name.split(), right_name.split()
    if len(left_tokens) == 1 and len(right_tokens) >= 2:
        return left_tokens[0] == right_tokens[0]
    if len(right_tokens) == 1 and len(left_tokens) >= 2:
        return right_tokens[0] == left_tokens[0]
    return False


def token_boundary_contains(text: str, name: str) -> bool:
    haystack, needle = norm(text), norm(name)
    return bool(needle and re.search(r"(?:^| )" + re.escape(needle) + r"(?: |$)", haystack))


def self_introduces_name(text: str, name: str) -> bool:
    target = norm(name)
    return any(norm(match.group(1)) == target for match in SELF_NAME_RE.finditer(text))


def directly_addresses_name(text: str, name: str) -> bool:
    words = norm(name).split()
    if not words or not plausible_name(name):
        return False
    pattern = r"\s+".join(re.escape(word) for word in words)
    leading = re.search(
        r"^\s*(?:oh\s*[,!-]?\s+)?"
        r"(?:(?:hey|hi|hello|thanks|okay|alright|so)\s*[,!-]?\s+)?"
        + pattern
        + r"\s*[,!:]",
        text,
        flags=re.I,
    )
    trailing = re.search(r",\s*" + pattern + r"\s*[?!.]?\s*$", text, flags=re.I)
    return bool(leading or trailing)


def load_inference_view(case_root: Path, frozen_path: Path) -> tuple[dict[str, list[dict]], dict]:
    frozen = json.loads(frozen_path.read_text())
    predictions = frozen.get("predictions")
    if not isinstance(predictions, list):
        raise ValueError("frozen cluster file has no predictions list")
    cluster_by_key = {
        (str(row["session"]), int(row["turn_idx"])): str(row["anonymous_merged_cluster"])
        for row in predictions
    }

    # Deliberately copy only text and coordinates.  All identity annotations in
    # full_case.json are discarded before event extraction.
    source = json.loads((case_root / "full_case.json").read_text())
    dialogues = source.get("dialogues")
    if not isinstance(dialogues, dict):
        raise ValueError("full_case.json must contain a session-to-turns dialogue map")
    text_by_key = {
        (str(session), turn_index): str(turn.get("text") or "")
        for session, turns in dialogues.items()
        for turn_index, turn in enumerate(turns)
    }
    if set(cluster_by_key) != set(text_by_key):
        raise ValueError("frozen acoustic coordinates differ from transcript coordinates")

    sessions: dict[str, list[dict]] = defaultdict(list)
    for key, cluster in cluster_by_key.items():
        session, turn_index = key
        sessions[session].append(
            {
                "session": session,
                "turn_idx": turn_index,
                "cluster": cluster,
                "text": text_by_key[key],
            }
        )
    for rows in sessions.values():
        rows.sort(key=lambda row: row["turn_idx"])
    return dict(sessions), frozen


def self_identification_events(session: str, rows: list[dict]) -> list[dict]:
    events = []
    for row in rows:
        for match in SELF_NAME_RE.finditer(row["text"]):
            name = match.group(1).strip()
            if plausible_name(name):
                events.append(
                    {
                        "cluster": row["cluster"],
                        "name": name,
                        "event_type": "self_identification",
                        "source_turn": row["turn_idx"],
                        "speaker_turn": row["turn_idx"],
                        "session": session,
                    }
                )
    return events


def direct_response_events(session: str, rows: list[dict]) -> list[dict]:
    events = []
    for source, response in zip(rows, rows[1:]):
        match = LEADING_VOCATIVE_RE.search(source["text"])
        kind = "leading_vocative" if match else None
        if not match:
            match = TRAILING_VOCATIVE_RE.search(source["text"])
            kind = "trailing_vocative" if match else None
        if not match or source["cluster"] == response["cluster"]:
            continue
        name = match.group(1).strip()
        if not plausible_name(name):
            continue
        remainder = source["text"][match.end():].lstrip()
        if kind == "leading_vocative" and re.match(
            r"(?:and\s+)?[A-Z][A-Za-z'’-]+\s*[,—-]", remainder
        ):
            continue
        events.append(
            {
                "cluster": response["cluster"],
                "name": name,
                "event_type": "direct_response",
                "source_turn": source["turn_idx"],
                "speaker_turn": response["turn_idx"],
                "session": session,
            }
        )
    return events


def validate_event(event: dict, rows: list[dict]) -> tuple[bool, list[str]]:
    by_turn = {row["turn_idx"]: row for row in rows}
    source = by_turn.get(event["source_turn"])
    speaker = by_turn.get(event["speaker_turn"])
    reasons = []
    if source is None or speaker is None:
        return False, ["missing_turn"]
    if not plausible_name(event.get("name")):
        reasons.append("implausible_name")
    if speaker["cluster"] != event["cluster"]:
        reasons.append("speaker_cluster_mismatch")
    if not token_boundary_contains(source["text"], event["name"]):
        reasons.append("name_not_observed")
    if event["event_type"] == "self_identification":
        if event["source_turn"] != event["speaker_turn"]:
            reasons.append("self_turn_mismatch")
        if not self_introduces_name(source["text"], event["name"]):
            reasons.append("no_self_identification_phrase")
    else:
        if event["speaker_turn"] - event["source_turn"] != 1:
            reasons.append("response_not_immediate")
        if source["cluster"] == speaker["cluster"]:
            reasons.append("same_voice_for_address_and_response")
        if not directly_addresses_name(source["text"], event["name"]):
            reasons.append("not_a_direct_address")
    return not reasons, reasons


def canonical_name(claims: list[dict]) -> str:
    full_names = [
        claim["name"].strip()
        for claim in claims
        if len(norm(claim["name"]).split()) >= 2
    ]
    pool = full_names or [claim["name"].strip() for claim in claims]
    counts = Counter(norm(name) for name in pool)
    selected = sorted(counts, key=lambda item: (-counts[item], -len(item), item))[0]
    return next(name for name in pool if norm(name) == selected)


def group_compatible_claims(claims: list[dict]) -> list[list[dict]]:
    groups: list[list[dict]] = []
    for claim in claims:
        matches = [
            group
            for group in groups
            if any(same_name_family(claim["name"], member["name"]) for member in group)
        ]
        if len(matches) == 1:
            matches[0].append(claim)
        else:
            groups.append([claim])
    return groups


def resolve_cluster_names(events: list[dict], clusters: list[str]) -> tuple[dict, list[dict]]:
    claims_by_cluster: dict[str, list[dict]] = defaultdict(list)
    for event in events:
        claims_by_cluster[event["cluster"]].append(event)

    resolved, conflicts = {}, []
    for cluster in clusters:
        claims = claims_by_cluster.get(cluster, [])
        groups = group_compatible_claims(claims)
        if not groups:
            resolved[cluster] = {
                "speaker_name": None,
                "status": "unknown",
                "evidence_count": 0,
            }
            continue

        # Explicit self-identification outranks address-response evidence.
        ranks = []
        for group in groups:
            self_count = sum(
                event["event_type"] == "self_identification" for event in group
            )
            ranks.append((int(self_count > 0), self_count, len(group)))
        best_rank = max(ranks)
        winners = [group for group, rank in zip(groups, ranks) if rank == best_rank]
        if len(winners) != 1:
            resolved[cluster] = {
                "speaker_name": None,
                "status": "conflict",
                "evidence_count": len(claims),
            }
            conflicts.append({"cluster": cluster, "claim_groups": groups})
            continue
        winner = winners[0]
        resolved[cluster] = {
            "speaker_name": canonical_name(winner),
            "status": "resolved",
            "evidence_count": len(winner),
            "evidence_types": sorted({event["event_type"] for event in winner}),
        }
        if len(groups) > 1:
            conflicts.append(
                {
                    "cluster": cluster,
                    "selected": winner,
                    "discarded_lower_rank": [group for group in groups if group is not winner],
                }
            )
    return resolved, conflicts


def build_stable_identities(resolved: dict) -> tuple[list[dict], dict[str, str]]:
    identities: list[dict] = []
    for cluster in sorted(resolved):
        name = resolved[cluster].get("speaker_name")
        matches = [
            identity
            for identity in identities
            if name
            and identity["canonical_name"]
            and same_name_family(name, identity["canonical_name"])
        ]
        if len(matches) == 1:
            matches[0]["acoustic_clusters"].append(cluster)
            matches[0]["observed_names"].append(name)
        else:
            identities.append(
                {
                    "canonical_name": name,
                    "acoustic_clusters": [cluster],
                    "observed_names": [name] if name else [],
                }
            )

    cluster_to_identity = {}
    for index, identity in enumerate(identities, start=1):
        identity_id = f"IDENTITY_{index:03d}"
        identity["stable_identity_id"] = identity_id
        if identity["observed_names"]:
            identity["canonical_name"] = canonical_name(
                [{"name": name} for name in identity["observed_names"]]
            )
        for cluster in identity["acoustic_clusters"]:
            cluster_to_identity[cluster] = identity_id
            resolved[cluster]["stable_identity_id"] = identity_id
    return identities, cluster_to_identity


def infer_identity_graph(case_root: Path, frozen_path: Path, output_path: Path) -> dict:
    sessions, frozen = load_inference_view(case_root, frozen_path)
    accepted, rejected = [], []
    for session in sorted(sessions, key=session_key):
        rows = sessions[session]
        proposed = self_identification_events(session, rows) + direct_response_events(session, rows)
        seen = set()
        for event in proposed:
            signature = (
                event["cluster"],
                norm(event["name"]),
                event["event_type"],
                event["source_turn"],
                event["speaker_turn"],
            )
            if signature in seen:
                continue
            seen.add(signature)
            valid, reasons = validate_event(event, rows)
            audited = {**event, "valid": valid, "rejection_reasons": reasons}
            (accepted if valid else rejected).append(audited)

    clusters = sorted(
        {row["cluster"] for rows in sessions.values() for row in rows}
    )
    resolved, conflicts = resolve_cluster_names(accepted, clusters)
    identities, cluster_to_identity = build_stable_identities(resolved)
    predictions = [
        {
            **row,
            **resolved[row["cluster"]],
            "stable_identity_id": cluster_to_identity[row["cluster"]],
        }
        for session in sorted(sessions, key=session_key)
        for row in sessions[session]
    ]
    result = {
        "case_id": case_root.name,
        "protocol": (
            "frozen anonymous acoustic IDs plus deterministic direct identity "
            "events from clean transcript; no roster, labels, or language model"
        ),
        "transcript_source": "clean_source_text_with_identity_fields_stripped",
        "cluster_input": str(frozen_path),
        "cluster_protocol": frozen.get("protocol"),
        "language_model_calls": 0,
        "accepted_identity_events": accepted,
        "rejected_identity_events": rejected,
        "resolved_clusters": resolved,
        "identity_conflicts": conflicts,
        "stable_identities": identities,
        "predictions_frozen": predictions,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def score_frozen_predictions(case_root: Path, result: dict) -> dict[str, Any]:
    """Open speaker labels only after the inference artifact has been written."""

    truth = {}
    for session_dir in sorted(
        (path for path in (case_root / "audio").iterdir() if path.is_dir()),
        key=lambda path: session_key(path.name),
    ):
        manifest = json.loads(
            (session_dir / f"{session_dir.name}_manifest.json").read_text()
        )
        session = str(manifest.get("session_id", session_dir.name))
        for turn in manifest.get("turns", []):
            truth[(session, int(turn["turn_idx"]))] = str(turn.get("speaker_name") or "")

    rows = result["predictions_frozen"]
    assigned = [row for row in rows if norm(row.get("speaker_name"))]
    alias_correct = sum(
        core_name(row["speaker_name"]).split()[0]
        == core_name(truth[(row["session"], row["turn_idx"])]).split()[0]
        for row in assigned
    )

    predicted_ids = sorted({row["stable_identity_id"] for row in rows})
    gold_ids = sorted(
        {core_name(truth[(row["session"], row["turn_idx"])]) for row in rows}
    )
    matrix = np.zeros((len(gold_ids), len(predicted_ids)), dtype=np.int64)
    gold_index = {name: index for index, name in enumerate(gold_ids)}
    predicted_index = {name: index for index, name in enumerate(predicted_ids)}
    for row in rows:
        gold_name = core_name(truth[(row["session"], row["turn_idx"])])
        matrix[gold_index[gold_name], predicted_index[row["stable_identity_id"]]] += 1
    matched_gold, matched_predicted = linear_sum_assignment(-matrix)
    identity_correct = int(matrix[matched_gold, matched_predicted].sum())

    gold_to_predicted: dict[str, set[str]] = defaultdict(set)
    predicted_to_gold: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        gold_name = core_name(truth[(row["session"], row["turn_idx"])])
        gold_to_predicted[gold_name].add(row["stable_identity_id"])
        predicted_to_gold[row["stable_identity_id"]].add(gold_name)

    wrong_bindings = []
    for cluster, resolution in result["resolved_clusters"].items():
        predicted_name = resolution.get("speaker_name")
        if not predicted_name:
            continue
        gold_names = {
            core_name(truth[(row["session"], row["turn_idx"])])
            for row in rows
            if row["cluster"] == cluster
        }
        if core_name(predicted_name).split()[0] not in {
            name.split()[0] for name in gold_names
        }:
            wrong_bindings.append(
                {
                    "cluster": cluster,
                    "predicted_name": predicted_name,
                    "gold_names": sorted(gold_names),
                }
            )

    return {
        "scoring_only": True,
        "turns": len(rows),
        "stable_id_correct": identity_correct,
        "stable_id_accuracy": round(identity_correct / len(rows), 6),
        "stable_identity_count": len(predicted_ids),
        "gold_speaker_count": len(gold_ids),
        "name_coverage": round(len(assigned) / len(rows), 6),
        "name_alias_correct_assigned": alias_correct,
        "name_alias_accuracy_assigned": (
            round(alias_correct / len(assigned), 6) if assigned else None
        ),
        "wrong_name_bindings": wrong_bindings,
        "over_split_gold_speakers": {
            name: sorted(identities)
            for name, identities in gold_to_predicted.items()
            if len(identities) > 1
        },
        "merged_gold_speakers": {
            identity: sorted(names)
            for identity, names in predicted_to_gold.items()
            if len(names) > 1
        },
    }
