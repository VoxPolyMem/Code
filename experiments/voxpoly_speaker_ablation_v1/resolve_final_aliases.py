#!/usr/bin/env python3
"""Resolve final speaker aliases without opening rosters or gold labels.

The upstream online speaker artifact is immutable input.  This post-processor
uses only transcript-grounded identity events already frozen in that artifact.
It first reserves aliases supported by self-identification, then preserves
uncontested upstream resolutions, and finally assigns remaining aliases to
unresolved stable identities with a global one-to-one constraint.

This is deliberately a *final alias* resolver.  It must not be reported as
strict causal per-turn name recognition because later transcript evidence may
name an identity that appeared earlier in the case.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import tempfile
import unicodedata
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any


RESOLVER_VERSION = "voxpoly-final-alias-global-one-to-one.v1"
PROTOCOL = "Online Acoustic + Final Predicted Alias (global one-to-one)"
MIN_DIRECT_EVENTS = 2
MIN_DIRECT_SESSIONS = 2


class AliasResolutionError(ValueError):
    """Raised when the frozen prediction artifact is incomplete or ambiguous."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def alias_key(name: str) -> str:
    """Normalize the first spoken-name token used by the benchmark scorer."""

    normalized = unicodedata.normalize("NFKC", name).strip().casefold()
    normalized = re.sub(r"^(?:mr|mrs|ms|miss|dr)\.?\s+", "", normalized)
    first = normalized.split()[0] if normalized.split() else ""
    return re.sub(r"[^\w'’-]", "", first, flags=re.UNICODE)


def _nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AliasResolutionError(f"{label} must be a nonempty string")
    return value.strip()


def _display_name(events: list[dict[str, Any]], key: str, fallback: str | None) -> str:
    # Preserve the exact upstream spelling whenever it denotes the selected
    # alias.  The resolver changes assignments, not cosmetic formatting.
    if fallback and alias_key(fallback) == key:
        return fallback.strip()
    choices: list[tuple[int, int, int, str]] = []
    counts = Counter(
        str(event.get("name")).strip()
        for event in events
        if isinstance(event.get("name"), str)
        and alias_key(str(event.get("name"))) == key
    )
    for position, event in enumerate(events):
        name = event.get("name")
        if not isinstance(name, str) or alias_key(name) != key:
            continue
        # Prefer explicit self-identification and fuller names.  The negative
        # position gives deterministic preference to the earliest evidence.
        choices.append(
            (
                1 if event.get("event_type") == "self_identification" else 0,
                counts[name.strip()],
                len(name.strip()),
                name.strip(),
            )
        )
    if not choices:
        raise AliasResolutionError(f"no display spelling for alias {key!r}")
    return max(choices)[3]


def _stable_layout(document: dict[str, Any]) -> tuple[
    list[str], dict[str, list[str]], dict[str, str | None]
]:
    rows = document.get("stable_identities")
    if not isinstance(rows, list) or not rows:
        raise AliasResolutionError("stable_identities must be a nonempty list")
    stable_ids: list[str] = []
    members: dict[str, list[str]] = {}
    current_names: dict[str, str | None] = {}
    seen_members: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise AliasResolutionError(f"stable_identities[{index}] must be an object")
        stable_id = _nonempty(row.get("stable_speaker_id"), f"stable_identities[{index}].stable_speaker_id")
        if stable_id in members:
            raise AliasResolutionError(f"duplicate stable identity: {stable_id}")
        acoustic_ids = row.get("acoustic_speaker_ids")
        if not isinstance(acoustic_ids, list) or not acoustic_ids:
            raise AliasResolutionError(f"{stable_id} has no acoustic_speaker_ids")
        normalized_members = [
            _nonempty(value, f"{stable_id}.acoustic_speaker_ids") for value in acoustic_ids
        ]
        if len(normalized_members) != len(set(normalized_members)):
            raise AliasResolutionError(f"{stable_id} repeats an acoustic member")
        overlap = seen_members.intersection(normalized_members)
        if overlap:
            raise AliasResolutionError(f"acoustic IDs belong to multiple stable identities: {sorted(overlap)}")
        seen_members.update(normalized_members)
        name = row.get("speaker_name")
        if name is not None:
            name = _nonempty(name, f"{stable_id}.speaker_name")
        stable_ids.append(stable_id)
        members[stable_id] = normalized_members
        current_names[stable_id] = name
    return stable_ids, members, current_names


def _events_by_stable(
    document: dict[str, Any], members: dict[str, list[str]]
) -> dict[str, list[dict[str, Any]]]:
    acoustic_to_stable = {
        acoustic: stable_id
        for stable_id, acoustic_ids in members.items()
        for acoustic in acoustic_ids
    }
    result: dict[str, list[dict[str, Any]]] = {stable_id: [] for stable_id in members}
    events = document.get("accepted_identity_events")
    if not isinstance(events, list):
        raise AliasResolutionError("accepted_identity_events must be a list")
    for position, event in enumerate(events):
        if not isinstance(event, dict):
            raise AliasResolutionError(f"accepted_identity_events[{position}] must be an object")
        if event.get("valid") is False:
            continue
        cluster = _nonempty(event.get("cluster"), f"identity event {position}.cluster")
        if cluster not in acoustic_to_stable:
            raise AliasResolutionError(f"identity event refers to unknown acoustic cluster: {cluster}")
        name = _nonempty(event.get("name"), f"identity event {position}.name")
        if not alias_key(name):
            raise AliasResolutionError(f"identity event {position} has an empty normalized alias")
        result[acoustic_to_stable[cluster]].append(copy.deepcopy(event))
    return result


def _candidate_stats(events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        grouped[alias_key(str(event["name"]))].append(event)
    stats: dict[str, dict[str, Any]] = {}
    for key, rows in grouped.items():
        self_count = sum(row.get("event_type") == "self_identification" for row in rows)
        direct_rows = [row for row in rows if row.get("event_type") == "direct_response"]
        sessions = {str(row.get("session")) for row in direct_rows if row.get("session") is not None}
        stats[key] = {
            "self_count": self_count,
            "direct_count": len(direct_rows),
            "direct_sessions": len(sessions),
            "evidence_count": len(rows),
        }
    return stats


def _best_unique_self_alias(stats: dict[str, dict[str, Any]]) -> str | None:
    ranked = sorted(
        ((values["self_count"], key) for key, values in stats.items() if values["self_count"]),
        reverse=True,
    )
    if not ranked:
        return None
    if len(ranked) > 1 and ranked[0][0] == ranked[1][0]:
        return None
    return ranked[0][1]


def _solve_one_to_one(
    stable_ids: list[str],
    candidate_scores: dict[str, dict[str, int]],
) -> dict[str, str]:
    """Find the maximum-support one-to-one assignment, allowing unresolved IDs."""

    aliases = sorted({key for scores in candidate_scores.values() for key in scores})
    alias_index = {key: index for index, key in enumerate(aliases)}

    @lru_cache(maxsize=None)
    def solve(position: int, used_mask: int) -> tuple[int, int, tuple[str, ...]]:
        if position == len(stable_ids):
            return 0, 0, ()
        stable_id = stable_ids[position]
        suffix_score, suffix_count, suffix_choice = solve(position + 1, used_mask)
        best = (suffix_score, suffix_count, ("",) + suffix_choice)
        for key in sorted(candidate_scores.get(stable_id, {})):
            bit = 1 << alias_index[key]
            if used_mask & bit:
                continue
            tail_score, tail_count, tail_choice = solve(position + 1, used_mask | bit)
            candidate = (
                candidate_scores[stable_id][key] + tail_score,
                1 + tail_count,
                (key,) + tail_choice,
            )
            # Score and coverage dominate.  On an exact tie use the
            # lexicographically smaller complete assignment deterministically.
            if candidate[:2] > best[:2] or (
                candidate[:2] == best[:2] and candidate[2] < best[2]
            ):
                best = candidate
        return best

    _, _, choices = solve(0, 0)
    return {
        stable_id: key
        for stable_id, key in zip(stable_ids, choices)
        if key
    }


def resolve_document(
    source: dict[str, Any], *, source_sha256: str = "unit-test"
) -> dict[str, Any]:
    if source.get("language_model_calls") != 0:
        raise AliasResolutionError("input must declare language_model_calls == 0")
    if "alias_resolver" in source:
        raise AliasResolutionError("input already contains an alias_resolver result")
    stable_ids, members, current_names = _stable_layout(source)
    events = _events_by_stable(source, members)
    stats = {stable_id: _candidate_stats(events[stable_id]) for stable_id in stable_ids}

    assignments: dict[str, str] = {}
    reasons: dict[str, str] = {}
    occupied: dict[str, str] = {}

    # Stage 1: explicit self-identification has priority over all indirect
    # response evidence.  Conflicting duplicate self names are left unresolved.
    self_proposals = {
        stable_id: key
        for stable_id in stable_ids
        for key in [_best_unique_self_alias(stats[stable_id])]
        if key is not None
    }
    proposal_counts = Counter(self_proposals.values())
    for stable_id in stable_ids:
        key = self_proposals.get(stable_id)
        # Do not let processing order choose between duplicate self claims.
        if key is None or proposal_counts[key] != 1:
            continue
        assignments[stable_id] = key
        reasons[stable_id] = "self_identification"
        occupied[key] = stable_id

    # Stage 2: preserve an upstream resolved name if it does not collide with
    # a self-identified alias.  This makes the post-processor conservative.
    for stable_id in stable_ids:
        if stable_id in assignments or not current_names[stable_id]:
            continue
        key = alias_key(current_names[stable_id] or "")
        if key and key not in occupied:
            assignments[stable_id] = key
            reasons[stable_id] = "preserved_upstream_resolution"
            occupied[key] = stable_id

    # Stage 3: globally assign only repeat-supported direct-response aliases.
    unresolved = [stable_id for stable_id in stable_ids if stable_id not in assignments]
    candidate_scores: dict[str, dict[str, int]] = {}
    for stable_id in unresolved:
        candidate_scores[stable_id] = {}
        for key, values in stats[stable_id].items():
            if key in occupied or values["self_count"]:
                continue
            if (
                values["direct_count"] < MIN_DIRECT_EVENTS
                or values["direct_sessions"] < MIN_DIRECT_SESSIONS
            ):
                continue
            candidate_scores[stable_id][key] = (
                values["direct_count"] * 1000
                + values["direct_sessions"] * 100
            )
    newly_assigned = _solve_one_to_one(unresolved, candidate_scores)
    for stable_id, key in newly_assigned.items():
        assignments[stable_id] = key
        reasons[stable_id] = "global_one_to_one_direct_response"
        occupied[key] = stable_id

    displays: dict[str, str] = {}
    for stable_id, key in assignments.items():
        displays[stable_id] = _display_name(
            events[stable_id], key, current_names[stable_id]
        )

    result = copy.deepcopy(source)
    changed_stable_ids = {
        stable_id
        for stable_id in stable_ids
        if displays.get(stable_id) != current_names.get(stable_id)
    }
    for row in result["stable_identities"]:
        stable_id = row["stable_speaker_id"]
        row["speaker_name"] = displays.get(stable_id)

    acoustic_to_stable = {
        acoustic: stable_id
        for stable_id, acoustic_ids in members.items()
        for acoustic in acoustic_ids
    }
    resolved_clusters = result.get("resolved_clusters")
    if isinstance(resolved_clusters, dict):
        for acoustic_id, row in resolved_clusters.items():
            if not isinstance(row, dict) or acoustic_id not in acoustic_to_stable:
                continue
            stable_id = acoustic_to_stable[acoustic_id]
            row["speaker_name"] = displays.get(stable_id)
            if stable_id in changed_stable_ids:
                row["status"] = "resolved_by_global_one_to_one"

    predictions = result.get("predictions_frozen")
    if not isinstance(predictions, list):
        raise AliasResolutionError("predictions_frozen must be a list")
    changed_turns = 0
    for position, row in enumerate(predictions):
        if not isinstance(row, dict):
            raise AliasResolutionError(f"predictions_frozen[{position}] must be an object")
        stable_id = _nonempty(
            row.get("stable_identity_id"),
            f"predictions_frozen[{position}].stable_identity_id",
        )
        if stable_id not in members:
            raise AliasResolutionError(f"prediction refers to unknown stable identity: {stable_id}")
        old_name = row.get("speaker_name")
        new_name = displays.get(stable_id)
        if old_name != new_name:
            row["speaker_name_before_alias_resolver"] = old_name
            row["speaker_name"] = new_name
            row["alias_resolver_source"] = reasons.get(stable_id)
            changed_turns += 1

    audit_rows = []
    for stable_id in stable_ids:
        audit_rows.append(
            {
                "stable_identity_id": stable_id,
                "acoustic_speaker_ids": members[stable_id],
                "speaker_name_before": current_names[stable_id],
                "speaker_name_after": displays.get(stable_id),
                "resolution_reason": reasons.get(stable_id, "unresolved"),
                "candidate_stats": stats[stable_id],
            }
        )
    result["protocol"] = PROTOCOL
    result["alias_resolver"] = {
        "version": RESOLVER_VERSION,
        "input_prediction_sha256": source_sha256,
        "inference_inputs": [
            "stable_identities",
            "accepted_identity_events",
            "predictions_frozen.stable_identity_id",
        ],
        "forbidden_inputs": [
            "participant_roster",
            "gold_speaker_labels",
            "target_speaker_count",
            "tts_voice_map",
        ],
        "language_model_calls": 0,
        "direct_response_gate": {
            "min_events": MIN_DIRECT_EVENTS,
            "min_distinct_sessions": MIN_DIRECT_SESSIONS,
        },
        "changed_stable_identity_count": len(changed_stable_ids),
        "changed_turn_count": changed_turns,
        "unresolved_stable_identity_ids": [
            stable_id for stable_id in stable_ids if stable_id not in assignments
        ],
        "assignment_audit": audit_rows,
    }
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing derived artifact; the input is never modified.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if input_path == output_path:
        raise AliasResolutionError("output must differ from the immutable input")
    if output_path.exists() and not args.force:
        raise FileExistsError(output_path)
    source = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(source, dict):
        raise AliasResolutionError("input JSON root must be an object")
    resolved = resolve_document(source, source_sha256=sha256_file(input_path))
    atomic_write_json(output_path, resolved)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "output_sha256": sha256_file(output_path),
                "alias_resolver": resolved["alias_resolver"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
