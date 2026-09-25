from __future__ import annotations

import hashlib
import json
import re
import tempfile
import unittest
import urllib.request
import zipfile
from pathlib import Path
from unittest.mock import patch

from adapter import AdapterContractError, load_case_bundle, sha256_file
from build_memory import (
    MemoryContractError,
    build_case,
    normalize_fact,
)
from core.provenance import select_context
from migrate_reply_edges import migrate_reply_edges
from verify_memory import verify_memory


HERE = Path(__file__).resolve().parent
WORK_ROOT = HERE.parents[1]


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class VoxPolyR12ContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.view_dir = self.root / "case_view" / "CASE_SYNTH"
        self.registry = self.root / "online_registry_state.npz"
        with zipfile.ZipFile(self.registry, "w") as archive:
            for name in (
                "speaker_ids.npy", "assigned_turns.npy", "updates.npy", "prototypes.npy"
            ):
                archive.writestr(name, b"synthetic-registry-member")
        self._write_bundle()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _write_bundle(self, *, causal: bool = False, mutate_view=None) -> None:
        refs = ["SPK_A", "SPK_B", "SPK_C"]
        names = {"SPK_A": "Alice", "SPK_B": "Bob", "SPK_C": "Cara"}
        turns = []
        sidecar_rows = []
        for index in range(14):
            ref = refs[index % len(refs)]
            turn_id = f"S1_T{index + 1:03d}"
            turns.append({
                "turn_id": turn_id,
                "ordinal": index + 1,
                "text": f"observable statement {index + 1}",
                "timestamp": None,
                "speaker_name": names[ref],
                "speaker_id": ref,
                "speech_act": "statement",
                "interaction_id": f"I{index // 2}",
                "reply_to_turn_id": None if index == 0 else f"S1_T{index:03d}",
            })
            sidecar_rows.append({
                "session": "S1",
                "turn_idx": index,
                "turn_id": turn_id,
                "speaker_display": names[ref],
                "predicted_speaker_name": names[ref],
                "stable_identity_id": ref,
                "acoustic_speaker_id": f"ACOUSTIC_{ref}",
            })
        view = {
            "schema_version": "synthetic.v1",
            "case_id": "CASE_SYNTH",
            "theme": "contract test",
            "session_dates": {"S1": {"date": "2026-01-01"}},
            "sessions": [{"session_id": "S1", "date": "2026-01-01"}],
            "dialogues": {"S1": turns},
        }
        if mutate_view:
            mutate_view(view)
        sidecar = {
            "schema_version": "voxpoly-speaker-view.v1.sidecar",
            "case_id": "CASE_SYNTH",
            "mode": "online_predicted",
            "online_alias_is_causal_at_each_turn": causal,
            "turns": sidecar_rows,
        }
        view_path = self.view_dir / "full_case.json"
        sidecar_path = self.view_dir / "speaker_identity_sidecar.json"
        write_json(view_path, view)
        write_json(sidecar_path, sidecar)
        manifest = {
            "schema_version": "voxpoly-speaker-view.v1.manifest",
            "case_id": "CASE_SYNTH",
            "mode": "online_predicted",
            "online_alias_is_causal_at_each_turn": causal,
            "qa_free": True,
            "strict_join_key": ["session", "turn_idx"],
            "inputs": {
                "identity_predictions_sha256": "a" * 64,
                "identity_prediction_protocol": "synthetic zero-LLM online EMA",
                "identity_prediction_policy": {
                    "match_cosine_threshold": 0.35,
                    "ema_update_cosine_threshold": 0.40,
                    "ema_alpha": 0.05,
                },
            },
            "outputs": {
                "full_case_sha256": sha256_file(view_path),
                "speaker_identity_sidecar_sha256": sha256_file(sidecar_path),
            },
            "counts": {"sessions": 1, "turns": 14, "prediction_rows": 14},
            "invariants": {"qa_or_gold_loaded_into_view": False},
        }
        write_json(self.view_dir / "speaker_view_manifest.json", manifest)

    @staticmethod
    def _fake_extractor(messages: list[dict[str, str]]) -> dict:
        content = messages[-1]["content"]
        rows = re.findall(r"\[(S\d+_T\d+)\|speaker_ref=([^\]]+)\]", content)
        first_ref, first_speaker = rows[0]
        second_ref, second_speaker = rows[1]
        return {"facts": [{
            "text": f"contextual fact grounded by {first_ref}",
            "subject": "observable subject",
            "predicate": "states",
            "object": "observable value",
            "fact_type": "state",
            "source_speaker_ref": first_speaker,
            "addressee_refs": [second_speaker],
            "claimant": None,
            "participants": [],
            "valid_time": None,
            "original_time_expression": None,
            "context_operations": ["reply_resolution"],
            "refer_ids": [first_ref, second_ref],
            "image_ids": [],
        }]}

    def _build(self, *, model: str = "gpt-4.1-mini", output_name: str = "memory.json"):
        return build_case(
            case_view_dir=self.view_dir,
            registry_state=self.registry,
            output_path=self.root / output_name,
            checkpoint_dir=self.root / "checkpoints",
            model=model,
            extractor=self._fake_extractor,
            enabled=True,
        )

    def test_default_off_and_strict_causal_guards(self) -> None:
        with self.assertRaises(AdapterContractError):
            load_case_bundle(self.view_dir, self.registry)
        with self.assertRaises(AdapterContractError):
            load_case_bundle(
                self.view_dir,
                self.registry,
                enabled=True,
                require_strict_causal_aliases=True,
            )
        self._write_bundle(causal=True)
        bundle = load_case_bundle(
            self.view_dir,
            self.registry,
            enabled=True,
            require_strict_causal_aliases=True,
        )
        self.assertTrue(bundle.alias_is_causal_at_each_turn)

    def test_qa_and_hidden_addressee_are_rejected(self) -> None:
        self._write_bundle(mutate_view=lambda view: view.update({"qa": [{"answer": "leak"}]}))
        with self.assertRaises(AdapterContractError):
            load_case_bundle(self.view_dir, self.registry, enabled=True)
        self._write_bundle(
            mutate_view=lambda view: view["dialogues"]["S1"][0].update(
                {"addressee_names": ["Bob"]}
            )
        )
        with self.assertRaises(AdapterContractError):
            load_case_bundle(self.view_dir, self.registry, enabled=True)

    def test_observable_reply_edge_is_preserved(self) -> None:
        bundle = load_case_bundle(self.view_dir, self.registry, enabled=True)
        by_ref = {row["refer_ids"][0]: row for row in bundle.raw}
        self.assertIsNone(by_ref["S1_T001"]["reply_to_turn_id"])
        self.assertEqual(by_ref["S1_T002"]["reply_to_turn_id"], "S1_T001")

    def test_unknown_reply_edge_fails_closed(self) -> None:
        self._write_bundle(
            mutate_view=lambda view: view["dialogues"]["S1"][1].update(
                {"reply_to_turn_id": "S9_T999"}
            )
        )
        with self.assertRaisesRegex(AdapterContractError, "unknown bottom turn"):
            load_case_bundle(self.view_dir, self.registry, enabled=True)

    def test_missing_reply_edge_field_remains_compatible(self) -> None:
        self._write_bundle(
            mutate_view=lambda view: [
                turn.pop("reply_to_turn_id", None)
                for turn in view["dialogues"]["S1"]
            ]
        )
        bundle = load_case_bundle(self.view_dir, self.registry, enabled=True)
        self.assertTrue(all(row["reply_to_turn_id"] is None for row in bundle.raw))

    def test_legacy_memory_can_be_enriched_without_network_or_overwrite(self) -> None:
        source_path = self.root / "legacy-memory.json"
        migrated_path = self.root / "reply-aware-memory.json"
        source = self._build(output_name=source_path.name)
        for row in source["raw"]:
            row.pop("reply_to_turn_id", None)
        source["meta"]["fingerprint"]["adapter_version"] = "voxpoly-r12-adapter-v1"
        write_json(source_path, source)
        source_hash = sha256_file(source_path)

        with patch.object(urllib.request, "urlopen", side_effect=AssertionError("network called")):
            report = migrate_reply_edges(
                source_memory_path=source_path,
                case_view_dir=self.view_dir,
                registry_state=self.registry,
                output_path=migrated_path,
                enabled=True,
            )

        self.assertEqual(report["api_calls"], 0)
        self.assertEqual(report["reply_edges"], 13)
        self.assertEqual(sha256_file(source_path), source_hash)
        migrated = json.loads(migrated_path.read_text(encoding="utf-8"))
        self.assertIsNone(migrated["raw"][0]["reply_to_turn_id"])
        self.assertEqual(migrated["raw"][1]["reply_to_turn_id"], "S1_T001")
        self.assertEqual(migrated["meta"]["reply_edge_migration"]["llm_calls"], 0)

    def test_full_memory_contract_is_shared_and_zero_api(self) -> None:
        with patch.object(urllib.request, "urlopen", side_effect=AssertionError("network called")):
            memory = self._build()
        self.assertEqual(memory["meta"]["complete_windows"], 2)
        self.assertEqual(memory["meta"]["extractor_calls_this_run"], 2)
        self.assertEqual(len(memory["raw"]), 14)
        self.assertEqual(len(memory["facts"]), 2)
        self.assertEqual(len(memory["collections"]), 1)
        self.assertEqual(len(memory["profiles"]), 3)
        first = next(row for row in memory["facts"] if "S1_T001" in row["refer_ids"])
        self.assertEqual(first["source_speaker_ref"], "SPK_A")
        self.assertEqual(first["evidence_speaker_refs"], ["SPK_A", "SPK_B"])
        self.assertEqual(first["addressee_refs"], ["SPK_B"])
        self.assertEqual(first["retrieval_text"], first["text"])
        self.assertNotIn("speaker=", first["retrieval_text"])
        self.assertTrue(all(not row["retrieval_enabled"] for row in memory["profiles"]))
        self.assertNotIn("prototypes", json.dumps(memory))
        rendered = json.dumps(memory).lower()
        for token in ('"qa"', "gold_evidence", "gt_addressee"):
            if token == '"qa"':
                self.assertNotIn(token, rendered)
        self.assertFalse(memory["meta"]["qa_consumed"])
        self.assertFalse(memory["meta"]["gt_addressee_consumed"])

        # The resulting bottom nodes remain directly consumable by the shared
        # provenance selector without a VoxPoly-specific branch.
        selected, diagnostics = select_context(memory["raw"][:5], top_k=3)
        self.assertEqual(len(selected), 3)
        self.assertFalse(diagnostics["uses_gold"])

    def test_invalid_addressee_fails_closed_and_group_is_preserved(self) -> None:
        bundle = load_case_bundle(self.view_dir, self.registry, enabled=True)
        rows = list(bundle.raw[:2])
        by_ref = {row["refer_ids"][0]: row for row in rows}
        base = self._fake_extractor([
            {"role": "user", "content": (
                f"[{rows[0]['refer_ids'][0]}|speaker_ref={rows[0]['speaker_ref']}]\n"
                f"[{rows[1]['refer_ids'][0]}|speaker_ref={rows[1]['speaker_ref']}]"
            )}
        ])["facts"][0]
        base["addressee_refs"] = ["OUTSIDE_PROFILE"]
        fact = normalize_fact(base, rows_by_ref=by_ref, valid_image_ids=set())
        self.assertEqual(fact["addressee_refs"], ["unknown"])
        base["addressee_refs"] = ["group"]
        fact = normalize_fact(base, rows_by_ref=by_ref, valid_image_ids=set())
        self.assertEqual(fact["addressee_refs"], ["group"])

    def test_principal_speaker_must_be_cited(self) -> None:
        bundle = load_case_bundle(self.view_dir, self.registry, enabled=True)
        rows = list(bundle.raw[:2])
        by_ref = {row["refer_ids"][0]: row for row in rows}
        item = self._fake_extractor([
            {"role": "user", "content": (
                f"[{rows[0]['refer_ids'][0]}|speaker_ref={rows[0]['speaker_ref']}]\n"
                f"[{rows[1]['refer_ids'][0]}|speaker_ref={rows[1]['speaker_ref']}]"
            )}
        ])["facts"][0]
        item["source_speaker_ref"] = "SPK_C"
        with self.assertRaises(MemoryContractError):
            normalize_fact(item, rows_by_ref=by_ref, valid_image_ids=set())

    def test_cache_fingerprint_and_output_overwrite_fail_closed(self) -> None:
        self._build()
        with self.assertRaises(MemoryContractError):
            self._build()
        with self.assertRaises(MemoryContractError):
            self._build(model="different-model", output_name="different.json")

    def test_frozen_default_paths_match_isolation_manifest(self) -> None:
        manifest = json.loads(
            (HERE / "FROZEN_ISOLATION_MANIFEST.json").read_text(encoding="utf-8")
        )
        for relative, expected in manifest["frozen_files"].items():
            path = WORK_ROOT / relative
            self.assertTrue(path.is_file(), relative)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected, relative)

    def test_independent_memory_verifier_checks_counts_fingerprint_and_closure(self) -> None:
        memory = self._build(output_name="verified-memory.json")
        fingerprint = memory["meta"]["fingerprint"]
        preflight = self.root / "preflight.json"
        write_json(preflight, {
            "source_view_sha256": fingerprint["source_view_sha256"],
            "sidecar_sha256": fingerprint["speaker_sidecar_sha256"],
            "identity_prediction_sha256": fingerprint["identity_prediction_sha256"],
            "prediction_protocol": fingerprint["identity_prediction_protocol"],
            "registry_state_sha256": fingerprint["registry_state_sha256"],
        })
        report = verify_memory(
            self.root / "verified-memory.json",
            self.root / "checkpoints",
            preflight,
            expected_turns=14,
            expected_windows=2,
            expected_profiles=3,
        )
        self.assertEqual(report["status"], "complete")
        self.assertTrue(report["refer_closure"])
        memory["facts"][0]["refer_ids"] = ["missing"]
        write_json(self.root / "verified-memory.json", memory)
        with self.assertRaises(ValueError):
            verify_memory(
                self.root / "verified-memory.json",
                self.root / "checkpoints",
                preflight,
                expected_turns=14,
                expected_windows=2,
                expected_profiles=3,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
