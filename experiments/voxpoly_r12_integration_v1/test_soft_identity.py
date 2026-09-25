from __future__ import annotations

import hashlib
import json
import tempfile
import types
import unittest
import urllib.request
from pathlib import Path
from unittest.mock import patch

from evaluate_matched_panel import (
    PersistentCallCache,
    main as evaluation_main,
    run_panel,
    sha256_json,
)
from make_matched_panel import build_manifest
from soft_identity import (
    MAX_IDENTITY_BOOST,
    resolve_speaker_hint,
    soft_identity_rerank,
)
from verify_matched_panel import verify


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


class SoftIdentityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.profiles = [
            {
                "node_id": "profile:SPK_A",
                "stable_speaker_id": "SPK_A",
                "speaker_name": "Alice Chen",
                "aliases": ["Alice Chen", "Alice"],
            },
            {
                "node_id": "profile:SPK_B",
                "stable_speaker_id": "SPK_B",
                "speaker_name": "Bob",
                "aliases": ["Bob"],
            },
            {
                "node_id": "profile:SPK_C",
                "stable_speaker_id": "SPK_C",
                "speaker_name": "Alice Park",
                "aliases": ["Alice Park"],
            },
        ]
        self.context = [
            {
                "node_id": f"raw:r{index:02d}",
                "layer": "raw",
                "text": f"memory {index}",
                "refer_ids": [f"r{index:02d}"],
            }
            for index in range(30)
        ]
        self.memory = {
            "raw": self.context,
            "profiles": self.profiles,
            "facts": [
                {
                    "node_id": "fact:source",
                    "source_speaker_ref": "SPK_A",
                    "addressee_refs": ["SPK_B"],
                    "refer_ids": ["r04"],
                },
                {
                    "node_id": "fact:addressee",
                    "source_speaker_ref": "SPK_B",
                    "addressee_refs": ["SPK_A"],
                    "refer_ids": ["r07"],
                },
            ],
        }

    def test_hint_resolution_is_exact_or_unique_and_ambiguous_fails_closed(self) -> None:
        self.assertEqual(
            resolve_speaker_hint("SPK_A", self.profiles).speaker_refs, ("SPK_A",)
        )
        self.assertEqual(
            resolve_speaker_hint("Bob", self.profiles).speaker_refs, ("SPK_B",)
        )
        # Alice is explicitly listed as an exact alias for SPK_A, so exact
        # identity wins even though another full name begins with Alice.
        self.assertEqual(
            resolve_speaker_hint("Alice", self.profiles).speaker_refs, ("SPK_A",)
        )
        profiles = [dict(row, aliases=[row["speaker_name"]]) for row in self.profiles]
        self.assertEqual(
            resolve_speaker_hint("Alice", profiles).speaker_refs, ()
        )

    def test_feature_off_and_unresolved_hint_are_exact_identity(self) -> None:
        for enabled, hint in ((False, "Alice Chen"), (True, "Nobody"), (True, None)):
            output, trace = soft_identity_rerank(
                self.context, {"speaker_hint": hint}, self.memory, enabled=enabled
            )
            self.assertEqual(
                [row["node_id"] for row in output],
                [row["node_id"] for row in self.context],
            )
            self.assertEqual(trace["changed_positions"], 0)

    def test_soft_boost_reorders_only_and_preserves_raw_safety_set(self) -> None:
        output, trace = soft_identity_rerank(
            self.context, {"speaker_hint": "Alice Chen"}, self.memory, enabled=True
        )
        original_ids = [row["node_id"] for row in self.context]
        output_ids = [row["node_id"] for row in output]
        self.assertEqual(set(output_ids), set(original_ids))
        self.assertEqual(len(output_ids), len(original_ids))
        self.assertGreater(trace["changed_positions"], 0)
        self.assertEqual(trace["boosted_candidates"], 2)
        self.assertTrue(trace["candidate_set_preserved"])
        self.assertFalse(trace["uses_benchmark_label"])
        self.assertFalse(trace["uses_gold"])
        for row in trace["boost_details"]:
            self.assertLessEqual(row["boost"], MAX_IDENTITY_BOOST)
            self.assertTrue(set(row["reasons"]).issubset(
                {"source_speaker_ref", "addressee_refs"}
            ))


class OrchestrationContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_panel_selection_is_invariant_to_labels_answers_and_gold(self) -> None:
        ids = [f"Q{index:03d}" for index in range(30)]
        case_a = self.root / "a.json"
        case_b = self.root / "b.json"
        write_json(case_a, {"qa": {"qa_pairs": [
            {"qa_id": value, "category": "A", "answer": "one", "gold": ["r1"]}
            for value in ids
        ]}})
        write_json(case_b, {"qa": {"qa_pairs": [
            {"qa_id": value, "category": "DIFFERENT", "answer": "two", "gold": ["r9"]}
            for value in ids
        ]}})
        selected_a = build_manifest(case_a, "CASE", 10, "seed")["qa_ids"]
        selected_b = build_manifest(case_b, "CASE", 10, "seed")["qa_ids"]
        self.assertEqual(selected_a, selected_b)

    def test_exact_message_cache_is_finite_and_reuses_success(self) -> None:
        calls = 0

        def caller(messages, **kwargs):
            nonlocal calls
            calls += 1
            if calls < 3:
                raise RuntimeError("transient")
            return "ok"

        cache = PersistentCallCache(
            self.root / "cache.json",
            caller,
            model="gpt-4.1-mini",
            max_attempts=3,
            retry_delay_seconds=0,
        )
        messages = [{"role": "user", "content": "hello"}]
        self.assertEqual(cache.call(messages), "ok")
        self.assertEqual(cache.call(messages), "ok")
        self.assertEqual(calls, 3)

    def test_paid_evaluator_is_default_off_before_opening_inputs(self) -> None:
        with patch.object(urllib.request, "urlopen", side_effect=AssertionError("network called")):
            code = evaluation_main([
                "--memory", str(self.root / "missing-memory.json"),
                "--panel", str(self.root / "missing-panel.json"),
                "--out-dir", str(self.root / "out"),
            ])
        self.assertEqual(code, 2)
        self.assertFalse((self.root / "out").exists())

    def test_verifier_accepts_only_matched_same_set_outputs(self) -> None:
        case_path = self.root / "case.json"
        write_json(case_path, {"qa": {"qa_pairs": [{"qa_id": "Q1"}]}})
        panel = build_manifest(case_path, "CASE", 1, "seed")
        panel_path = self.root / "panel.json"
        write_json(panel_path, panel)
        out = self.root / "result"
        route = {"speaker_hint": "Alice", "routes": []}
        route_sha = sha256_json(route)
        base_arm = {
            "prediction": "x",
            "score": 1.0,
            "judge": {"score": 1.0},
            "route_plan_sha256": route_sha,
            "context_node_ids": ["raw:r1", "raw:r2"],
            "retrieved_refer_ids": ["r1", "r2"],
            "evidence_recall_at_30": 1.0,
            "n_context": 2,
        }
        document = {
            "schema_version": "voxpoly-soft-identity-matched.v1",
            "panel_manifest_sha256": hashlib.sha256(panel_path.read_bytes()).hexdigest(),
            "memory_sha256": "unused",
            "config": {
                "model": "gpt-4.1-mini",
                "strategy": "unified_hybrid_route",
                "top_k": 30,
                "identity_intervention": "order_only_soft_boost",
                "candidate_set_must_match": True,
                "uses_benchmark_label_for_retrieval": False,
                "uses_gold_for_retrieval": False,
            },
            "pairs": [{
                "qa_id": "Q1",
                "question": "q",
                "reference_answer": "a",
                "gold_evidence": ["r1"],
                "route_plan": route,
                "route_plan_sha256": route_sha,
                "content_only": {
                    **base_arm,
                    "identity_trace": {
                        "uses_question_text": False,
                        "uses_benchmark_label": False,
                        "uses_answer": False,
                        "uses_gold": False,
                        "candidate_set_preserved": True,
                        "changed_positions": 0,
                    },
                },
                "soft_identity": {
                    **base_arm,
                    "context_node_ids": ["raw:r2", "raw:r1"],
                    "identity_trace": {
                        "uses_question_text": False,
                        "uses_benchmark_label": False,
                        "uses_answer": False,
                        "uses_gold": False,
                        "candidate_set_preserved": True,
                        "changed_positions": 2,
                        "resolved_speaker_refs": ["SPK_A"],
                        "boost_details": [{
                            "node_id": "raw:r2",
                            "boost": 0.02,
                            "reasons": ["source_speaker_ref"],
                        }],
                    },
                },
            }],
        }
        write_json(out / "matched_pairs.json", document)
        report = verify(out, panel_path, require_complete=True)
        self.assertEqual(report["status"], "complete")
        self.assertTrue(report["same_raw_recall_set"])
        document["pairs"][0]["soft_identity"]["context_node_ids"] = ["raw:r3"]
        write_json(out / "matched_pairs.json", document)
        with self.assertRaises(ValueError):
            verify(out, panel_path, require_complete=True)

    def test_complete_matched_orchestrator_with_fake_services(self) -> None:
        case_path = self.root / "case.json"
        write_json(case_path, {"qa": {"qa_pairs": [{
            "qa_id": "Q1",
            "question_text": "Who made the decision?",
            "answer": "Alice",
            "evidence": ["r04"],
            "category": "MUST_NOT_REACH_RERANK",
        }]}})
        panel = build_manifest(case_path, "CASE", 1, "seed")
        panel_path = self.root / "panel.json"
        write_json(panel_path, panel)
        memory = {
            "meta": {"identity_retrieval_enabled": False},
            "raw": [
                {"node_id": f"raw:r{index:02d}", "refer_ids": [f"r{index:02d}"],
                 "text": f"memory {index}", "speaker": "speaker", "date": "2026-01-01"}
                for index in range(30)
            ],
            "facts": [{
                "source_speaker_ref": "SPK_A",
                "addressee_refs": ["SPK_B"],
                "refer_ids": ["r04"],
            }],
            "profiles": [{
                "stable_speaker_id": "SPK_A",
                "speaker_name": "Alice",
                "aliases": ["Alice"],
            }],
        }
        memory_path = self.root / "memory.json"
        write_json(memory_path, memory)

        adapters_module = types.ModuleType("adapters.voxpoly")
        adapters_module.qa_pairs = lambda case: case["qa"]["qa_pairs"]
        adapters_module.gold_evidence = lambda qa: list(qa.get("evidence") or [])
        planner_module = types.ModuleType("core.planner")

        def plan(question, has_image, router):
            router([{"role": "user", "content": "router"}])
            return {
                "strategy": "unified_hybrid_route",
                "router_parse_ok": True,
                "uses_benchmark_label": False,
                "speaker_hint": "Alice",
                "routes": [{"layer": "raw", "retrievers": ["dense", "bm25"],
                            "query": question}],
            }

        planner_module.unified_hybrid_plan = plan
        prompts_module = types.ModuleType("eval.bench_prompts")
        prompts_module.answer_messages = lambda q, c, **kwargs: [
            {"role": "user", "content": "answer:" + ",".join(x["node_id"] for x in c)}
        ]
        prompts_module.judge_messages = lambda q, a, p: [
            {"role": "user", "content": "judge:" + p}
        ]
        evaluation_module = types.ModuleType("evaluation.voxpoly")

        class FakeIndex:
            def __init__(self, path):
                self.rows = json.loads(path.read_text())["raw"]

            def execute(self, plan_value, top_k):
                return self.rows[:top_k], [{"channel": "fake"}], None

        evaluation_module.LayerIndex = FakeIndex
        evaluation_module.question_text = lambda qa: qa["question_text"]
        calls = []

        def fake_llm(messages, **kwargs):
            calls.append(messages)
            content = messages[0]["content"]
            if content == "router":
                return "{}"
            if content.startswith("judge:"):
                return '{"score": 1.0, "reasoning": "synthetic"}'
            return "Alice"

        with patch.dict("sys.modules", {
            "adapters.voxpoly": adapters_module,
            "core.planner": planner_module,
            "eval.bench_prompts": prompts_module,
            "evaluation.voxpoly": evaluation_module,
        }), patch.object(
            urllib.request, "urlopen", side_effect=AssertionError("network called")
        ):
            document = run_panel(
                memory_path=memory_path,
                panel_path=panel_path,
                out_dir=self.root / "matched",
                model="gpt-4.1-mini",
                max_attempts=2,
                retry_delay_seconds=0,
                call_llm=fake_llm,
                extract_json=json.loads,
            )
        self.assertEqual(len(document["pairs"]), 1)
        pair = document["pairs"][0]
        self.assertEqual(
            set(pair["content_only"]["context_node_ids"]),
            set(pair["soft_identity"]["context_node_ids"]),
        )
        self.assertGreater(pair["soft_identity"]["identity_trace"]["changed_positions"], 0)
        self.assertNotIn("MUST_NOT_REACH_RERANK", json.dumps(
            pair["soft_identity"]["identity_trace"]
        ))
        report = verify(self.root / "matched", panel_path, require_complete=True)
        self.assertEqual(report["status"], "complete")


if __name__ == "__main__":
    unittest.main(verbosity=2)
