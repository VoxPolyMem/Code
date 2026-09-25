import unittest

from core.evidence_bundle import (
    evidence_bundle_active,
    evidence_bundle_enabled,
    make_evidence_group,
    organize_evidence_bundles,
)


class EvidenceBundleTest(unittest.TestCase):
    def test_flag_is_default_off(self):
        self.assertFalse(evidence_bundle_enabled({}))
        self.assertTrue(evidence_bundle_enabled({"MPMEM_EVIDENCE_BUNDLE_V1": "1"}))

    def test_soft_gate_uses_model_route_decisions(self):
        env = {"MPMEM_EVIDENCE_BUNDLE_SOFT_GATE_V1": "1"}
        active, reason = evidence_bundle_active({
            "sub_questions": ["a", "b"],
            "model_needs_event_scope": True,
        }, env)
        self.assertTrue(active)
        self.assertEqual(reason, "model_cross_turn_decomposition")
        active, reason = evidence_bundle_active({
            "sub_questions": ["a", "b"],
            "model_needs_visual_memory": True,
        }, env)
        self.assertFalse(active)
        self.assertEqual(reason, "model_gate_rejected")
        active, reason = evidence_bundle_active({
            "sub_questions": ["a", "b"],
            "model_needs_event_scope": True,
            "model_needs_temporal_reasoning": True,
        }, env)
        self.assertFalse(active)
        self.assertEqual(reason, "global_temporal_order_preserved")

    def test_group_rejects_single_and_broad_provenance(self):
        self.assertIsNone(make_evidence_group("f1", ["t1"], source_rank=1))
        self.assertIsNone(make_evidence_group(
            "f1", [f"t{i}" for i in range(7)], source_rank=1
        ))

    def test_membership_is_preserved_and_group_is_chronological(self):
        group = make_evidence_group("f1", ["t1", "t2"], source_rank=1)
        rows = [
            {"node_id": "r2", "refer_ids": ["t2"], "date": "2026-02-02",
             "ordinal": 2, "evidence_groups": [group]},
            {"node_id": "noise", "refer_ids": ["n"], "date": "2026-01-01"},
            {"node_id": "r1", "refer_ids": ["t1"], "date": "2026-02-02",
             "ordinal": 1, "evidence_groups": [group]},
        ]
        bundled, diagnostics = organize_evidence_bundles(rows, enabled=True)
        self.assertEqual([row["node_id"] for row in bundled], ["r1", "r2", "noise"])
        self.assertEqual(sorted(row["node_id"] for row in bundled),
                         sorted(row["node_id"] for row in rows))
        self.assertEqual(diagnostics["selected_groups"], 1)
        self.assertTrue(diagnostics["membership_preserved"])
        self.assertIn("Evidence bundle 1", bundled[0]["evidence_bundle_header"])

    def test_overlapping_groups_do_not_duplicate_rows(self):
        g1 = make_evidence_group("f1", ["t1", "t2"], source_rank=1)
        g2 = make_evidence_group("f2", ["t2", "t3"], source_rank=2)
        rows = [
            {"node_id": "r1", "evidence_groups": [g1]},
            {"node_id": "r2", "evidence_groups": [g1, g2]},
            {"node_id": "r3", "evidence_groups": [g2]},
        ]
        bundled, _ = organize_evidence_bundles(rows, enabled=True)
        self.assertEqual(len(bundled), 3)
        self.assertEqual(len({row["node_id"] for row in bundled}), 3)


if __name__ == "__main__":
    unittest.main()
