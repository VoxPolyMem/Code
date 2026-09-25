import unittest

from core.provenance import select_context


class ProvenanceSelectorTest(unittest.TestCase):
    def test_keeps_bounded_support_then_prefers_new_evidence(self):
        candidates = [
            {"node_id": "collection:S1", "memory_layer": "collection",
             "refer_ids": ["T1", "T2", "T3", "T4"]},
            {"node_id": "raw:T1", "refer_ids": ["T1"]},
            {"node_id": "raw:T2", "refer_ids": ["T2"]},
            {"node_id": "raw:T3", "refer_ids": ["T3"]},
            {"node_id": "raw:T5", "refer_ids": ["T5"]},
            {"node_id": "raw:T6", "refer_ids": ["T6"]},
        ]
        selected, info = select_context(candidates, top_k=4, support_quota=2)
        ids = [row["node_id"] for row in selected]
        self.assertEqual(ids[:3], ["collection:S1", "raw:T1", "raw:T2"])
        self.assertEqual(ids[3], "raw:T5")
        self.assertEqual(info["support_rows"], 2)
        self.assertFalse(info["uses_gold"])

    def test_deduplicates_same_memory_id(self):
        selected, info = select_context([
            {"node_id": "raw:T1", "refer_ids": ["T1"]},
            {"node_id": "raw:T1", "refer_ids": ["T1"]},
        ])
        self.assertEqual(len(selected), 1)
        self.assertEqual(info["candidate_count"], 1)

    def test_upper_node_can_be_retrieval_only_scaffold(self):
        candidates = [
            {"node_id": "collection:S1", "memory_layer": "collection",
             "refer_ids": ["T1", "T2"]},
            {"node_id": "raw:T1", "refer_ids": ["T1"]},
            {"node_id": "raw:T2", "refer_ids": ["T2"]},
            {"node_id": "raw:T3", "refer_ids": ["T3"]},
        ]
        selected, info = select_context(
            candidates, top_k=3, support_quota=2, expose_upper=False
        )
        self.assertEqual(
            [row["node_id"] for row in selected],
            ["raw:T1", "raw:T2", "raw:T3"],
        )
        self.assertEqual(info["upper_scaffold_hidden"], 1)
        self.assertEqual(info["selected_count"], 3)



if __name__ == "__main__":
    unittest.main()
