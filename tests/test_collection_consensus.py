import unittest

from core.collection_consensus import rerank_collections


class CollectionConsensusTest(unittest.TestCase):
    def test_small_cross_layer_bonus_breaks_ambiguous_collection_rank(self):
        collections = [
            {"node_id": "C-wrong", "score": 0.065, "refer_ids": ["X"]},
            {"node_id": "C-right", "score": 0.063, "refer_ids": ["A", "B"]},
        ]
        evidence = [
            {"node_id": "raw:A", "refer_ids": ["A"]},
            {"node_id": "raw:B", "refer_ids": ["B"]},
            {"node_id": "raw:X", "refer_ids": ["X"]},
        ]
        ranked = rerank_collections(collections, evidence, support_weight=0.25)
        self.assertEqual(ranked[0]["node_id"], "C-right")
        self.assertEqual(ranked[0]["cross_layer_support_ranks"], [1, 2])

    def test_no_provenance_leaves_base_order(self):
        collections = [
            {"node_id": "C1", "score": 0.2, "refer_ids": []},
            {"node_id": "C2", "score": 0.1, "refer_ids": []},
        ]
        self.assertEqual(
            [row["node_id"] for row in rerank_collections(collections, [])],
            ["C1", "C2"],
        )


if __name__ == "__main__":
    unittest.main()
