import unittest

from core.query_fusion import normalize_query_variants


class QueryVariantFusionTest(unittest.TestCase):
    def test_rewrites_share_one_outer_vote(self):
        channels, info = normalize_query_variants([
            ("fact.dense", 0.75, [
                {"node_id": "A"}, {"node_id": "B"},
            ]),
            ("fact.dense", 0.75, [
                {"node_id": "B"}, {"node_id": "C"},
            ]),
            ("raw.bm25", 1.0, [{"node_id": "R"}]),
        ])
        self.assertEqual(len(channels), 2)
        self.assertEqual(info["collapsed_query_votes"], 1)
        fact = next(row for row in channels if row[0] == "fact.dense")
        self.assertEqual(fact[1], 0.75)
        self.assertEqual(fact[2][0]["node_id"], "B")


if __name__ == "__main__":
    unittest.main()
