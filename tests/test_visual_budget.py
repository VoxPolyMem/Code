import unittest

from core.visual_budget import prioritize_visual_rows


class VisualBudgetTest(unittest.TestCase):
    def test_visual_rank_controls_payload_without_leaving_context(self):
        context = [
            {"node_id": "A", "image_paths": ["a.jpg"]},
            {"node_id": "B", "image_paths": ["b.jpg"]},
            {"node_id": "C", "image_paths": ["c.jpg"]},
        ]
        visual = [
            {"node_id": "C", "image_paths": ["c.jpg"]},
            {"node_id": "X", "image_paths": ["x.jpg"]},
            {"node_id": "B", "image_paths": ["b.jpg"]},
        ]
        rows = prioritize_visual_rows(visual, context, lambda row: row["node_id"])
        self.assertEqual([row["node_id"] for row in rows], ["C", "B", "A"])


if __name__ == "__main__":
    unittest.main()
