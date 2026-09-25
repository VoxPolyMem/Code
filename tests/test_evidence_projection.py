import unittest

from core.evidence_projection import project_grounding


class EvidenceProjectionTest(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {
                "node_id": f"raw:{i}", "refer_ids": [f"T{i}"],
                "session_id": "S1", "ordinal": i,
                "image_captions": ({"S1:img.jpg": "image"} if i == 1 else {}),
            }
            for i in range(4)
        ]
        self.by_ref = {row["refer_ids"][0]: row for row in self.rows}
        self.by_image = {"S1:img.jpg": [self.rows[1]]}
        self.by_position = {("S1", row["ordinal"]): row for row in self.rows}

    def test_projects_direct_media_and_one_hop_neighbors(self):
        upper = {"refer_ids": ["T2"], "images": ["S1:img.jpg"]}
        projected = project_grounding(
            upper, self.by_ref, self.by_image, self.by_position, neighbor_radius=1
        )
        relations = {row["node_id"]: relation for row, _, relation in projected}
        self.assertEqual(set(relations), {"raw:1", "raw:2", "raw:3"})
        self.assertEqual(relations["raw:1"], "media")
        self.assertEqual(relations["raw:2"], "direct")
        self.assertEqual(relations["raw:3"], "adjacent")


if __name__ == "__main__":
    unittest.main()
