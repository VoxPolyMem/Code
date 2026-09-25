import unittest

from core.temporal_contract import (
    apply_frozen_time_range,
    stable_sort_answer_visible_raw,
    temporal_contract_enabled,
)


class TemporalContractTest(unittest.TestCase):
    def test_default_off_requires_explicit_opt_in(self):
        self.assertFalse(temporal_contract_enabled({}))
        self.assertTrue(temporal_contract_enabled({"MPMEM_TEMPORAL_CONTRACT_V1": "1"}))

    def test_structured_range_prototype_is_not_active(self):
        rows = [
            {"node_id": "raw:1", "date": "2024-08-01"},
            {"node_id": "raw:2", "date": "2024-08-05"},
            {"node_id": "raw:3", "date": "2024-08-11"},
        ]
        self.assertIs(
            apply_frozen_time_range(
                rows,
                {"start": "2024-08-01", "end": "2024-08-10"},
            ),
            rows,
        )

    def test_date_list_prototype_is_not_active(self):
        rows = [{"node_id": "raw:1", "date": "2024-08-01"}]
        self.assertIs(
            apply_frozen_time_range(rows, ["2024-08-01", "2024-08-02"]),
            rows,
        )

    def test_frozen_exact_date_filter_is_retained(self):
        rows = [
            {"node_id": "raw:1", "date": "2024-08-01"},
            {"node_id": "raw:2", "date": "2024-08-05"},
            {"node_id": "raw:3", "date": "2024-08-11"},
        ]
        self.assertEqual(
            [row["node_id"] for row in apply_frozen_time_range(
                rows, "2024-08-05"
            )],
            ["raw:2"],
        )

    def test_filter_no_hit_and_free_text_return_original_list(self):
        rows = [{"node_id": "raw:1", "date": "2024-08-01"}]
        self.assertIs(apply_frozen_time_range(rows, "early August"), rows)
        self.assertIs(
            apply_frozen_time_range(rows, "2025-01-01"), rows
        )

    def test_frozen_filter_accepts_a_custom_date_getter(self):
        rows = [
            {"node_id": "raw:1", "metadata": {"date": "2024-08-01"}},
            {"node_id": "raw:2", "metadata": {"date": "2024-08-02"}},
        ]
        self.assertEqual(
            apply_frozen_time_range(
                rows, "2024-08-02",
                date_getter=lambda row: row["metadata"]["date"],
            ),
            [rows[1]],
        )

    def test_non_temporal_sort_is_object_identical(self):
        rows = [
            {"node_id": "raw:D2:2", "date": "2024-08-02"},
            {"node_id": "raw:D1:1", "date": "2024-08-01"},
        ]
        self.assertIs(
            stable_sort_answer_visible_raw(
                rows, sort_by="score", enabled=True
            ),
            rows,
        )
        self.assertIs(
            stable_sort_answer_visible_raw(
                rows, sort_by="time", enabled=False
            ),
            rows,
        )

    def test_collection_does_not_suppress_raw_chronology(self):
        collection = {
            "node_id": "collection:event",
            "layer": "collection",
            "text": "navigation only",
        }
        later = {
            "node_id": "raw:dialogue1:session2:5",
            "layer": "raw",
            "date": "2024-08-02",
            "session_id": "session2",
            "ordinal": 5,
            "image_captions": {"session2:img5.jpg": "later"},
        }
        earlier = {
            "node_id": "raw:dialogue1:session1:9",
            "layer": "raw",
            "date": "2024-08-01",
            "session_id": "session1",
            "ordinal": 9,
            "image_captions": {"session1:img9.jpg": "earlier"},
        }
        rows = [collection, later, earlier]
        result = stable_sort_answer_visible_raw(
            rows, sort_by="time", enabled=True
        )
        self.assertIs(result[0], collection)
        self.assertEqual(
            [row["node_id"] for row in result[1:]],
            [earlier["node_id"], later["node_id"]],
        )
        self.assertCountEqual(result, rows)
        self.assertIs(result[1], earlier)
        self.assertEqual(
            result[1]["image_captions"], {"session1:img9.jpg": "earlier"}
        )

    def test_same_date_uses_natural_session_and_ordinal_order(self):
        rows = [
            {"node_id": "raw:S10_T002", "date": "2024-08-01", "refer_ids": ["S10_T002"]},
            {"node_id": "raw:S2_T010", "date": "2024-08-01", "refer_ids": ["S2_T010"]},
            {"node_id": "raw:S2_T003", "date": "2024-08-01", "refer_ids": ["S2_T003"]},
        ]
        result = stable_sort_answer_visible_raw(
            rows, sort_by="time", enabled=True
        )
        self.assertEqual(
            [row["refer_ids"][0] for row in result],
            ["S2_T003", "S2_T010", "S10_T002"],
        )


if __name__ == "__main__":
    unittest.main()
