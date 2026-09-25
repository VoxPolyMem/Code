import os
import unittest
from pathlib import Path
from unittest.mock import patch

from core.temporal_contract import (
    apply_frozen_time_range,
    stable_sort_answer_visible_raw,
    temporal_contract_enabled,
)


def legacy_apply_time_range(rows, value):
    """Exact pre-CTEC H2H helper, retained here only as a differential oracle."""
    if not isinstance(value, str) or len(value) != 10:
        return rows
    filtered = [row for row in rows if str(row.get("date") or "") == value]
    return filtered or rows


class H2HR12Full190DefaultOffTest(unittest.TestCase):
    def test_flag_must_be_absent_and_absence_is_off(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertNotIn("MPMEM_TEMPORAL_CONTRACT_V1", os.environ)
            self.assertFalse(temporal_contract_enabled())

    def test_shared_date_helper_matches_legacy_contract(self):
        rows = [
            {"node_id": "raw:1", "date": "2024-08-01"},
            {"node_id": "raw:2", "date": "2024-08-05"},
            {"node_id": "raw:3", "date": ""},
        ]
        values = [
            None,
            "",
            "early August",
            "2024-08-01",
            "2024-08-03",
            ["2024-08-01", "2024-08-05"],
            {"start": "2024-08-01", "end": "2024-08-05"},
        ]
        for value in values:
            with self.subTest(value=value):
                expected = legacy_apply_time_range(rows, value)
                actual = apply_frozen_time_range(rows, value)
                self.assertEqual(actual, expected)
                if expected is rows:
                    self.assertIs(actual, rows)

    def test_candidate_sort_is_identity_when_disabled(self):
        rows = [
            {"node_id": "raw:2", "date": "2024-08-02", "ordinal": 2},
            {"node_id": "raw:1", "date": "2024-08-01", "ordinal": 1},
        ]
        self.assertIs(
            stable_sort_answer_visible_raw(
                rows, sort_by="time", enabled=False
            ),
            rows,
        )

    def test_h2h_executor_keeps_the_legacy_else_branch(self):
        workspace = Path(__file__).resolve().parents[1]
        if not (workspace / "evaluation" / "h2h.py").is_file():
            workspace = workspace / "source"
        source = (workspace / "evaluation" / "h2h.py").read_text(encoding="utf-8")
        self.assertIn("temporal_contract = temporal_contract_enabled()", source)
        self.assertIn("if temporal_contract:", source)
        self.assertIn(
            'elif plan.get("sort_by") == "time" and not node:', source
        )
        self.assertIn("apply_frozen_time_range(ranked, time_range)", source)


if __name__ == "__main__":
    unittest.main()
