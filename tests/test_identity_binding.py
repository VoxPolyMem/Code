from __future__ import annotations

import unittest

from core.identity_binding import (
    bind_addressee_to_candidates,
    bind_speaker_from_references,
)


ROWS = {
    "r1": {"speaker": "Una"},
    "r2": {"speaker": "Mia"},
    "r3": {"speaker": "Una"},
}


class IdentityBindingTest(unittest.TestCase):
    def test_single_cited_speaker_repairs_unknown_without_adding_evidence(self):
        result = bind_speaker_from_references("unknown", ["r1", "r3"], ROWS)
        self.assertEqual(result.value, "Una")
        self.assertEqual(result.source, "single_cited_speaker")
        self.assertEqual(result.candidates, ("Una",))

    def test_multiple_cited_speakers_require_a_grounded_model_choice(self):
        result = bind_speaker_from_references("@mia", ["r1", "r2"], ROWS)
        self.assertEqual(result.value, "Mia")
        self.assertEqual(result.source, "model_grounded_multi_speaker")

    def test_multiple_cited_speakers_reject_unknown(self):
        with self.assertRaisesRegex(ValueError, "not one of cited speakers"):
            bind_speaker_from_references("unknown", ["r1", "r2"], ROWS)

    def test_unknown_reference_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown refer_ids"):
            bind_speaker_from_references("Una", ["missing"], ROWS)

    def test_addressee_binding_fails_closed(self):
        rows = [
            ("@Una", "Una", "model_grounded_candidate"),
            ("everyone", "group", "model_group"),
            ("unknown", "unknown", "model_unknown"),
            ("not-in-window", "unknown", "unresolved_candidate"),
        ]
        for proposed, expected, source in rows:
            with self.subTest(proposed=proposed):
                result = bind_addressee_to_candidates(proposed, ["Una", "Mia"])
                self.assertEqual(result.value, expected)
                self.assertEqual(result.source, source)


if __name__ == "__main__":
    unittest.main()
