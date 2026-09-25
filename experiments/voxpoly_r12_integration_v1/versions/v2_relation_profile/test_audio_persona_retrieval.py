from __future__ import annotations

import inspect
import unittest

try:
    from .audio_persona_adapter import (
        annotate_raw_relation_context,
        speaker_profile_prior_raw_context,
    )
except ImportError:
    from audio_persona_adapter import (
        annotate_raw_relation_context,
        speaker_profile_prior_raw_context,
    )


def memory() -> dict:
    raw = [
        {
            "node_id": f"raw:T{index:03d}",
            "text": f"turn {index}",
            "refer_ids": [f"T{index:03d}"],
            "speaker_ref": "SPK_A" if index in {1, 2} else "SPK_B",
        }
        for index in range(1, 41)
    ]
    return {
        "raw": raw,
        "facts": [],
        "profiles": [
            {"node_id": "profile:SPK_A", "stable_speaker_id": "SPK_A", "refer_ids": ["T001", "T002"]},
            {"node_id": "profile:SPK_B", "stable_speaker_id": "SPK_B", "refer_ids": ["T003"]},
        ],
    }


class AudioPersonaRetrievalTests(unittest.TestCase):
    def test_profile_prior_keeps_global_evidence_and_top30_budget(self) -> None:
        store = memory()
        dense = list(store["raw"])
        bm25 = list(reversed(store["raw"]))
        result, trace = speaker_profile_prior_raw_context(
            question="What did I agree to do?",
            query_identity={"stable_asker_ref": "SPK_A", "confidence": "high"},
            memory=store,
            dense_raw=dense,
            bm25_raw=bm25,
        )
        ids = [row["node_id"] for row in result]
        self.assertEqual(len(ids), 30)
        self.assertEqual(len(ids), len(set(ids)))
        self.assertIn("raw:T001", ids)
        self.assertTrue(any(row["speaker_ref"] == "SPK_B" for row in result))
        self.assertEqual(trace["route"], "raw_dense_bm25_rrf_with_profile_provenance_prior")

    def test_public_contract_is_qa_blind(self) -> None:
        forbidden = {"answer", "gold", "category", "question_type"}
        self.assertFalse(forbidden & set(inspect.signature(speaker_profile_prior_raw_context).parameters))

    def test_relation_annotation_uses_only_unanimous_grounded_fact_votes(self) -> None:
        store = memory()
        store["facts"] = [
            {
                "refer_ids": ["T001"], "addressee": "Bob",
                "addressee_refs": ["SPK_B"], "source_speaker_ref": "SPK_A",
            },
            {
                "refer_ids": ["T002"], "addressee": "Bob",
                "addressee_refs": ["SPK_B"], "source_speaker_ref": "SPK_A",
            },
            {
                "refer_ids": ["T002"], "addressee": "Cara",
                "addressee_refs": ["SPK_C"], "source_speaker_ref": "SPK_A",
            },
            {
                "refer_ids": ["T003"], "addressee": "Bob",
                "addressee_refs": ["SPK_B"], "source_speaker_ref": "SPK_A",
            },
        ]
        rows = annotate_raw_relation_context(store, store["raw"][:3])
        self.assertEqual(rows[0]["relation_addressee"], "Bob")
        self.assertEqual(
            rows[0]["relation_label_source"],
            "grounded_fact_source_speaker_unanimous",
        )
        self.assertEqual(rows[1]["relation_addressee"], "unknown")
        self.assertEqual(rows[1]["relation_label_source"], "grounded_fact_ambiguous")
        self.assertEqual(rows[2]["relation_addressee"], "unknown")


if __name__ == "__main__":
    unittest.main()
