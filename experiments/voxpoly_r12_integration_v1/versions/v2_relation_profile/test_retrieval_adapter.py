from __future__ import annotations

import inspect
import unittest

try:
    from .retrieval_adapter import (
        RetrievalContractError,
        relation_profile_context,
        select_semantic_anchors,
    )
except ImportError:
    from retrieval_adapter import (
        RetrievalContractError,
        relation_profile_context,
        select_semantic_anchors,
    )


def make_memory() -> dict:
    raw = []
    for number in range(1, 41):
        session = "S2" if number == 40 else "S1"
        ordinal = 2 if number == 40 else number
        row = {
            "node_id": f"raw:T{number:03d}",
            "layer": "raw",
            "text": f"generic turn number {number}",
            "retrieval_text": f"generic turn number {number}",
            "speaker_ref": "SPK_B",
            "session_id": session,
            "ordinal": ordinal,
            "refer_ids": [f"T{number:03d}"],
            "reply_to_turn_id": None,
        }
        raw.append(row)
    raw[28]["text"] = raw[28]["retrieval_text"] = "The cobalt folder is assigned tomorrow"
    raw[28]["speaker_ref"] = "SPK_A"
    raw[30]["text"] = raw[30]["retrieval_text"] = "I accepted the cobalt assignment"
    raw[30]["reply_to_turn_id"] = "T029"
    raw[34]["text"] = raw[34]["retrieval_text"] = "Cobalt status belongs to the requester"
    raw[34]["speaker_ref"] = "SPK_A"
    raw[39]["text"] = raw[39]["retrieval_text"] = "wrong session nearby ordinal"
    facts = [
        {
            "node_id": "fact:cobalt-reply",
            "layer": "fact",
            "text": "The cobalt folder assignment was accepted",
            "retrieval_text": "The cobalt folder assignment was accepted",
            "source_speaker_ref": "SPK_B",
            "addressee_refs": ["SPK_A"],
            "refer_ids": ["T029", "T031"],
        },
        {
            "node_id": "fact:cobalt-profile",
            "layer": "fact",
            "text": "The requester owns the cobalt status",
            "retrieval_text": "The requester owns the cobalt status",
            "source_speaker_ref": "SPK_A",
            "addressee_refs": ["SPK_B"],
            "refer_ids": ["T035"],
        },
        {
            "node_id": "fact:different-event",
            "layer": "fact",
            "text": "A different event has another state",
            "retrieval_text": "A different event has another state",
            "source_speaker_ref": "SPK_A",
            "addressee_refs": ["SPK_B"],
            "refer_ids": ["T040"],
        },
    ]
    return {
        "raw": raw,
        "facts": facts,
        "profiles": [
            {"stable_speaker_id": "SPK_A", "refer_ids": ["T029", "T035"]},
            {"stable_speaker_id": "SPK_B", "refer_ids": ["T001"]},
        ],
    }


class RetrievalAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.memory = make_memory()
        self.by_id = {row["node_id"]: row for row in self.memory["raw"]}
        self.base = [dict(row) for row in self.memory["raw"][:30]]
        # The semantic rank can place a raw turn outside the old R12 context.
        self.global_rank = [self.by_id["raw:T029"]] + [
            row for row in self.memory["raw"] if row["node_id"] != "raw:T029"
        ][:39]

    def test_public_contract_has_no_gold_or_category_argument(self) -> None:
        parameters = set(inspect.signature(relation_profile_context).parameters)
        self.assertFalse(parameters & {"gold", "answer", "category", "question_type"})

    def test_exact_quote_anchor_and_reply_packet_preserve_raw_prefix(self) -> None:
        result, trace = relation_profile_context(
            self.base,
            self.global_rank,
            self.memory,
            'Who responded to "The cobalt folder is assigned tomorrow"?',
            relation_budget=4,
            profile_state_budget=0,
        )
        ids = [row["node_id"] for row in result]
        self.assertEqual(len(ids), 30)
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue({row["node_id"] for row in self.base[:24]}.issubset(ids))
        self.assertIn("raw:T029", ids)  # old tail is structurally protected
        self.assertIn("raw:T031", ids)  # incoming reply/fact sibling
        self.assertNotIn("raw:T040", ids)  # same ordinal, different session
        self.assertGreaterEqual(trace["protected_base_tail_count"], 1)
        self.assertTrue(trace["raw_only_output"])
        self.assertTrue(all(len(row["refer_ids"]) == 1 for row in result))

    def test_quoted_gate_does_not_expand_unmatched_text(self) -> None:
        anchors = select_semantic_anchors('Who said "an entirely absent quotation"?', self.global_rank)
        self.assertEqual(anchors, [])

    def test_high_identity_selects_state_but_low_only_reorders(self) -> None:
        high, high_trace = relation_profile_context(
            self.base,
            self.global_rank,
            self.memory,
            "What item is relevant to the requester?",
            query_identity={"stable_asker_ref": "SPK_A", "confidence": "high"},
            ranked_facts=[self.memory["facts"][1]],
            relation_budget=0,
            profile_state_budget=2,
        )
        self.assertIn("raw:T035", {row["node_id"] for row in high})
        self.assertTrue(high_trace["uses_query_identity"])

        low, low_trace = relation_profile_context(
            self.base,
            self.global_rank,
            self.memory,
            "What item is relevant to the requester?",
            query_identity={"stable_asker_ref": "SPK_A", "confidence": "low"},
            ranked_facts=[self.memory["facts"][1]],
            relation_budget=0,
            profile_state_budget=2,
        )
        self.assertEqual(
            {row["node_id"] for row in low},
            {row["node_id"] for row in self.base},
        )
        self.assertTrue(low_trace["low_confidence_soft_reorder"])

    def test_unknown_identity_fails_closed(self) -> None:
        with self.assertRaises(RetrievalContractError):
            relation_profile_context(
                self.base,
                self.global_rank,
                self.memory,
                "What is the cobalt status?",
                query_identity={"stable_asker_ref": "SPK_Z", "confidence": "high"},
            )

    def test_profile_state_prefers_semantic_event_neighborhood(self) -> None:
        result, _ = relation_profile_context(
            self.base,
            self.global_rank,
            self.memory,
            "Which item remains pertinent?",
            query_identity={"stable_asker_ref": "SPK_A", "confidence": "high"},
            # The wrong-session fact is deliberately ranked first.  The
            # correct-session fact wins through QA-blind raw event coherence.
            ranked_facts=[self.memory["facts"][2], self.memory["facts"][1]],
            relation_budget=0,
            profile_state_budget=1,
        )
        ids = {row["node_id"] for row in result}
        self.assertIn("raw:T035", ids)
        self.assertNotIn("raw:T040", ids)


if __name__ == "__main__":
    unittest.main()
