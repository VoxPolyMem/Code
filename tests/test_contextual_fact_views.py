import unittest

from core.contextual_fact_views import (
    deduplicate_facts,
    materialize_ablation_view,
    normalize_fact,
)


class ContextualFactViewTest(unittest.TestCase):
    def setUp(self):
        self.valid_refs = {"D1:1", "D1:2"}
        self.base = {
            "text": "Lin chose the second blue dress recommended earlier.",
            "subject": "Lin",
            "predicate": "chose",
            "object": "the second blue dress",
            "fact_type": "decision",
            "source_role": "user",
            "participants": ["Lin"],
            "context_operations": ["coreference", "reply_resolution"],
            "refer_ids": ["D1:1", "D1:2"],
            "image_ids": [],
        }

    def test_normalization_preserves_roles_and_references(self):
        fact = normalize_fact(
            self.base,
            character="Lin",
            valid_refer_ids=self.valid_refs,
            valid_image_ids=set(),
        )
        self.assertEqual(fact["speaker"], "Lin")
        self.assertEqual(fact["addressee"], "assistant")
        self.assertEqual(fact["refer_ids"], ["D1:1", "D1:2"])

    def test_ablation_views_require_no_llm_regeneration(self):
        contextual = normalize_fact(
            self.base,
            character="Lin",
            valid_refer_ids=self.valid_refs,
            valid_image_ids=set(),
        )
        single = normalize_fact(
            {**self.base, "text": "Lin likes blue.", "refer_ids": ["D1:1"],
             "context_operations": []},
            character="Lin",
            valid_refer_ids=self.valid_refs,
            valid_image_ids=set(),
        )
        facts = [contextual, single]
        self.assertEqual(len(materialize_ablation_view(facts, "full")), 2)
        self.assertEqual(len(materialize_ablation_view(facts, "no_coreference")), 1)
        self.assertEqual(len(materialize_ablation_view(facts, "single_turn_only")), 1)
        stripped = materialize_ablation_view(facts, "no_role_metadata")
        self.assertEqual(stripped[0]["retrieval_text"], stripped[0]["text"])

    def test_invalid_reference_is_rejected(self):
        with self.assertRaises(ValueError):
            normalize_fact(
                {**self.base, "refer_ids": ["D9:99"]},
                character="Lin",
                valid_refer_ids=self.valid_refs,
                valid_image_ids=set(),
            )

    def test_overlap_dedup_keeps_richer_provenance(self):
        one = normalize_fact(
            {**self.base, "refer_ids": ["D1:2"], "context_operations": []},
            character="Lin",
            valid_refer_ids=self.valid_refs,
            valid_image_ids=set(),
        )
        two = normalize_fact(
            self.base,
            character="Lin",
            valid_refer_ids=self.valid_refs,
            valid_image_ids=set(),
        )
        result = deduplicate_facts([one, two])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["refer_ids"], ["D1:1", "D1:2"])


if __name__ == "__main__":
    unittest.main()
