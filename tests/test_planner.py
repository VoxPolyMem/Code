import json
import unittest

from core.planner import (
    heuristic_plan,
    mg_keyword_r1_plan,
    model_type_fixed_plan,
    two_level_dynamic_plan,
    unified_hybrid_plan,
)


class RouteStrategiesTest(unittest.TestCase):
    def test_heuristic_collection(self):
        plan = heuristic_plan("How many bird photos were shared?")
        self.assertEqual(plan["intent"], "VISUAL_COLLECTION")
        self.assertEqual(plan["routes"][0]["layer"], "collection")

    def test_heuristic_single_visual(self):
        plan = heuristic_plan("Which image shows the orchid?")
        self.assertEqual(plan["intent"], "VISUAL_SINGLE")
        self.assertEqual(plan["routes"][0]["layer"], "raw")

    def test_heuristic_text_uses_raw_and_fact(self):
        plan = heuristic_plan("Where did Lin Wei go last week?")
        self.assertEqual([r["layer"] for r in plan["routes"]], ["raw", "fact"])

    def test_mg_keyword_port_temporal_sort(self):
        plan = mg_keyword_r1_plan("Which happened earlier, the trip or the meeting?")
        self.assertEqual(plan["intent"], "TEMPORAL")
        self.assertEqual(plan["sort_by"], "time")

    def test_mg_keyword_port_extracts_explicit_date_without_label(self):
        plan = mg_keyword_r1_plan(
            "How many vehicle photos were shared on 2024-07-21?"
        )
        self.assertEqual(plan["time_range"], "2024-07-21")
        self.assertFalse(plan["uses_benchmark_label"])

    def test_mg_keyword_port_visual_collection(self):
        plan = mg_keyword_r1_plan("How many images show the blue flower?", True)
        self.assertEqual(plan["intent"], "VISUAL_COLLECTION")
        self.assertEqual([row["layer"] for row in plan["routes"]], ["collection", "raw"])

    def test_model_type_maps_to_frozen_route(self):
        plan = model_type_fixed_plan(
            "When was the booking changed?",
            False,
            lambda _: json.dumps({"intent": "TEMPORAL"}),
        )
        self.assertEqual(plan["intent"], "TEMPORAL")
        self.assertEqual(plan["sort_by"], "time")
        self.assertEqual(plan["routes"][0]["retrievers"], ["dense", "bm25"])

    def test_model_type_invalid_output_falls_back(self):
        plan = model_type_fixed_plan("What happened?", False, lambda _: "not-json")
        self.assertEqual(plan["intent"], "TEXT_FACT")
        self.assertFalse(plan["router_parse_ok"])

    def test_model_type_accepts_fenced_json(self):
        plan = model_type_fixed_plan(
            "What was agreed during the museum podcast episode?",
            False,
            lambda _: 'Here is the route:\n```json\n{"intent":"EVENT_SCOPED"}\n```',
        )
        self.assertEqual(plan["intent"], "EVENT_SCOPED")
        self.assertEqual(plan["routes"][0]["layer"], "collection")
        self.assertTrue(plan["router_parse_ok"])

    def test_dynamic_validates_two_levels(self):
        payload = {
            "rewritten_query": "orchid photos",
            "routes": [
                {"layer": "collection", "retrievers": ["image", "caption"]},
                {"layer": "fact", "retrievers": ["bm25", "invalid", "dense"]},
                {"layer": "raw", "retrievers": ["dense"]},
            ],
            "sort_by": "score",
        }
        plan = two_level_dynamic_plan(
            "Which orchid photos?", True, lambda _: json.dumps(payload)
        )
        self.assertEqual(len(plan["routes"]), 2)
        self.assertEqual(plan["routes"][0]["layer"], "collection")
        self.assertEqual(plan["routes"][1]["retrievers"], ["bm25", "dense"])

    def test_dynamic_invalid_output_keeps_raw_fallback(self):
        plan = two_level_dynamic_plan("What happened?", False, lambda _: "[]")
        self.assertEqual(plan["routes"], [{
            "layer": "raw",
            "retrievers": ["dense", "bm25"],
            "query": "What happened?",
        }])

    def test_dynamic_merges_duplicate_layers(self):
        payload = {
            "routes": [
                {"layer": "fact", "retrievers": ["dense"]},
                {"layer": "fact", "retrievers": ["bm25"]},
            ]
        }
        plan = two_level_dynamic_plan("What happened?", False, lambda _: json.dumps(payload))
        self.assertEqual(len(plan["routes"]), 1)
        self.assertEqual(plan["routes"][0]["retrievers"], ["dense", "bm25"])

    def test_subquestion_cap3_is_default_off_and_preserves_frozen_cap2(self):
        payload = {
            "sub_questions": ["first", "second", "third"],
            "routes": [{"layer": "raw", "retrievers": ["dense"]}],
        }
        plan = two_level_dynamic_plan(
            "What happened?", False, lambda _: json.dumps(payload)
        )
        self.assertEqual(plan["sub_questions"], ["first", "second"])

    def test_dynamic_temporal_keeps_raw(self):
        payload = {
            "routes": [{"layer": "fact", "retrievers": ["dense"]}],
            "sort_by": "time",
        }
        plan = two_level_dynamic_plan("What happened first?", False, lambda _: json.dumps(payload))
        self.assertEqual([row["layer"] for row in plan["routes"]], ["fact", "raw"])

    def test_dynamic_question_image_adds_visual_retrieval(self):
        payload = {"routes": [{"layer": "fact", "retrievers": ["dense"]}]}
        plan = two_level_dynamic_plan("What is this?", True, lambda _: json.dumps(payload))
        raw = next(row for row in plan["routes"] if row["layer"] == "raw")
        self.assertIn("image", raw["retrievers"])
        self.assertIn("caption", raw["retrievers"])

    def test_dynamic_model_visual_decision_adds_visual_retrieval(self):
        payload = {
            "needs_visual_memory": True,
            "routes": [{"layer": "fact", "retrievers": ["dense"]}],
        }
        plan = two_level_dynamic_plan(
            "What value is visible on the remembered curve?",
            False,
            lambda _: json.dumps(payload),
        )
        raw = next(row for row in plan["routes"] if row["layer"] == "raw")
        self.assertIn("image", raw["retrievers"])
        self.assertTrue(plan["model_needs_visual_memory"])

    def test_dynamic_model_event_decision_adds_collection(self):
        payload = {
            "needs_event_scope": True,
            "routes": [{"layer": "fact", "retrievers": ["dense"]}],
        }
        plan = two_level_dynamic_plan(
            "Which image caused the warning?", False, lambda _: json.dumps(payload)
        )
        self.assertIn("collection", [row["layer"] for row in plan["routes"]])
        self.assertTrue(plan["model_needs_event_scope"])

    def test_unified_hybrid_always_keeps_raw_dense_bm25(self):
        payload = {"routes": [{"layer": "collection", "retrievers": ["dense"]}]}
        plan = unified_hybrid_plan(
            "What was decided in the meeting?", False, lambda _: json.dumps(payload)
        )
        raw = next(row for row in plan["routes"] if row["layer"] == "raw")
        self.assertEqual(raw["query"], "What was decided in the meeting?")
        self.assertIn("dense", raw["retrievers"])
        self.assertIn("bm25", raw["retrievers"])
        self.assertFalse(any(row["layer"] == "fact" for row in plan["routes"]))

    def test_unified_hybrid_keeps_model_selected_specialized_fact_query(self):
        payload = {
            "routes": [{"layer": "fact", "retrievers": ["dense"],
                        "query": "specialized query"}]
        }
        question = "What happened across the four events?"
        plan = unified_hybrid_plan(question, False, lambda _: json.dumps(payload))
        fact_queries = [row["query"] for row in plan["routes"] if row["layer"] == "fact"]
        self.assertEqual(fact_queries, ["specialized query"])
        fact = next(row for row in plan["routes"] if row["layer"] == "fact")
        self.assertEqual(fact["retrievers"], ["dense", "bm25"])

    def test_unified_visual_event_adds_grounded_fact_path(self):
        payload = {
            "needs_visual_memory": True,
            "needs_event_scope": True,
            "routes": [
                {"layer": "collection", "retrievers": ["image", "caption"]}
            ],
        }
        plan = unified_hybrid_plan(
            "Which picture caused the concern?", False,
            lambda _: json.dumps(payload),
        )
        self.assertEqual(
            {row["layer"] for row in plan["routes"]},
            {"raw", "fact", "collection"},
        )
        self.assertTrue(plan["model_needs_visual_memory"])
        self.assertTrue(plan["model_needs_event_scope"])


if __name__ == "__main__":
    unittest.main()
