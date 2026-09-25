import json
import unittest

from core.iterative_retrieval import (
    configured_top_k,
    iterative_limits,
    iterative_retrieval_enabled,
    merge_round_contexts,
    prioritize_selected_evidence,
    retrieve_iteratively,
)


def _plan(query="original"):
    return {
        "strategy": "unified_hybrid_route",
        "rewritten_query": query,
        "routes": [{
            "layer": "raw", "retrievers": ["dense", "bm25"], "query": query,
        }],
        "sub_questions": [],
        "time_range": None,
        "speaker_hint": None,
        "sort_by": "score",
        "model_needs_visual_memory": False,
        "model_needs_event_scope": False,
        "model_needs_temporal_reasoning": False,
    }


class IterativeRetrievalTest(unittest.TestCase):
    def test_default_off_and_five_round_decreasing_budget(self):
        self.assertFalse(iterative_retrieval_enabled({}))
        self.assertTrue(iterative_retrieval_enabled({"MPMEM_ITERATIVE_RETRIEVAL_V1": "1"}))
        max_rounds, budgets = iterative_limits({
            "MPMEM_ITERATIVE_MAX_ROUNDS": "5",
            "MPMEM_ITERATIVE_ROUND_BUDGETS": "30,16,10,6,4",
        })
        self.assertEqual(max_rounds, 5)
        self.assertEqual(budgets, (30, 16, 10, 6, 4))
        self.assertEqual(configured_top_k({}), 30)
        self.assertEqual(configured_top_k({"MPMEM_FINAL_TOP_K": "20"}), 20)
        self.assertEqual(configured_top_k({"MPMEM_FINAL_TOP_K": "999"}), 60)

    def test_round_merge_deduplicates_and_allows_targeted_new_evidence(self):
        initial = [{"node_id": f"r{i}", "refer_ids": [f"t{i}"]} for i in range(30)]
        retry = [
            {"node_id": "gold", "refer_ids": ["gold-turn"]},
            {"node_id": "r0", "refer_ids": ["t0"]},
        ]
        packed, info = merge_round_contexts(
            [initial, retry], (30, 8), top_k=30
        )
        ids = [row["node_id"] for row in packed]
        self.assertIn("gold", ids)
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(info["round_stats"][1]["new_unique"], 1)
        self.assertGreaterEqual(info["deduplicated"], 1)

    def test_insufficient_round_sees_history_and_current_top30(self):
        calls = []
        responses = [
            {
                "sufficient": False,
                "reasoning": "the triggering image is absent",
                "missing_evidence": ["the image and the adjacent reaction"],
                "rewritten_query": "crowded image followed by Mia reaction",
                "routes": [{
                    "layer": "raw",
                    "retrievers": ["caption", "bm25"],
                    "query": "crowded image Mia reaction",
                }],
                "sub_questions": ["Which image shows a crowd?", "How did Mia react?"],
                "time_range": None,
                "speaker_hint": "Mia",
                "sort_by": "score",
                "needs_visual_memory": True,
                "needs_event_scope": True,
                "needs_temporal_reasoning": False,
            },
            {
                "sufficient": True,
                "reasoning": "the image and reaction are now both present",
                "missing_evidence": [],
                "supporting_memory_ids": ["gold"],
                "supporting_image_ids": ["session1:img6.jpg"],
            },
        ]

        def router(messages):
            calls.append(messages)
            return json.dumps(responses[len(calls) - 1])

        def execute(plan):
            query = plan.get("rewritten_query")
            if query == "original":
                rows = [{"node_id": "r0", "refer_ids": ["t0"], "text": "noise"}]
            else:
                rows = [{
                    "node_id": "gold", "refer_ids": ["t9"],
                    "text": "Mia was concerned by the crowd.",
                    "image_captions": {"session1:img6.jpg": "a crowded place"},
                }]
            return {
                "context": rows,
                "memory_image_candidates": rows,
                "trace": [],
                "need_memory_images": bool(query != "original"),
                "selected_collection": None,
                "expanded_refer_count": 0,
            }

        result = retrieve_iteratively(
            question="Which picture caused Mia's concern about crowds?",
            has_question_image=False,
            initial_plan=_plan(),
            execute=execute,
            call_router=router,
            environ={
                "MPMEM_ITERATIVE_MAX_ROUNDS": "5",
                "MPMEM_ITERATIVE_ROUND_BUDGETS": "30,8,5,3,2",
            },
        )
        self.assertEqual(result["iterative_retrieval"]["rounds_executed"], 2)
        self.assertEqual(result["iterative_retrieval"]["stop_reason"], "evidence_sufficient")
        self.assertEqual(len(calls), 2)
        self.assertIn("CURRENT PACKED TOP-1 EVIDENCE", calls[0][1]["content"])
        self.assertIn("PREVIOUS RETRIEVAL ACTIONS", calls[1][1]["content"])
        self.assertIn("crowded image followed by Mia reaction", calls[1][1]["content"])
        self.assertIn("do not require the memory to already contain a prewritten final summary", calls[0][0]["content"])
        self.assertIn("gold", [row["node_id"] for row in result["context"]])
        self.assertTrue(result["need_memory_images"])
        self.assertEqual(result["context"][0]["node_id"], "gold")
        self.assertEqual(
            result["iterative_retrieval"]["evidence_selector"]["matched_image_ids"],
            ["session1:img6.jpg"],
        )

    def test_selected_evidence_reorders_without_changing_membership(self):
        rows = [
            {"node_id": "noise", "image_captions": {"s1:img9.jpg": "noise"}},
            {"node_id": "gold", "image_captions": {"s1:img1.jpg": "gold"}},
        ]
        ordered, info = prioritize_selected_evidence(
            rows, ["gold"], ["s1:img1.jpg"]
        )
        self.assertEqual([row["node_id"] for row in ordered], ["gold", "noise"])
        self.assertEqual(sorted(row["node_id"] for row in ordered), ["gold", "noise"])
        self.assertTrue(info["membership_preserved"])

    def test_no_new_evidence_stops_before_runaway(self):
        def router(_messages):
            return json.dumps({
                "sufficient": False,
                "missing_evidence": ["another detail"],
                "rewritten_query": "different query",
                "routes": [{
                    "layer": "fact", "retrievers": ["dense"],
                    "query": "different query",
                }],
                "sub_questions": [],
                "time_range": None,
                "speaker_hint": None,
                "sort_by": "score",
                "needs_visual_memory": False,
                "needs_event_scope": False,
                "needs_temporal_reasoning": False,
            })

        def execute(_plan_value):
            return {
                "context": [{"node_id": "same", "refer_ids": ["t1"]}],
                "memory_image_candidates": [], "trace": [],
                "need_memory_images": False, "selected_collection": None,
                "expanded_refer_count": 0,
            }

        result = retrieve_iteratively(
            question="q", has_question_image=False, initial_plan=_plan(),
            execute=execute, call_router=router,
            environ={"MPMEM_ITERATIVE_MAX_ROUNDS": "5"},
        )
        self.assertEqual(result["iterative_retrieval"]["rounds_executed"], 2)
        self.assertEqual(result["iterative_retrieval"]["stop_reason"], "no_new_evidence")


if __name__ == "__main__":
    unittest.main()
