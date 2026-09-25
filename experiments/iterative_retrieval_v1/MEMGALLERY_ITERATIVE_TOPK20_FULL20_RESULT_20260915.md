# Mem-Gallery iterative TopK20 full20 result (2026-09-15)

## Completion and fixed evaluation contract

The iterative-retrieval candidate completed all 20 Mem-Gallery topics and all
1,711 QA rows at 2026-09-15T21:10:30+08:00. It used frozen r12 contextual
memory and frozen initial route plans; `gpt-4.1-mini` through Velen performed
the iterative sufficiency, answer, and judge calls. Retrieval used a TopK20
final packed context, a maximum of five total retrieval rounds, decreasing
new-evidence admission budgets, and no benchmark labels, answers, or gold
evidence in routing.

## Full20 micro result

| System | QA | LLM-score | Acc@0.75 | Paired W/T/L |
|---|---:|---:|---:|---:|
| frozen r4 TopK30 | 1,711 | 0.854909 | 0.866160 | -- |
| frozen r12 contextual TopK30 | 1,711 | 0.855348 | 0.870836 | -- |
| iterative TopK20 | 1,711 | **0.860023** | **0.869082** | 121/1,454/136 vs r4 |

On the exactly aligned 1,711 keys, iterative TopK20 improves LLM-score over
r4 by **0.005114** and Acc@0.75 by **0.002922**. It improves LLM-score over
the frozen r12 parent by **0.004676**, while its Acc@0.75 is lower by
0.001753. The effect is small and should be reported as a full-benchmark
increment rather than a broad claim that every type improves.

## Per-type result of iterative TopK20

| Type | QA | LLM-score | Acc@0.75 |
|---|---:|---:|---:|
| AR | 184 | 0.8886 | 0.8750 |
| CD | 81 | 0.6790 | 0.6790 |
| FR | 219 | 0.8881 | 0.9041 |
| KR | 81 | 0.8333 | 0.8395 |
| MR | 206 | 0.8701 | 0.9272 |
| TR | 123 | 0.8923 | 0.8943 |
| TTL | 337 | 0.8947 | 0.9110 |
| VR | 174 | 0.7629 | 0.7586 |
| VS | 306 | 0.8750 | 0.8660 |
| **Micro overall** | **1,711** | **0.8600** | **0.8691** |

The largest positive paired score differences versus r4 are VR (+0.0876), CD
(+0.0247), AR (+0.0149), and TR (+0.0142). KR (-0.0370) and MR (-0.0170)
remain the main regressions to analyze.

## Retrieval and cost

- Iterative rounds beyond the initial round: 0 for 1,233 questions, 1 for
  426, 2 for 49, and 3 for 3. No question reached the five-round cap.
- Successful model calls: 5,125.
- Total tokens: 14,228,366.
- Observed Velen cost: **CNY 44.7785**.

Primary result files are under
`experiments/iterative_retrieval_v1/results/memgallery_topk20_full20/` and
the resumable execution log is
`experiments/iterative_retrieval_v1/memgallery_topk20_full20.log`.
