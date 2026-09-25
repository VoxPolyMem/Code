# VoxPolyBench reproduction

The full AudioMem evaluation consumes benchmark-provided transcript text while
using audio for turn-speaker identity and query-asker identity. ASR is an
interface boundary and is intentionally disabled in the reported v1 run.

## Speaker pipeline

Run `audio/speaker/online_speaker_identity_pipeline.py` per case. It performs
ECAPA encoding, online anonymous clustering, guarded EMA prototype updates,
fragment reconciliation, and name-to-cluster binding. The resulting registry
and resolved sidecar are joined to raw turns by exact session/turn coordinates.

## Memory and retrieval

The frozen implementation lives under:

```text
experiments/voxpoly_r12_integration_v1/
  versions/v2_relation_profile/
    fullcase/all18_current_audio_v1/
```

`prepare_all18_inputs.py` creates a QA-free case view. `run_memory_lane.py`
builds contextual atomic facts. `evaluate_case_iterative3.py` applies the
shared planner with Top-30 and round budgets 30, 8, and 5. A high-confidence
audio asker enables profile-grounded raw retrieval; otherwise the system uses
the global shared route. No benchmark category, answer, or gold evidence is
available to memory construction or retrieval.

All paths and Python executables are configurable with `.env`. The reported
run used 18 cases and 1,527 QA.

