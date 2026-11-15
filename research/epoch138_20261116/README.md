# Epoch138 evidence archive — 2026-11-16

Start with [stage review](../../docs/VTLA_EPOCH138_REVIEW_20261116_ZH.md) and [next-stage plan](../../docs/VTLA_NEXT_STAGE_PLAN_20261116_ZH.md).

Retained behavior: `cwmj_epoch138_gripensemble_v1`, w3_d85, all14 dimensions, causal same-target-time ensemble. Existing language role hold and physics checks remain. alignguard_v1/v2 are archived, disabled experiments; do not run them as the current baseline.

These are exact selected source copies, not launch authorization. Original scripts retain historical absolute work paths and refuse/reuse versioned directories; do not run them in place on existing artifacts. Reconstruction requires the private base model, simulator assets/runtime, and dependencies.json. No GPU job, training or collection is started by this archive.

Standalone CPU ensemble checks (NumPy installed):

```sh
PYTHONPATH=research/epoch138_20261116/support:research/epoch138_20261116/experiments/cwmj_epoch138_gripensemble_v1 python research/epoch138_20261116/experiments/cwmj_epoch138_gripensemble_v1/test_grip.py
PYTHONPATH=research/epoch138_20261116/support:research/epoch138_20261116/experiments/cwmj_epoch138_weighted_v1 python research/epoch138_20261116/experiments/cwmj_epoch138_weighted_v1/test_weighted.py
```

Evidence summaries are development results, not new generalization scores. Original complete videos, synchronous force tables, observations, checkpoints and logs remain local and excluded from Git. Selected source snapshots and every included file are bound in MANIFEST.json. Baseline selection is a documented fixed comparison condition, not an optimality claim.
