# Pre-registered, single-run simulated open-grasp negative control

**Simulation-only; owner explicitly approved one isolated trial.** Existing five `pick_diverse_bottles` expert successes have plentiful continuing contact but no true grasp-miss counterexample; seed100001 failed during scene initialization, and an older lift-pot seed0 failed planning before finger contact. None proves what the policy should do after a failed grasp.

## Fixed intervention and isolation

- Use the **already expert-successful** relay scene seed `100000`, same `pick_diverse_bottles` task, source RoboTwin and demo_clean camera/robot settings, but create a fresh **different** local directory `data/negative/relay_pickdiverse_s100000_open_grasp1` and run at most once. It is a **counterfactual trial**, not an independent seed; it must never cross or count toward the five-seed split.
- After successful task setup, temporarily wrap only this private robot instance's `grasp_actor` so its two calls command `gripper_pos=1.0` (stay open) instead of default closed `0.0`. The source methods `move`, lift, place, `check_success` and original RoboTwin file bytes remain unchanged. This explicitly alters actuator commands **in simulation**, not hardware safety limits.
- Accept the outcome as a controlled negative only if the grasp hook ran **exactly twice**, the motion planner still succeeded and the unmodified goal check **failed**. If planner fails, the task unexpectedly succeeds, ForceProbe fails, images mismatch or contact telemetry is incomplete, retain failure as a *different diagnostic*, not a grasp-miss proof.
- Persist a **negative-only** SHA gate before HDF5/video conversion, then validate same-run raw/obs/frame/three-camera HDF5 and hash-bound `negative_completion.json`; never write the expert-positive names `expert_success_gate.json` or `completion.json`, never use `datasets/cards/`, and never train on this without a separate scientific/quality review. All RGB/HDF5/pickle/video/action bytes stay in ignored local `data/negative/`.

## GPU and deadline

Before one bounded 660-second trial: acquire this relay 3090's interactive lease, confirm no other compute process and Strata still stopped at the owner's request. Release the lease even if the trial fails. Do **not** restart Strata before the already scheduled **2026-11-02 12:00 Asia/Shanghai** restoration; that restoration must itself yield if a different GPU job is active. Unit tests and dry-run run first. This produces at most one controlled counterexample, not a failure-rate estimate or evidence that tactile conditioning improves a real policy.
