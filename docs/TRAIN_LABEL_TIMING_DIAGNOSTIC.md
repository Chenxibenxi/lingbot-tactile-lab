# Train-only label/timing diagnostic — no fitted model

The same-run relay expert dataset uses **actual physical capture steps**, not a synthetic fixed 50 fps clock. `src/train_label_diagnostic.py` revalidates the five candidate expert cards and pinned seed split but reads actual command labels **only for the three training trials** (309 pairs); held-out validation/test labels are not consulted to choose a training objective. It compares the observed 14D current commanded state with the **next saved frame's 14D expert command** and groups examples by strictly observed past tactile contact, never by future force or goal success.

| Training stratum | Pairs | Pairs with any command component change > 1e-5 | Mean absolute 14D command change |
|---|---:|---:|---:|
| Current observed prior contact | 73 | 52 | 0.005345 |
| No prior contact | 236 | 202 | 0.013521 |

Only **283/309** adjacent training pairs are exactly 15 physics steps apart (~60 ms for dt≈4 ms); the other 26 have step intervals 1–14. Thus a packaged 50 fps timestamp must not be used as the physical action horizon. The mean 14D next-command change is roughly **2.5× smaller while contact is already observed** than before contact. This is descriptive, **not** evidence contact is useless: a stable grasp can demand little commanded movement while still relying on touch for safety or correction. Conversely, a training signal consisting solely of next saved-frame command imitation may be dominated by approach/transport frames; pre-register contact-window and per-joint/gripper metrics and gather failure/perturbation counterexamples before an improvement claim.

The absolute value of the pooled 14D number mixes arm-joint and gripper command scales; it is **not a calibrated physical unit, control latency, force limit, or online task-success score**. Existing status is `train_fold_diagnostic_only_no_model_fit`, and the parent split/norm/weights remain unapproved. Reproduce counts from `datasets/relay_train_label_diagnostic_v1.json` (a small sanitized aggregate); no raw per-frame action trajectory is committed.
