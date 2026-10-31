# Future contact onset has only three training positives

`src/expert_contact_target_audit.py` revalidates all five relay expert candidate cards and the pinned actual-seed split, then reads **only the three training trials'** original per-physics-step telemetry. For each usable current frame it examines the genuinely subsequent interval `(capture_physics_step, next_frame_physics_step]`, with no interpolation and a strict no-missing-step gate. The observed 25-step history at the current frame is input; future interval contact is **a possible label only, never a feature or gripper safety threshold**. Each side's prior status is computed from its own two fingers and valid history mask. No validation/test interval target was accessed for this design decision.

| Train actual seed | Usable next-frame pairs | Left/right any future-contact intervals | Left/right newly contacting from no prior side contact | Both hands first contact in same interval |
|---:|---:|---:|---:|---:|
| 100000 | 102 | 25 / 25 | 1 / 1 | 1 |
| 100003 | 104 | 26 / 26 | 1 / 1 | 1 |
| 100014 | 103 | 25 / 25 | 1 / 1 | 1 |
| **Total** | **309** | **76 / 76** | **3 / 3** | **3** |

A naive next-frame **contact-onset auxiliary classifier would have only three independent positive training intervals**, one per expert seed, while both hands onset together in this selected `lift_pot` setting. Its apparent accuracy could be achieved by always predicting no onset. Even the 76 positive any-contact intervals mainly reflect already maintained grasp and should not be presented as distinct successes or independent trials. These are support-count diagnostics, not a trained model or a recommendation to expose future force to inference. The aggregate card `datasets/relay_train_future_contact_support_v1.json` remains diagnostic, with candidate split and all normalization/weight approvals still false. Collect more diverse task and failed/near-contact examples before selecting an auxiliary target or claiming anticipation of touch.
