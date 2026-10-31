# Real relay data: matched-input ablation audit, not a model result

`src/paired_ablation.py` revalidates all five expert cards and the unapproved seed-disjoint split, then **CPU-decodes all 518 source samples** (three actual HDF5 RGB cameras, 14D command state and next 14D label) and constructs three views of each *identical* paired example:

1. `visual_only`: same RGB, instruction, state and target; zero tactile history/mask and invalid tactile flag.
2. `tactile`: same sample's untouched causally collected five-bin history and coverage mask.
3. `permuted_causal_history`: reorder only bins and fingers **within the already observed causal past**; never fetch another frame, another episode or a future force value. This is a limited chronology/assignment control, **not** independent shuffled-sample supervision.

Every variant was checked to keep the RGB `(240,320,3)` uint8 arrays for all cameras, current state and *exact same next-command target*, metadata, and strictly positive next-frame physics-step delta. The visual-only variant has zero mask/force; permutation preserves the multiset of visible past values and mask entries. Audit SHA256 for the private `data/reports/relay_five_seed_matched_ablation_audit.json` is `8a1726353e96cbce52a57301a518965625790946e30a8a3ddd11a0b3a02aa54f`; split file SHA256 is `c94d94d546bcc2bee45c78f3075b6c03b4b128f022490804ee7b89b67f44ebf5`. The raw audit stays gitignored; no RGB or action bytes enter Git.

| Fold | Independent seeds | Matched pairs | Pairs with any nonzero past proxy | Pairs changed by permutation |
|---|---:|---:|---:|---:|
| train | 3 | 309 | 73 | 73 |
| validation | 1 | 104 | 25 | 25 |
| test | 1 | 105 | 25 | 25 |
| **total** | **5** | **518** | **123** | **123** |

**Negative finding:** in **395/518** frames the two force channels in the observed past are zero, so tactile and permuted-force views are identical (apart from reordering equivalent zero bins). An aggregate imitation metric on all 518 pairs would mostly measure pre-contact imitation, not tactile utility. Any later controlled experiment must pre-register per-contact-window metrics and matched visual-only controls, and gather additional counterexamples/independent seeds before claiming improvement. Five expert-success-selected seeds and one task cannot estimate population success or generalization. This audit does **not** approve the pilot split/normalization, fit a model, load a VLA checkpoint or evaluate an online policy.
