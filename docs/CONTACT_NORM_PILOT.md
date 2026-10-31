# Positive-contact, zero-preserving normalization candidate

**CPU-only research candidate; neither normalization nor tactile model training has been approved.** The underlying same-run relay-only expert dataset and seed-disjoint pilot split are detailed in `RELAY_FIVE_SEED_PILOT.md`. This changes no original RoboTwin/LingBot files and uses no GPU.

## Why the first candidate is unsuitable

In the **train** fold (seeds 100000/100003/100014; 309 causal pairs), each finger has 1,545 complete 5-bin observations, but only 348–356 have positive mean/peak contact-proxy signal (22.5–23.0%). Fitting all bins made all 12 medians zero and the 25–75% spread zero, triggering the arbitrary `0.1` IQR floor even for the four contact-fraction indicators. A raw `log1p(12 N)/0.1` could create an out-of-scale input from routine contact.

## Opt-in v2 algorithm and train-only diagnostic

`src/contact_normalization.py` fits median and IQR of `log1p` **only over strictly positive force bins** among complete `coverage_mask==1` training samples; absent force maps to **exact zero**, and observed contact fraction is left in `[0,1]` with median 0, scale 1. `src/tactile_adapter.py` implements this as a separate persisted `norm_mode=1` and bounds the two force channels after normalization to `[-4,+4]`, without changing default `norm_mode=0` or zero-init vision identity. The train-only bound is an experimental hyperparameter, not an actuator limit.

The real-fold candidate in `configs/relay_lift_pot_contact_norm_candidate_v2.json` has positive log-force medians ≈2.53–2.58 per finger/channel, but **even positive-only IQR <0.1** on all eight force channels, so their scale still uses the floor 0.1. This is a near-constant contact signal, not proof of informative force magnitudes. On training bins, applying the **exact zero-preserving transform** would clip ~4.76% of positive force components (or ~1.08% of all components); it leaves all absent-force components exactly zero. This fraction is a training-fold diagnostic, not a held-out metric. The initial diagnostic that subtracted the positive median from zero-force bins and yielded ~78% clipping was **the wrong transform**, was discarded, and must not be used in reports.

Both v1 and v2 pilot configs remain `training_approval=false` and are rejected by `TactileStateResidual.set_normalization`. The 5-seed split remains `candidate_not_approved`; validation/test examples were never used to fit these statistics or choose the bound. Before training/claiming gains, review this v2 contract, verify contact-positive distribution under additional seed/object conditions, set up a matched-image visual-only baseline plus shuffled/masked-tactile negative controls, and only then record a controlled adapter experiment. SAPIEN impulse/dt is an uncalibrated contact proxy, never a hardware force threshold.
