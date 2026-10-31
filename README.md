# LingBot tactile lab

Independent, **private** LingBot-VLA × RoboTwin research repository for gripper force/contact sensing, same-run visuo-tactile datasets, model-input training, inference and controlled-feedback evaluation. The end goal is to **modify and improve** the train/eval pipeline, retaining the existing vision-only 185800 run as an immutable reference—not to stop at read-only logging.

**Repository status:** owner-reviewed `research/tactile-foundation` was pushed to the exact private `Chenxibenxi/lingbot-tactile-lab` remote; local and remote commit `271612c` matched, and remote Contents read passed. No `main` branch was pushed/created; GitHub made the only existing research branch its default. The token exposed earlier in chat was never used and should be revoked; the new token was entered only via a masked local dialog. Raw HDF5, private recordings and weights remain local. See the [Chinese research report](docs/TACTILE_RESEARCH_REPORT_ZH.md).

## Single-machine provenance

All **new** raw tactile data, calibrated parameters, model checkpoints and evaluations must be generated on the relay workstation's **single RTX 3090**, stored locally in `data/` and `models/`. Historic LXC samples from PhiZJU E009–E011 demonstrated feasibility, but **are explicitly excluded from this repo's training datasets and reported relay benchmark**. The relay's existing RoboTwin copy is `/mnt/relaydata/lingbot/RoboTwin`; original vision-only HDF5s are reference input only and lack physical timestamps.

> GPU gate: the relay 3090 normally hosts Strata. Five successful expert captures have now run under bounded interactive leases, only after Strata reported no active/queued tasks; its loaded baseline was restored after each batch. Never preempt an active workload or run a GPU job without fresh occupancy and lease checks.

## Layout and status

| Path | Role | Current status |
|---|---|---|
| `src/force_probe.py`, `src/make_overlay.py`, `src/make_expert_overlay.py` | SAPIEN per-finger impulse telemetry and isolated RoboTwin overlay | Relay expert seed100048 captured and per-finger telemetry verified |
| `src/record_expert_demo.py`, `src/validate_expert_data.py`, `src/run_expert_relay.sh`, `src/strata_idle_gate.py` | Same-run expert capture plus source-verified Strata busy/queued gate | Five lift-pot + five diverse-bottle successful candidates; Strata stopped per owner until scheduled noon restoration |
| `src/align_temporal.py`, `src/build_paired_index.py`, `src/register_dataset.py` | Causal 25-step×5-bin×4-finger alignment, 14D next-frame labels and **relay-only source verification** | Ten unassigned candidate cards across two tasks, 1,112 usable pairs; **not approved for model training** |
| `src/paired_samples.py`, `src/paired_ablation.py`, `src/tactile_adapter.py` | Same-run samples, matched RGB/label three-view input audit and gated tactile state residual | All 518 pairs CPU-decoded; only 123 contain nonzero past force; **no model trained** |
| `src/split_and_norm.py`, `src/contact_normalization.py`, `src/plan_relay_training.py` | Task-filtered seed-disjoint pilots and train-only v1/v2 contact statistics | Two separate five-seed 3/1/1 pilot splits; every normalization remains an **unapproved candidate** |
| `src/contact_weight_candidate.py`, `src/train_label_diagnostic.py`, `src/contact_stratified_eval.py` | Train-only contact weight/next-command timing diagnostics and preregistered full-coverage scorer | Training labels vary by real physical interval; weights still unapproved; **no model trained or scored** |
| `src/tactile_rpc.py` | Opt-in causal tactile payload and actual RoboTwin msgpack transport contract | CPU roundtrip verified; **not yet wired into running client/server** |
| `src/window_math.py`, `src/live_tactile_window.py`, `src/shadow_forward.py` | Shared offline/live causal window and physics→RPC→state-token shadow pathway | CPU parity-tested; **not a deployed VLA inference** |
| `src/patch_policy_client.py` | Reversible, isolated client-source generator; requires `tactile_v1` server capability before attaching a true infer-call marker | AST/unit checked only; **old server cannot be used as tactile inference** |
| `src/server_tactile_bridge.py` | Strict transform/batch preprocessing and conditional server capability metadata | CPU mocks tested; **no real VLA server or trained adapter loaded** |
| `src/patch_model_v2.py` | Isolated strict-source generator for optional FlowMatchingV2 state-token residual | CPU AST/identity tests only; **not loaded with a real VLA checkpoint** |
| `src/patch_safe_eval_launcher.py` | Private visual-policy eval launcher that checks curobo paths without writing shared assets or conda `.pth` | Local 154800 BF16 seed100048 success **with passive contact only**; zero tactile-model inference |
| `src/validate_policy_trace.py` | Audit real infer-attempt markers against camera captures and causal physics windows | Relay success trace passed; **diagnostic only, missing trainable image/action HDF5** |
| `src/policy_rollout_capture.py`, `src/patch_policy_capture_client.py`, `src/validate_policy_rollout.py` | Opt-in actual infer RGB/causal tactile and **executed** 14D commands; HDF5-to-force/camera/request validator | 154800/seed100048 relay capture passed, candidate SHA card; **self-rollout, not expert labels or approved training** |
| `src/contact_targets.py`, `src/expert_contact_target_audit.py` | Future-only contact labels bounded by executed policy actions or verified expert next-frame intervals | Train3 lift-pot expert seeds offer just three new-onset intervals; **no auxiliary-head training approved** |
| `src/register_negative_diagnostic.py` | SHA-bound, relay-only rejected expert telemetry kept outside the success registry | One planner failure: 1,044 steps, 77 markers, no contact/HDF5; **not a grasp-failure training example** |
| `src/controlled_negative.py`, `src/record_controlled_negative.py`, `src/validate_controlled_negative.py` | One opt-in simulated open-grasp intervention, negative-only HDF5/outcome gate | Same seed100000: planner succeeds, goal fails; initial three-camera RGB matches, contact falls; **quarantined, not expert training** |
| `tests/` | Offline unit/contract regressions | 94 tests passing |
| `configs/` | Planned experimental capture/window parameters and model-study registry | Initial settings, not tuned |
| `docs/` | Temporal constraints, controlled baseline and research plan | Draft research assets |
| `data/paired/`, `data/negative/`, `data/reports/` | Relay-generated bytes only, **local and gitignored** | Ten positive expert candidates separate from a simulated controlled negative; no raw bytes in Git |
| `models/` | Local weights/adapters (gitignored); registry tracks hashes | Await tactile training |
| `work/` | Generated overlay and scratch, gitignored | Create only for a specific trial |

## First reproducible CPU checks

```bash
cd /mnt/relaydata/lingbot-tactile-lab
PYTHONPATH="$PWD/src" /mnt/relaydata/lingbot/conda/envs/lingbot-eval310/bin/python -B \
  -m unittest discover -s tests -p 'test_*.py' -v
PYTHONPATH="$PWD/src" python3 -B src/make_expert_overlay.py \
  --source /mnt/relaydata/lingbot/RoboTwin --out "$PWD/work/overlay_expert"  # dry-run
```

Inspect the relay runner with `bash src/run_expert_relay.sh --dry-run relay_liftpot_s100009_expert0 lift_pot 100009`. A real trial additionally requires a freshly verified idle GPU, an interactive GPU lease, Strata's atomic unload/stop only when it has no requests, and restoration plus lease release; the runner alone does **not** acquire a lease. New trial IDs cannot reuse previous outputs. Each successful episode must pass `src/validate_expert_data.py` and receive a hash-bound completion manifest before becoming a candidate. Old launcher `start_robotwin_infer_and_eval.sh` may write back through symlinks and modify conda `.pth`; do not invoke it on an overlay without isolating those writable paths.

## Where this goes next

1. Relay-only paired demonstrations, actual per-frame physics ticks and manifest SHA256 (no fabricated FPS alignment).
2. The CPU paired-sample reader, ≥5-actual-seed split gate, train-only candidate normalization and zero-init adapter are prepared. Five relay expert seeds meet only a technical pilot split gate; candidate preprocessing remains explicitly unapproved. Gather more independent contact-rich/failure episodes, review preprocessing, and compare matched-data vision-only versus tactile models before training claims. Labels remain 14D with real physical steps, not synthetic 50fps timestamps.
3. Opt-in tactile encoder and gated adapter in LingBot training **and policy RPC**, with sensor-absent fallback; train and evaluate real use of tactile via shuffled-signal negative control.
4. Independent low-level gripper feedback, separate safety bounds, a 2×2 model × feedback ablation, multiple contact-rich tasks and reproducible results.

[Temporal/data contract](docs/TIMELINE_DESIGN.md) · [Model integration evidence and next wiring](docs/MODEL_INTEGRATION.md) · [Research roadmap](docs/RESEARCH_PLAN.md) · [Local data policy](datasets/README.md).

The paper motivating the research is [Yuan et al., VTAM](https://arxiv.org/abs/2603.23481) with Ismini Lourentzou. Its GelSight images and deformation-derived virtual force are not equivalent to our SAPIEN impulse/dt proxy.
