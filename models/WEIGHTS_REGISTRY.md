> 2026-11-06 最终更新：修复加载缓冲区和推理一致性后，v3 已完成三组×120步真实训练及 seed100030 成对仿真。独立进程输出完全一致；离线未通过预登记收益标准，闭环两组均失败，尚未证明触觉提升。v1 效果解释作废，v2 加载阶段中止。结果见 `docs/TACTILE_EFFECT_20261106_RESULTS_ZH.md`。

# Model / adapter registry

As of 2026-11-06, three real 65,664-parameter adapters have completed the bounded terminal-H50 pilot (120 AdamW steps each). This proves training/in-process conditioning, not a task-success improvement or RPC deployment. The PhiZJU vision-only 185800 checkpoint remains an external reference; do **not** copy its ~26 GB of safetensors into this repo or GitHub. Future relay-trained adapters/checkpoints go in ignored local `models/` with model config, task dataset/split manifest hashes, optimizer schedule, measured metrics, artifact URI, byte size and SHA256 recorded here. Model V1 should be a zero-initialized gated tactile adapter with the original 14D action interface preserved.

## External vision-only reference: 154800 (not tactile trained)

- Local full FP32 checkpoint: `/home/admin001/PhiZJU/submission/03_模型权重/lingbot-vla-v2-6b-lora-1f-154800/checkpoints/global_step_154800/hf_ckpt/`. Its six original shards remain in place on `/home`; no copy was added to this repository. BF16 inference is feasible on the relay RTX 3090, but the historical FP32 metrics must not be conflated with this BF16 diagnostic.
- Isolated relay `lift_pot` visual-only targeted seed `100014`: 0/1 despite pot height exceeding 0.82 m briefly; targeted seed `100048`: 1/1 with separately logged *passive* SAPIEN contact, no tactile model inputs. See `docs/STORAGE_AND_CHECKPOINT_REPLAY.md`. These two selected trials do not measure success rate.
- The locally **reconstructed** 93000 BF16 full checkpoint was removed after one failed diagnostic run per user request, along with its local link. Only the pre-existing original 93000 adapter and official base remain; the old 26/30 log does not verify reconstructed bytes or current performance.

## 2026-11-06 terminal-H50 matched pilot (private artifacts)

Base: immutable 154800, 1,624 keys strict-loaded; train seed100023; validation100026/100027. Fixed last-step checkpoints, no validation selection. See `docs/TACTILE_EFFECT_PILOT_V1.md` and `data/reports/tactile_effect_pilot_v1/result.json`.

| Arm | Local adapter | SHA256 | Steps |
|---|---|---|---|
| zero | `/mnt/relaydata/lingbot-tactile-lab/models/tactile_effect_pilot_v1/zero/adapter.safetensors` | `585b3df466af2893456850f419f3e3c7ee3df086e18af210fe574713ed92db69` | 120 |
| true | `/mnt/relaydata/lingbot-tactile-lab/models/tactile_effect_pilot_v1/true/adapter.safetensors` | `3ecac96555f66c389f987c47555529bef5fd89758b70aae8310d5bab1e489cc6` | 120 |
| reverse | `/mnt/relaydata/lingbot-tactile-lab/models/tactile_effect_pilot_v1/reverse/adapter.safetensors` | `7248e3f0424c555656ce08ba903ea49ef67e28d9bd89c77a436b4335a4e308ce` | 120 |

Optimizer state and full normalization/source/protocol metadata are stored beside each adapter. No full baseline copies or private raw data were uploaded. The preregistered offline support criterion was **not met**.


## Corrected v3 pilot (buffer restore + reproducible MoE path)

Independent-process full-action exact equality passed before fitting. Same data and optimizer settings as invalidated v1. No predeclared benefit demonstrated.

| Arm | Artifact | SHA256 | Steps |
|---|---|---|---|
| zero | `/mnt/relaydata/lingbot-tactile-lab/models/tactile_effect_pilot_v3/zero/adapter.safetensors` | `cbb631892bf2214e083528e7e1f4c1213c268b5fa99b5aa7e513171212b5732c` | 120 |
| true | `/mnt/relaydata/lingbot-tactile-lab/models/tactile_effect_pilot_v3/true/adapter.safetensors` | `13c2250c0a610a8bb259ad47c4c01dcf59e40fa78eaa347e9e635269980480e8` | 120 |
| reverse | `/mnt/relaydata/lingbot-tactile-lab/models/tactile_effect_pilot_v3/reverse/adapter.safetensors` | `30ef62dae3df8b2742e4551239fb6c1aa1a98166328056cff3c85feec42e3544` | 120 |

Details: `docs/TACTILE_EFFECT_20261106_RESULTS_ZH.md`. v1 rows above are invalidated historical artifacts, not deployment candidates.
