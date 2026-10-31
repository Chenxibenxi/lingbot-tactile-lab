# Model / adapter registry

No tactile model has been trained yet. The PhiZJU vision-only 185800 checkpoint remains an external reference; do **not** copy its ~26 GB of safetensors into this repo or GitHub. Future relay-trained adapters/checkpoints go in ignored local `models/` with model config, task dataset/split manifest hashes, optimizer schedule, measured metrics, artifact URI, byte size and SHA256 recorded here. Model V1 should be a zero-initialized gated tactile adapter with the original 14D action interface preserved.

## External vision-only reference: 154800 (not tactile trained)

- Local full FP32 checkpoint: `/home/admin001/PhiZJU/submission/03_模型权重/lingbot-vla-v2-6b-lora-1f-154800/checkpoints/global_step_154800/hf_ckpt/`. Its six original shards remain in place on `/home`; no copy was added to this repository. BF16 inference is feasible on the relay RTX 3090, but the historical FP32 metrics must not be conflated with this BF16 diagnostic.
- Isolated relay `lift_pot` visual-only targeted seed `100014`: 0/1 despite pot height exceeding 0.82 m briefly; targeted seed `100048`: 1/1 with separately logged *passive* SAPIEN contact, no tactile model inputs. See `docs/STORAGE_AND_CHECKPOINT_REPLAY.md`. These two selected trials do not measure success rate.
- The locally **reconstructed** 93000 BF16 full checkpoint was removed after one failed diagnostic run per user request, along with its local link. Only the pre-existing original 93000 adapter and official base remain; the old 26/30 log does not verify reconstructed bytes or current performance.
