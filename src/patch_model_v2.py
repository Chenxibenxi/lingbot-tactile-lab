"""Generate an UNINSTALLED, isolated FlowMatchingV2 tactile state-token prototype.

Requires PYTHONPATH including this repo's src when eventually imported. This is
not evidence of training, old-checkpoint compatibility, or deployed inference.
The source checkout and its checkpoints remain untouched.
"""
from __future__ import annotations
import argparse
from pathlib import Path

SOURCE=Path('/mnt/relaydata/lingbot/lingbot-vla-v2/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py')


def once(src, old, new):
    if src.count(old)!=1:
        raise ValueError(f'unsupported V2 source anchor: {old[:70]!r}, count={src.count(old)}')
    return src.replace(old,new)


def patch_model(source):
    source=once(source,'from lingbotvla.models.loader import LingBotVLAWeightLoader',
                'from tactile_adapter import TactileStateResidual\nfrom lingbotvla.models.loader import LingBotVLAWeightLoader')
    source=once(source,'        self.state_proj = nn.Linear(self.config.max_state_dim, self.config.proj_width)',
                '''        self.state_proj = nn.Linear(self.config.max_state_dim, self.config.proj_width)
        # New adapter weights are opt-in: vision-only old checkpoints see no new keys.
        self.tactile_adapter = (TactileStateResidual(self.config.proj_width)
                                if getattr(self.config, "tactile_enabled", False) else None)''')
    source=once(source,'    def forward(\n        self,\n        images,\n        img_masks,\n        lang_tokens,\n        lang_masks,\n        state,\n        actions,',
                '''    def _tactile_delta(self, state, tactile_history, tactile_mask, tactile_age_ms, tactile_valid):
        if tactile_history is None:
            return None
        if self.tactile_adapter is None:
            raise RuntimeError("tactile inputs provided without a wired adapter")
        base = (self._fp32_linear(self.state_proj, state)
                if getattr(self.config, "action_fp32", False) else self.state_proj(state))
        return self.tactile_adapter(base, tactile_history, tactile_mask,
                                    tactile_age_ms, tactile_valid) - base

    def embed_suffix(self, state, noisy_actions, timestep, *, tactile_delta=None):
        time_emb, embs, pad, att = super().embed_suffix(state, noisy_actions, timestep)
        if tactile_delta is None:
            return time_emb, embs, pad, att
        if tactile_delta.shape != embs[:, 0].shape:
            raise ValueError("tactile state-token width/batch mismatch")
        state_token = embs[:, :1] + tactile_delta[:, None].to(dtype=embs.dtype)
        return time_emb, torch.cat([state_token, embs[:, 1:]], dim=1), pad, att

    def forward(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        actions,''')
    source=once(source,'        future_video_current_patch=None,\n    ) -> Tensor:',
                '''        future_video_current_patch=None,
        tactile_history=None,
        tactile_mask=None,
        tactile_age_ms=None,
        tactile_valid=None,
    ) -> Tensor:''')
    source=once(source,'        time_embs, suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(\n            state, x_t, time\n        )',
                '''        tactile_delta = self._tactile_delta(state, tactile_history, tactile_mask,
                                             tactile_age_ms, tactile_valid)
        time_embs, suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(
            state, x_t, time, tactile_delta=tactile_delta
        )''')
    source=once(source,'        image_grid_thw=None,\n    ) -> Tensor:',
                '''        image_grid_thw=None,
        tactile_history=None,
        tactile_mask=None,
        tactile_age_ms=None,
        tactile_valid=None,
    ) -> Tensor:''')
    source=once(source,'        while time >= -dt / 2:',
                '''        # Encode tactile once per request, not once per denoising step.
        tactile_delta = self._tactile_delta(state, tactile_history, tactile_mask,
                                             tactile_age_ms, tactile_valid)
        while time >= -dt / 2:''')
    source=once(source,'                prefix_position_ids=prefix_position_ids,\n            )\n\n            x_t += dt * v_t',
                '''                prefix_position_ids=prefix_position_ids,
                tactile_delta=tactile_delta,
            )

            x_t += dt * v_t''')
    source=once(source,'        prefix_position_ids=None,\n    ):\n        """Predict velocity at time t using cached Qwen3-VL prefix states."""',
                '''        prefix_position_ids=None,
        tactile_delta=None,
    ):
        """Predict velocity at time t using cached Qwen3-VL prefix states."""''')
    source=once(source,'            state,\n            x_t,\n            timestep,\n        )',
                '            state,\n            x_t,\n            timestep,\n            tactile_delta=tactile_delta,\n        )')
    source=once(source,'            future_video_current_patch=future_video_current_patch,\n        )',
                '''            future_video_current_patch=future_video_current_patch,
            tactile_history=kwargs.get("tactile_history"),
            tactile_mask=kwargs.get("tactile_mask"),
            tactile_age_ms=kwargs.get("tactile_age_ms"),
            tactile_valid=kwargs.get("tactile_valid"),
        )''')
    compile(source,'modeling_lingbot_vla_v2.py','exec')
    return source


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=SOURCE)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--create',action='store_true')
    args=p.parse_args()
    src=args.source.resolve(strict=True)
    out=args.out.resolve()
    if Path(__file__).resolve().parents[1]/'work' not in out.parents or src==out:
        raise ValueError('output must be a new file in this repo work/')
    generated=patch_model(src.read_text())
    if args.create:
        out.parent.mkdir(parents=True,exist_ok=True)
        with out.open('x',encoding='utf-8') as file: file.write(generated)
        print('CREATED_UNINSTALLED_MODEL_PATCH',out)
    else:
        print('DRY_RUN_AST_OK; base model unchanged',out)


if __name__=='__main__':main()
