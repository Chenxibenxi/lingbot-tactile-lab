"""Generate an UNINSTALLED, fail-closed LingBotV2 server tactile routing preview.

The real deployed server is untouched. The preview keeps the ORIGINAL model
import: no approved tactile adapter exists, and nothing sets capability_ready
true. In that state incoming tactile keys are rejected rather than silently
ignored, while visual-only behavior keeps the original call/transform path.
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path
from validate_expert_data import sha256

SOURCE=Path('/mnt/relaydata/lingbot/lingbot-vla-v2/deploy/lingbot_vla_v2_policy.py')
SOURCE_SHA256='2caf39d560892ccefa0415dd9df3d4c4beab4a0ef4be40a996e5277756b52413'


def once(source,old,new,*,count=1):
    if source.count(old)!=count:
        raise ValueError(f'unsupported deployment source anchor: {old[:65]!r}, count={source.count(old)}')
    return source.replace(old,new)


def patch_server(source):
    source=once(source,
        'from lingbotvla.models.vla.lingbot_vla.modeling_lingbot_vla_v2 import LingbotVlaV2Policy',
        'from lingbotvla.models.vla.lingbot_vla.modeling_lingbot_vla_v2 import LingbotVlaV2Policy\n'
        'from server_tactile_bridge import prepare_input, model_tactile_kwargs')
    source=once(source,
        "        observation =  self.vla.feature_transform.apply(observation, policy_eval=True)",
        "        observation = prepare_input(observation, self.vla.feature_transform,\n"
        "            tactile_enabled=getattr(self, '_tactile_capability_ready', False))")
    source=once(source,
        "            image_grid_thw=self._to_device_image_grid_thw(observation.get('image_grid_thw'), device),",
        "            image_grid_thw=self._to_device_image_grid_thw(observation.get('image_grid_thw'), device),\n"
        "            **model_tactile_kwargs(observation, device=device, dtype=dtype, batched=False),")
    source=once(source,
        '                            image_grid_thw=self._to_device_image_grid_thw(image_grid_thw, device),',
        '                            image_grid_thw=self._to_device_image_grid_thw(image_grid_thw, device),\n'
        '                            **model_tactile_kwargs(observation, device=device, dtype=dtype, batched=True),',
        count=3)
    ast.parse(source,filename='lingbot_vla_v2_policy_tactile_preview.py')
    return source


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=SOURCE)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--create',action='store_true')
    args=p.parse_args()
    source=args.source.resolve(strict=True)
    out=args.out.resolve()
    repo=Path(__file__).resolve().parents[1]
    if source!=SOURCE or sha256(source)!=SOURCE_SHA256 or repo/'work' not in out.parents or out==source:
        p.error('requires exact read-only source and a private repo work/ preview output')
    generated=patch_server(source.read_text())
    if not args.create:
        print('DRY_RUN_SOURCE_AST_ONLY; original server and real model unchanged')
        return
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x',encoding='utf-8') as file:file.write(generated)
    print('CREATED_UNINSTALLED_SERVER_PREVIEW',out)


if __name__=='__main__':main()
