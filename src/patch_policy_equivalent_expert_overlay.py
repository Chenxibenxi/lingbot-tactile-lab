"""Uninstalled expert overlay hook for ACTUALLY executed take_action calls.

Extends the existing source-SHA-guarded expert overlay; no rule expert method is
replaced, no dense target is reclassified as a high-level action, and nothing
runs unless an isolated future technical driver really invokes take_action.
"""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
from make_expert_overlay import patched_source as expert_overlay,replace_once
from make_overlay import create_overlay
from validate_expert_data import sha256

SOURCE=Path('/mnt/relaydata/lingbot/RoboTwin/envs/_base_task.py')
ORIGINAL_SHA='448f7152b65cb9102217cf5463aa821d72810ca56f63d5a797ec7bd43e23e101'
ANCHOR='''        if self.take_action_cnt == self.step_lim or self.eval_success:
            return

        eval_video_freq = 1  # fixed'''
REPLACEMENT='''        if self.take_action_cnt == self.step_lim or self.eval_success:
            return

        # Instrument ONLY a real original take_action execution. Existing rule
        # expert move/take_dense_action calls never enter this method.
        if (getattr(self, "_force_probe", None) is not None and
                os.environ.get("TACTILE_POLICY_EQUIVALENT_EXPERT") == "1"):
            self._force_probe.begin_motion("policy_action")
        eval_video_freq = 1  # fixed'''


def generate(original):
    if (sha256(SOURCE)!=ORIGINAL_SHA or
            hashlib.sha256(original.encode('utf-8')).hexdigest()!=ORIGINAL_SHA):
        raise ValueError('original RoboTwin base task SHA/content changed; no policy pilot patch')
    text=expert_overlay(original)
    text=replace_once(text,ANCHOR,REPLACEMENT)
    compile(text,'_base_task.py','exec')
    return text


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=SOURCE)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--create',action='store_true')
    args=p.parse_args()
    if args.source.resolve(strict=True)!=SOURCE or sha256(SOURCE)!=ORIGINAL_SHA:
        p.error('only immutable original relay RoboTwin source allowed')
    output=args.out.resolve(strict=False)
    repo=Path(__file__).resolve().parents[1]
    if repo/'work' not in output.parents or output==SOURCE or output.exists():
        p.error('output must be a NEW private overlay under this repo work/')
    if not args.create:
        generate(SOURCE.read_text())
        print('EXPERT_EQUIVALENT_OVERLAY_DRY_RUN_ONLY no execution or data approval')
        return
    print('CREATED_PRIVATE_EXPERT_CONTROLLER_OVERLAY',
          create_overlay(SOURCE.parent.parent,output,transform=generate))


if __name__=='__main__':main()
