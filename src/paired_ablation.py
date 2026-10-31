"""CPU-only same-data visual/tactile/within-history permutation study contract.

No model training, rollout or causal future is available to these views.
A permuted five-bin past history is a weak chronology control, not a proof
of tactile utility or a replacement for held-out controlled model comparisons.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from paired_samples import CAMERAS,PairedEpisode
from plan_relay_training import build_plan

MODES=('visual_only','tactile','permuted_causal_history')


def make_view(sample, mode):
    if mode not in MODES:raise ValueError(f'unknown ablation mode {mode}')
    inp=sample['input'];target=sample['target'];meta=sample['metadata']
    if 'action_command_14' in inp or 'tactile_future' in inp:
        raise ValueError('target or future tactile information leaked into input')
    history=np.asarray(inp['tactile_history']);mask=np.asarray(inp['tactile_mask'])
    if history.shape!=(5,4,3) or mask.shape!=(5,4):
        raise ValueError('unexpected causal tactile window geometry')
    if not np.isfinite(history).all() or not np.isfinite(mask).all():
        raise ValueError('nonfinite tactile view')
    view={'input':dict(inp),'target':dict(target),'metadata':dict(meta)}
    if mode=='visual_only':
        view['input']['tactile_history']=np.zeros_like(history)
        view['input']['tactile_mask']=np.zeros_like(mask)
        view['input']['tactile_window_valid']=False
        view['input']['tactile_age_ms']=np.float32(0.)
    elif mode=='tactile':
        view['input']['tactile_history']=history.copy()
        view['input']['tactile_mask']=mask.copy()
    else:
        # Reorder ONLY already observed bins/fingers; never fetch another
        # frame, another episode or an unknown future physics step.
        view['input']['tactile_history']=history[::-1,::-1,:].copy()
        view['input']['tactile_mask']=mask[::-1,::-1].copy()
    return view


def audit_pilot_views(repo,split_path,*,instruction):
    repo=Path(repo).resolve(strict=True);split_path=Path(split_path).resolve(strict=True)
    if repo/'datasets' not in split_path.parents:
        raise ValueError('split outside relay research registry')
    frozen=json.loads(split_path.read_text())
    cards=sorted((repo/'datasets/cards').glob('*.json'))
    current=build_plan(repo,cards,task='lift_pot',instruction=instruction)['split']
    if frozen!=current or frozen['status']!='candidate_not_approved' or frozen['evaluation_claims_approved']:
        raise ValueError('split/card mismatch or unexpectedly approved candidate')
    summary={'schema':'lingbot_tactile_ablation_input_audit_v1',
             'status':'data_views_only_no_model_fitting',
             'split_sha256':hashlib.sha256(split_path.read_bytes()).hexdigest(),
             'variants':list(MODES),'folds':{}}
    for fold in ('train','validation','test'):
        count=0;changed=0;nonzero=0
        for trial in frozen[fold]:
            episode=PairedEpisode(repo,trial,instruction=instruction)
            for index in range(len(episode)):
                sample=episode.sample(index,include_images=True)
                views={mode:make_view(sample,mode) for mode in MODES}
                reference=views['tactile'];visual=views['visual_only'];permuted=views['permuted_causal_history']
                for mode,view in views.items():
                    if not np.array_equal(view['target']['action_command_14'],reference['target']['action_command_14']):
                        raise ValueError(f'target diverged for {mode}')
                    if view['metadata']!=reference['metadata'] or not np.array_equal(view['input']['command_state_14'],reference['input']['command_state_14']):
                        raise ValueError(f'source state diverged for {mode}')
                    for camera in CAMERAS:
                        img=view['input']['images_rgb'][camera]
                        if img.shape!=(240,320,3) or img.dtype!=np.uint8 or not np.array_equal(img,reference['input']['images_rgb'][camera]):
                            raise ValueError(f'RGB diverged for {camera}/{mode}')
                if np.any(visual['input']['tactile_history']) or np.any(visual['input']['tactile_mask']) or visual['input']['tactile_window_valid']:
                    raise ValueError('visual-only view retained tactile data')
                if (not np.array_equal(np.sort(permuted['input']['tactile_history'].reshape(-1)),
                                       np.sort(reference['input']['tactile_history'].reshape(-1))) or
                    not np.array_equal(np.sort(permuted['input']['tactile_mask'].reshape(-1)),
                                       np.sort(reference['input']['tactile_mask'].reshape(-1)))):
                    raise ValueError('permutation changed causal past values')
                if sample['metadata']['next_frame_step_delta']<=0:
                    raise ValueError('noncausal target step')
                count+=1;nonzero+=int(np.any(reference['input']['tactile_history'][...,:2]>0))
                changed+=int(not np.array_equal(permuted['input']['tactile_history'],reference['input']['tactile_history']))
        summary['folds'][fold]={'trials':len(frozen[fold]),'matched_pairs':count,
                                'pairs_with_nonzero_past_force':nonzero,
                                'pairs_changed_by_permutation':changed}
    return summary


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--split',type=Path,required=True)
    p.add_argument('--instruction',default='Lift the pot with both arms')
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();report=audit_pilot_views(args.repo,args.split,instruction=args.instruction)
    if args.out.exists():p.error('refusing to overwrite report')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x') as file:json.dump(report,file,indent=2);file.write('\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
