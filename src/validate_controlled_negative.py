"""Independent SHA-bound validator for controlled negative RGB episodes.

Negative manifests are intentionally incompatible with positive expert cards.
No raw trajectory, HDF5 or video belongs in the Git-tracked card.
"""
from __future__ import annotations
import hashlib
import json
import socket
from pathlib import Path
from validate_expert_data import validate,read_jsonl


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as file:
        for b in iter(lambda:file.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def build_card(repo,trial):
    repo=Path(repo).resolve(strict=True);trial=Path(trial).resolve(strict=True)
    if trial.parent!=repo/'data/negative':raise ValueError('controlled negative must be under isolated relay negative root')
    if (trial/'completion.json').exists() or (trial/'expert_success_gate.json').exists():
        raise ValueError('positive expert completion forbidden')
    gate_path=trial/'negative_gate.json';manifest_path=trial/'negative_completion.json'
    gate=json.loads(gate_path.read_text());manifest=json.loads(manifest_path.read_text())
    if gate.get('schema')!='lingbot_controlled_open_grasp_gate_v1' or manifest.get('schema')!='lingbot_controlled_negative_complete_v1':
        raise ValueError('not a controlled negative source')
    if manifest.get('negative_gate_sha256')!=sha256(gate_path):raise ValueError('negative gate SHA mismatch')
    for record in (gate,manifest):
        if (record.get('task')!='pick_diverse_bottles' or record.get('seed')!=100000 or record.get('episode')!=0 or
            record.get('source_machine_role')!='relay_single_rtx3090' or record.get('source_hostname')!=socket.gethostname() or
            record.get('planner_ok') is not True or record.get('goal_ok') is not False or
            record.get('intervened_grasps')!=2 or record.get('accepted_for_expert_training') is not False):
            raise ValueError('controlled negative identity/intervention mismatch')
    if gate.get('status')!='controlled_negative_not_training_approved' or manifest.get('status')!='quarantined_negative_not_training_or_eval':
        raise ValueError('controlled negative unexpectedly approved')
    raws=[p for p in (trial/'force').glob('*.jsonl') if not p.name.endswith(('.obs.jsonl','.frames.jsonl','.policy.jsonl'))]
    if len(raws)!=1:raise ValueError('expected exactly one same-run raw stream')
    raw=raws[0];obs=raw.with_suffix('.obs.jsonl');frames=raw.with_suffix('.frames.jsonl')
    hdf5=trial/'demonstration/data/episode0.hdf5'
    current=validate(raw,obs,frames,hdf5)  # source RGB/timeline/command validation, no positive completion
    if gate.get('source_sha256')!={k:current['sha256'][k] for k in ('raw','obs','frames')}:
        raise ValueError('negative gate raw stream SHA mismatch')
    for name in ('task','seed','episode','physics_steps','camera_observations','saved_frames','dt_s','saved_first_step','saved_last_step','sha256'):
        if manifest.get(name)!=current[name]:raise ValueError(f'negative manifest {name} differs from source bytes')
    if gate.get('saved_frame_markers')!=current['saved_frames']:
        raise ValueError('negative gate camera count mismatch')
    _,*physics=read_jsonl(raw)
    contacts={side:sum(int(r[side]['points']>0) for r in physics) for side in ('left','right')}
    if manifest.get('contact_physics_steps_by_side')!=contacts:
        raise ValueError('negative contact counts differ from source physics')
    return {'schema':'lingbot_controlled_negative_card_v1',
            'status':'quarantined_controlled_negative_not_training_or_eval',
            'trial_id':trial.name,'source_machine_role':'relay_single_rtx3090',
            'task':current['task'],'seed':current['seed'],'episode':current['episode'],
            'perturbation':'simulation_only_two_grasp_targets_open',
            'matched_positive_seed':100000,'planner_ok':True,'goal_ok':False,
            'physics_steps':current['physics_steps'],'saved_frames':current['saved_frames'],
            'contact_physics_steps_by_side':contacts,
            'observed_any_contact':any(contacts.values()),
            'validated_same_run_rgb_hdf5':True,
            'accepted_for_expert_training':False,
            'source_sha256':{**current['sha256'],'negative_gate':sha256(gate_path),'negative_completion':sha256(manifest_path)},
            'note':'Controlled simulated open-grasp task failure; not an independent seed, a hardware force limit, or an approved expert training example.'}


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--trial',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():p.error('refusing to overwrite negative diagnostic card')
    card=build_card(args.repo,args.trial)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x') as file:json.dump(card,file,indent=2);file.write('\n')
    print('CONTROLLED_NEGATIVE_CARD_QUARANTINED',card['trial_id'],card['physics_steps'],card['saved_frames'])


if __name__=='__main__':main()
