"""Join validated same-run expert HDF5 frames with causal finger-force histories.

Output is an INDEX for future LeRobot conversion; image pixels remain in HDF5.
Action label uses the exact old convention state[t+1], never future contact.
"""
import argparse
import json
from pathlib import Path
from align_temporal import align
from validate_expert_data import read_jsonl, validate


def build(raw, obs, frames, hdf5, completion):
    import h5py
    check=validate(raw,obs,frames,hdf5,completion)
    markers={row['observation_marker_index']:row for row in align(raw,obs)}
    saved=read_jsonl(frames)[1:]
    with h5py.File(hdf5,'r') as dataset:
        states=dataset['joint_action/vector']
        for t in range(len(saved)-1):
            frame=saved[t]
            tactile=markers[frame['obs_index']]
            if tactile['anchor_physics_step']!=frame['physics_step']:
                raise ValueError('frame/physics anchor changed after validation')
            next_frame=saved[t+1]
            if next_frame['physics_step']<frame['physics_step']:
                raise ValueError('future frame step precedes observation')
            yield {
                'schema':'phizju_paired_expert_index_v1',
                'task':check['task'],'seed':check['seed'],'episode':check['episode'],
                'sample_index':t,
                'input':{
                    'frame_index':t,'capture_physics_step':frame['physics_step'],
                    'capture_sim_time_s':frame['sim_time_s'],
                    'image_ref':{cam:f'observation/{cam}/rgb[{t}]' for cam in ('head_camera','left_camera','right_camera')},
                    'command_state_14':frame['command_state_14'],
                    'actual_joints':frame['actual_joints'],
                    'tactile_history':tactile['features'],
                    'tactile_mask':tactile['sample_fraction_mask'],
                    'tactile_finger_order':tactile['finger_order'],
                    'tactile_feature_order':tactile['feature_order'],
                    'tactile_window_valid':tactile['quality']['window_valid'],
                    'tactile_quality':tactile['quality'],
                },
                'target':{
                    'action_command_14':states[t+1].tolist(),
                    'action_source_frame_index':t+1,
                    'next_frame_physics_step':next_frame['physics_step'],
                },
                'evidence':{'hdf5_sha256':check['sha256']['hdf5'],'raw_sha256':check['sha256']['raw'],
                            'completion_schema':'phizju_expert_episode_complete_v1'},
            }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('raw','obs','frames','hdf5','completion','out'):
        p.add_argument('--'+name,required=True,type=Path)
    args=p.parse_args()
    if args.out.resolve() in {getattr(args,k).resolve() for k in ('raw','obs','frames','hdf5','completion')}:
        p.error('output must differ from all inputs')
    from tempfile import NamedTemporaryFile
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with NamedTemporaryFile(mode='w',dir=args.out.parent,encoding='utf-8',prefix='.paired_',delete=False) as fh:
        temp=Path(fh.name)
        try:
            count=valid=0
            for row in build(args.raw,args.obs,args.frames,args.hdf5,args.completion):
                fh.write(json.dumps(row,allow_nan=False)+'\n')
                count+=1; valid+=row['input']['tactile_window_valid']
            if count<1:
                raise ValueError('empty paired demonstration')
        except Exception:
            fh.close();temp.unlink(missing_ok=True)
            raise
    temp.replace(args.out)
    print(f'paired frames={count}; complete tactile histories={valid}; out={args.out}')


if __name__=='__main__':main()
