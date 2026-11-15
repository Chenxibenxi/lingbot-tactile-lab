"""Same-run observation capture with explicit causal interval sequence."""
import copy,json,hashlib
from pathlib import Path
import numpy as np
from run_cwmj46_model_loop import export_policy_images,json_safe_observation

def capture_observation(env, directory, name, *, observation=None, include_contact_persistence=False):
    directory=Path(directory);obs=env.observe() if observation is None else observation
    if abs(float(obs["decision_time_s"])-float(env.env.data.time))>1e-9:raise ValueError("stale observation")
    export=export_policy_images(obs)
    images=directory/(name+'_images.npz');metadata=directory/(name+'.json')
    if images.exists() or metadata.exists():raise FileExistsError(name)
    now=float(obs['decision_time_s'])
    intervals=copy.deepcopy(env.env._tactile_intervals)
    if any(v['interval_end_s']>now+1e-9 for v in intervals):raise ValueError('future interval')
    persistence=None
    if include_contact_persistence:
        from tactile_contact_persistence import summarize_contact_persistence
        samples=env.env._last_interval_substeps
        persistence={side:summarize_contact_persistence(samples,decision_time_s=now,fingers=fingers)
                     for side,fingers in [('left',('fl_link7','fl_link8')),('right',('fr_link7','fr_link8'))]}
    np.savez_compressed(images,**export['images'])
    data={'observation':json_safe_observation(obs),'images_file':images.name,'image_sha256':export['sha256_by_env_key'],
          'causal_tactile_intervals':intervals,'all_intervals_valid':bool(intervals) and all(v['sensor_valid'] for v in intervals),
          'sensor_sequence_semantics':'completed force interval statistics only; diagnostic pose/contact geometry excluded',
          'diagnostic_labels_in_policy_input':False}
    if persistence is not None:
        data['contact_persistence']={'schema':'causal_contact_persistence_v1','scope':'latest measured control interval only; no cross-interval continuity claim','hands':persistence}
    metadata.write_text(json.dumps(data,indent=2))
    return {'metadata':metadata.name,'metadata_sha256':hashlib.sha256(metadata.read_bytes()).hexdigest(),'images':images.name,'images_sha256':hashlib.sha256(images.read_bytes()).hexdigest(),'time_s':now}
