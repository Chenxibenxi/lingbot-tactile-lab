"""CPU-only visual/tactile input from ONE SHA-bound relay TRAIN expert observation.

For an untrained ZERO-adapter numerical identity smoke, not model fitting or
validation/test scoring. The target action and future contact never reach VLA.
"""
from __future__ import annotations
import json
from pathlib import Path
from types import SimpleNamespace
import torch
import yaml
from torchvision.transforms.v2 import Resize
from paired_samples import PairedEpisode
from probe_real_v2_meta import TRAIN,TOKENIZER,build_old_config,verify_sources
from validate_expert_data import sha256

TRIAL='relay_liftpot_s100000_expert0'
SPLIT='relay_lift_pot_seed_split_candidate_v1.json'
MODEL_ROOT=Path('/mnt/relaydata/lingbot/lingbot-vla-v2')
NORM=MODEL_ROOT/'assets/norm_stats/robotwin_clean.json'
ROBOT=MODEL_ROOT/'configs/robot_configs/robotwin.yaml'
CAMERAS={'head_camera':'observation.images.cam_high',
         'left_camera':'observation.images.cam_left_wrist',
         'right_camera':'observation.images.cam_right_wrist'}


def check_train_anchor(repo, trial_id=TRIAL):
    repo=Path(repo).resolve(strict=True)
    if repo!=Path(__file__).resolve().parents[1] or trial_id!=TRIAL:
        raise ValueError('only original relay TRAIN seed100000 may enter visual smoke')
    split=json.loads((repo/'datasets'/SPLIT).read_text())
    if (split.get('schema')!='lingbot_tactile_seed_split_plan_v1' or
            split.get('source_machine_role')!='relay_single_rtx3090' or
            split.get('status')!='candidate_not_approved' or
            split.get('evaluation_claims_approved') is not False or
            trial_id not in split['train'] or
            trial_id in split['validation']+split['test']):
        raise ValueError('unapproved or heldout source is not valid for smoke')
    card=repo/'datasets/cards'/f'{trial_id}.json'
    if sha256(card)!=split['expert_card_sha256'].get(trial_id):
        raise ValueError('real TRAIN source card SHA drifted')
    verify_sources()
    if not NORM.is_file() or not ROBOT.is_file():
        raise ValueError('original visual normalization or robot mapping unavailable')
    return repo,card


def prepare(repo,*,trial_id=TRIAL,index=0):
    repo,card=check_train_anchor(repo,trial_id)
    if type(index) is not int or index!=0:
        raise ValueError('only fixed first causal TRAIN observation, no tuning/selecting frames')
    from lingbotvla.models import build_processor
    from lingbotvla.data.vla_data.utils import FeatureTransform
    episode=PairedEpisode(repo,trial_id,instruction='Lift the pot')
    item=episode.sample(index,include_images=True)
    cfg=build_old_config()
    cfg.num_steps=2  # only numerical identity/peak safety, not deployed policy cadence
    data=SimpleNamespace(**yaml.safe_load(TRAIN.read_text())['data'])
    processor=build_processor(str(TOKENIZER))
    transform=FeatureTransform(str(ROBOT),data,cfg,processor,
                               chunk_size=cfg.chunk_size,norm_stats_path=str(NORM))
    images=item['input']['images_rgb']
    resize=Resize((256,256))
    obs={model_name:resize(torch.as_tensor(images[cam]).permute(2,0,1).contiguous().float())
         for cam,model_name in CAMERAS.items()}
    obs.update({'observation.state':torch.as_tensor(item['input']['command_state_14']),
                'task':item['input']['instruction']})
    transformed=transform.apply(obs,policy_eval=True)
    required={'images','img_masks','lang_tokens','lang_masks','state','image_grid_thw'}
    if not required.issubset(transformed) or transformed['images'].ndim!=3 or transformed['state'].shape!=(55,):
        raise ValueError('original Qwen multimodal transform shape changed')
    fields=item['input'];tactile={
        'tactile_history':torch.as_tensor(fields['tactile_history']).unsqueeze(0),
        'tactile_mask':torch.as_tensor(fields['tactile_mask']).unsqueeze(0),
        'tactile_age_ms':torch.tensor([fields['tactile_age_ms']]),
        'tactile_valid':torch.tensor([fields['tactile_window_valid']],dtype=torch.bool)}
    if not bool(tactile['tactile_valid'].item()):
        raise ValueError('first causal train visual sample lacks valid tactile history')
    report={'schema':'lingbot_real_vla_train_visual_smoke_input_v1',
            'trial_role':'train','trial_id':trial_id,'source_card_sha256':sha256(card),
            'frame_index':item['metadata']['frame_index'],
            'capture_physics_step':fields['capture_physics_step'],
            'tactile_source_causal_valid':True,'future_target_exposed_to_model':False,
            'training_approval':False,'sample_index':index,'num_steps_smoke_only':cfg.num_steps}
    return cfg,transformed,tactile,report
