"""Observation-only VLA inference and original physical action decoding.

The caller owns model loading, sensor clock verification, GPU lease, and delayed
execution. Scene labels, material, future candidates and targets are not accepted.
"""
import time
import torch
from torchvision.transforms.v2 import Resize
from prepare_real_vla_visual_smoke import CAMERAS
from contact_modality_controls import observation_control
from contact_policy_action_decode import decode_actions

VISUAL_KEYS=('images','img_masks','state','lang_tokens','lang_masks','image_grid_thw')


def prepare_observation(transform, *, images_rgb, command_state_14, instruction):
    if set(images_rgb)!=set(CAMERAS):raise ValueError('exactly three original RGB cameras required')
    state=torch.as_tensor(command_state_14).float()
    if state.shape!=(14,) or not torch.isfinite(state).all():raise ValueError('finite 14D command state required')
    resize=Resize((256,256));obs={}
    for camera,alias in CAMERAS.items():
        x=torch.as_tensor(images_rgb[camera])
        if x.ndim!=3 or x.shape[-1]!=3 or x.dtype!=torch.uint8:
            raise ValueError('original uint8 HWC RGB required')
        obs[alias]=resize(x.permute(2,0,1).contiguous().float())
    if not isinstance(instruction,str) or not instruction.strip():raise ValueError('task instruction required')
    obs.update({'observation.state':state,'task':instruction})
    transformed=transform.apply(obs,policy_eval=True)
    visual={k:transformed[k] if k=='image_grid_thw' else transformed[k].unsqueeze(0) for k in VISUAL_KEYS}
    return visual,transformed


class ContactPolicyInference:
    def __init__(self,model,transform,*,mode='true',clip_grippers=True):
        if mode not in ('true','missing','zero'):raise ValueError('mismatch needs separately frozen donor protocol')
        self.model=model;self.transform=transform;self.mode=mode;self.clip_grippers=clip_grippers
        model.eval()

    def predict(self,*,images_rgb,command_state_14,instruction,contact_observation,noise_seed):
        required={'tactile_history','tactile_mask','tactile_age_ms','tactile_valid','contact_proprio','command_history','command_history_mask'}
        if set(contact_observation)!=required:raise ValueError('exactly seven causal contact input fields required')
        torch.cuda.synchronize();start=time.perf_counter()
        visual,metadata=prepare_observation(self.transform,images_rgb=images_rgb,command_state_14=command_state_14,instruction=instruction)
        visual={k:v.cuda().to(torch.bfloat16) if v.is_floating_point() else v.cuda() for k,v in visual.items()}
        tactile={k:v.cuda() for k,v in observation_control(contact_observation,self.mode).items()}
        generator=torch.Generator(device='cuda').manual_seed(noise_seed)
        noise=torch.randn((1,50,55),generator=generator,device='cuda',dtype=torch.bfloat16)
        with torch.inference_mode():actions=self.model.model.sample_actions(**visual,**tactile,noise=noise)
        if actions.shape!=(1,50,55):raise ValueError('unexpected VLA action chunk shape')
        decoded=decode_actions(self.transform,actions[0],metadata,clip_grippers=self.clip_grippers)
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        return dict(**decoded,normalized_actions=actions.detach().float().cpu(),elapsed_seconds=elapsed,
                    latency_scope='image preprocessing + GPU transfer + generation + physical decode; excludes sensor acquisition',
                    noise_seed=noise_seed,mode=self.mode)
