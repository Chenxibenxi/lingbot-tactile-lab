"""Native 50-ms CWMJ policy sampling; no labels or future targets accepted."""
import time
import torch
from contact_policy_inference import VISUAL_KEYS
class NativeCWMJInference:
    def __init__(self,model):
        self.model=model;model.eval()
    def predict(self,visual,payload,*,noise_seed):
        required={'tactile_history','tactile_mask','tactile_age_ms','tactile_valid','contact_proprio','command_history','command_history_mask'}
        if set(payload)!=required or set(visual)!=set(VISUAL_KEYS):raise ValueError('exact causal observation keys required')
        if payload['tactile_history'].shape!=(1,3,4):raise ValueError('native three-interval forces required')
        torch.cuda.synchronize();start=time.perf_counter()
        gpu=lambda v:v.cuda().to(torch.bfloat16) if v.is_floating_point() else v.cuda()
        visual={k:gpu(v) for k,v in visual.items()};payload={k:gpu(v) for k,v in payload.items()}
        generator=torch.Generator(device='cuda').manual_seed(noise_seed)
        noise=torch.randn((1,50,55),generator=generator,device='cuda',dtype=torch.bfloat16)
        old_cache=self.model.config.use_cache;self.model.config.use_cache=True
        try:
            with torch.inference_mode():actions=self.model.sample_actions(**visual,**payload,noise=noise,action_step_s=.05)
        finally:self.model.config.use_cache=old_cache
        torch.cuda.synchronize()
        if actions.shape!=(1,50,55) or not torch.isfinite(actions).all():raise ValueError('invalid generated action chunk')
        return {'actions':actions.detach().float().cpu(),'elapsed_s':time.perf_counter()-start,'action_step_s':.05,'noise_seed':noise_seed}
