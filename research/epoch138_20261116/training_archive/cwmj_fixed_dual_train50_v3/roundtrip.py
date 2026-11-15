from pathlib import Path
import json,numpy as np
from cwmj_full_trajectory_data import FullTrajectoryDataset
from cwmj_model_bridge_v1 import build_feature_transform,decode_model_actions
from contact_policy_inference import prepare_observation
from prepare_real_vla_visual_smoke import CAMERAS
R=Path('/mnt/relaydata/lingbot-tactile-lab');O=R/'data/cwmj_fixed_dual_train50_windows_v3';transform=build_feature_transform();d=FullTrajectoryDataset(R/'data/cwmj_fixed_dual_train50_windows_v3/index.json',transform);contract=json.loads((R/'datasets/cwmj_aloha_14d_contract_v3_1.json').read_text());seen=set();selected=[]
for i,r in enumerate(d.rows):
 key=(r['seed'],r['stage'])
 if key not in seen or r['offset'] in [1,2,113,114,161,162]:selected.append(i);seen.add(key)
results=[]
for i in selected:
 x=d[i];r=x['window'];p=Path(r['observation']);meta=json.loads(p.read_text())
 with np.load(p.parent/meta['images_file']) as images:rgb={key:images[v.split('.')[-1]] for key,v in CAMERAS.items()}
 visual,metadata=prepare_observation(transform,images_rgb=rgb,command_state_14=meta['observation']['command_history'][-1]['action_14d_normalized'],instruction=meta['observation']['language']['instruction'])
 decoded=decode_model_actions(transform,metadata,x['batch']['actions'][0],contract=contract);raw=np.asarray(decoded.get('raw_commands_14d',decoded.get('commands_14d')));n=r['valid_steps'];expected=d.actions[r['seed']][r['offset']:r['offset']+n];error=float(np.abs(raw[:n]-expected).max());assert error<1e-5,(i,error);results.append({'index':i,'seed':r['seed'],'stage':r['stage'],'valid_steps':n,'max_abs_error':error})
result={'passed':True,'windows_checked':len(results),'seeds':sorted({x['seed'] for x in d.rows}),'max_abs_error':max(x['max_abs_error'] for x in results),'scope':'All50 trajectories: every stage entry and initial/tail boundaries; no learned-model capability claim.','entries':results};(O/'roundtrip_validation.json').write_text(json.dumps(result,indent=2))
print({k:v for k,v in result.items() if k not in ["entries","seeds"]})
