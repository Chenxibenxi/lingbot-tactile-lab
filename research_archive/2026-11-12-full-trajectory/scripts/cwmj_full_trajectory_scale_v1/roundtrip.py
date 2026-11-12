from pathlib import Path
import json,numpy as np
from cwmj_full_trajectory_data import FullTrajectoryDataset
from cwmj_model_bridge_v1 import build_feature_transform,decode_model_actions
from contact_policy_inference import prepare_observation
from prepare_real_vla_visual_smoke import CAMERAS
R=Path('/mnt/relaydata/lingbot-tactile-lab');O=R/'data/reports/cwmj_full_trajectory_roundtrip_v1';O.mkdir(exist_ok=False);transform=build_feature_transform();d=FullTrajectoryDataset(R/'data/cwmj_full_trajectory_windows_v1/index.json',transform);contract=json.loads((R/'datasets/cwmj_aloha_14d_contract_v3_1.json').read_text());seen=set();selected=[]
for i,r in enumerate(d.rows):
 if r['seed'] not in [48000,48063,48127]:continue
 key=(r['seed'],r['stage'])
 if key not in seen or r['offset'] in [1,2,93,94,141,142]:selected.append(i);seen.add(key)
results=[]
for i in selected:
 x=d[i];r=x['window'];p=Path(r['observation']);meta=json.loads(p.read_text())
 with np.load(p.parent/meta['images_file']) as images:rgb={key:images[v.split('.')[-1]] for key,v in CAMERAS.items()}
 visual,metadata=prepare_observation(transform,images_rgb=rgb,command_state_14=meta['observation']['command_history'][-1]['action_14d_normalized'],instruction=meta['observation']['language']['instruction'])
 decoded=decode_model_actions(transform,metadata,x['batch']['actions'][0],contract=contract);raw=np.asarray(decoded.get('raw_commands_14d',decoded.get('commands_14d')));n=r['valid_steps'];expected=d.actions[r['seed']][r['offset']:r['offset']+n];error=float(np.abs(raw[:n]-expected).max());assert error<1e-5,(i,error);results.append({'index':i,'seed':r['seed'],'stage':r['stage'],'valid_steps':n,'max_abs_error':error})
result={'passed':True,'windows_checked':len(results),'seeds':[48000,48063,48127],'max_abs_error':max(x['max_abs_error'] for x in results),'scope':'Stage starts and history/tail boundaries in three training trajectories; not all18304windows. Valid actions only; no learned model evaluation. Original robotwin norm provenance unresolved.','entries':results};(O/'result.json').write_text(json.dumps(result,indent=2))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(10,3));ax.plot([x['max_abs_error'] for x in results],'.');ax.set(xlabel='Selected window',ylabel='Maximum absolute decode error');fig.tight_layout();fig.savefig(O/'errors.png',dpi=130);plt.close(fig)
(O/'report.md').write_text('# 完整轨迹动作转换往返核验\n\n![误差](errors.png)\n\n'+f"3个训练场景，共{len(results)}个阶段起点及边界窗口，最大有效动作往返误差{result['max_abs_error']:.3g}，均小于1e-5。使用原FeatureTransform及原解码器，无截断拼接。\n\n"+'检查范围为分阶段边界抽样，未逐个转换所有18304窗口。回合尾部填充动作不参与比较与训练损失。原robotwin归一化统计适配性仍是独立问题；往返正确不等于学习已成功。\n');print({k:v for k,v in result.items() if k!='entries'})
