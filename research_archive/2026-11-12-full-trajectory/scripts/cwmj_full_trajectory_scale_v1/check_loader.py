import json
from pathlib import Path
from cwmj_full_trajectory_data import FullTrajectoryDataset
from cwmj_model_bridge_v1 import build_feature_transform
R=Path('/mnt/relaydata/lingbot-tactile-lab');d=FullTrajectoryDataset(R/'data/cwmj_full_trajectory_windows_v1/index.json',build_feature_transform());rows=[]
for i in [0,1,2,93,94,141,142,143,len(d)-1]:
 x=d[i];rows.append({'index':i,'valid_steps':x['window']['valid_steps'],'mask_sum':int(x['batch']['joint_mask'].sum()),'tactile_mask':x['payload']['tactile_mask'].tolist()})
p=R/'data/reports/cwmj_full_trajectory_scale_summary_v1/loader_check.json';p.write_text(json.dumps(rows,indent=2));print(json.dumps(rows))
