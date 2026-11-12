from pathlib import Path
import json,sys,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/mnt/relaydata/lingbot-tactile-lab');seed=int(sys.argv[1]);D=R/'data/cwmj_full_trajectory_scale_v1'/str(seed);O=R/'data/reports/cwmj_full_trajectory_capture_v1'/str(seed);O.mkdir(parents=True,exist_ok=True);res=json.loads((D/'result.json').read_text());audit=json.loads((D/'validation.json').read_text());rows=res['records'];last={r['stage']:r for r in rows if r['execution']['accepted']};fig,axes=plt.subplots(3,3,figsize=(12,9))
for ax,(stage,row) in zip(axes.flat,last.items()):
 with np.load(D/row['post']['images']) as im:ax.imshow(im['cam_high'])
 ax.set_title(stage,fontsize=9);ax.axis('off')
fig.tight_layout();fig.savefig(O/'stages.png',dpi=120);plt.close(fig)
size=sum(p.stat().st_size for p in D.iterdir() if p.is_file());out={'seed':seed,'capture_status':res['status'],'provenance_audit':audit,'bytes':size,'estimated_128_raw_GiB':size*128/1024**3,'task_quality':json.loads((D/'task_quality.json').read_text()) if (D/'task_quality.json').exists() else None,'training_approval':False,'reason':'Task outcome and stage-specific expert quality not yet certified'};(O/'result.json').write_text(json.dumps(out,indent=2));(O/'report.md').write_text('# 完整轨迹同源采集\n\n![实际阶段末帧](stages.png)\n\n```json\n'+json.dumps(out,indent=2)+'\n```\n\n第一步静置作为触觉历史预热，随后143个连续50ms转移，边界图像不重复存储。通过来源校验不等于专家质量合格；未获批入库。\n');print(json.dumps(out))
