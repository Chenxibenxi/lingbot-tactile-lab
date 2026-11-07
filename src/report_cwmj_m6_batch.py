"""Audit all fixed M6 collection attempts; freeze data gate before any fitting."""
from pathlib import Path
import json,shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from prepare_cwmj_m6 import FOLDS,REPO
from cwmj_m6_samples import M6TerminalEpisode
from validate_expert_data import sha256

def report():
 root=REPO/'data/reports/cwmj_m6';rows=[];train=[];val=[];sources=[];rejected=[]
 for seed,fold in FOLDS.items():
  r=json.loads((root/f'seed_{seed}/result.json').read_text());rows.append(r)
  if r['status']!='accepted_success_source_verified':rejected.append({'seed':seed,'status':r['status'],'failure':r['failure']});continue
  ep=M6TerminalEpisode(seed);samples=[ep.sample(i,False) for i in range(len(ep))];nc=sum(s['metadata']['contact'] for s in samples)
  if nc<4 or len(samples)-nc<4:rejected.append({'seed':seed,'status':'insufficient_predeclared_contact_strata'});continue
  (train if fold=='train' else val).append(seed);sources.append({'seed':seed,'fold':fold,'samples':len(samples),'contact_samples':nc,'source_sha256':ep.hashes})
 gate={'new_train_seeds':train,'new_validation_seeds':val,'sources':sources,'rejected':rejected,'passed':len(train)>=4 and len(val)>=2,'minimum_new_train':4,'minimum_new_validation':2,'per_episode_min_contact':4,'per_episode_min_noncontact':4,'training_authorized_by_this_manifest':False,'collection_protocol_sha256':sha256(REPO/'docs/CWMJ_M6_DATA_EXPANSION_PLAN_ZH.md'),'scenario':'cwmj_single_wine_bottle_v2','prior_training_not_implicitly_merged':True}
 with (root/'data_manifest.json').open('x') as f:json.dump(gate,f,indent=2)
 fig,ax=plt.subplots(figsize=(11,4),layout='constrained');xs=range(len(rows));contact=[r['contact_windows'] for r in rows];other=[r['valid_windows']-r['contact_windows'] for r in rows]
 ax.bar(xs,contact,label='Causal past-contact windows',color='#529c88');ax.bar(xs,other,bottom=contact,label='Valid no-contact windows',color='#b6cad3');ax.set_xticks(list(xs),[str(r['seed']) for r in rows],rotation=45,ha='right');ax.axvline(11.5,color='gray',ls='--');ax.set(ylabel='Windows (not independent episodes)',title='M6 fixed 12 train / 4 validation collection');ax.legend(fontsize=8)
 for i,r in enumerate(rows):
  if not r['valid_windows']:ax.text(i,1,'rejected',rotation=90,ha='center',va='bottom',fontsize=7)
 fig.savefig(root/'coverage.png',dpi=160);plt.close(fig)
 table='\n'.join(f"| {r['seed']} | {r['fold']} | {r['status']} | {r['valid_windows']} | {r['contact_windows']} | {(r.get('failure') or {}).get('exception_type','—')} |" for r in rows)
 text=f'''# M6 CWMJ独立数据扩展：完整图文报告

## 目的与结论

固定12个新训练seed55–66、4个新验证seed67–70，各一次，未按结果补选。通过预先规定的新数据门的训练seed为{train}，验证seed为{val}。数据门通过={gate['passed']}（至少4新train、2新validation，各至少4接触及4非接触窗口）。这只衡量可用专家数据，不是策略收益或模型训练结果。

![完整采集覆盖](coverage.png)

| seed | fold | 状态 | 有效窗口 | 接触窗口 | 异常 |
|---|---|---|---:|---:|---|
{table}

## 方法、数据契约和失败

原在线专家、stride20、128请求/10000物理步/900秒，原稳定门与成功判据不变；同CWMJ单酒瓶双侧资产。每个完整成功回合再次核验源代码、资产factory/manifest、动作/触觉/观测物理时钟、HDF5及completion SHA。失败回合与预检拒绝均保留，不把缺失动作当标签，不把策略rollout混入专家数据。

图中窗口不是独立样本，训练/验证分组按seed；同一瓶几何限制了泛化范围。专家规划失败不等于训练策略失败，也不意味着触觉无效。逐seed图文报告包含实际接触曲线和成功回合相机帧；有数据异常则排除并保留原因。

## 下一步

数据门未通过则不拟合，先定位瓶颈；通过则依据data_manifest.json冻结多训练初始化、同预算zero/true/mismatch及独立确认协议，再开始训练。旧训练45/46只能在新训练manifest显式合并，旧验证49永不入训练；51–54保留探索性测试身份，71–86预留独立确认，不以本次专家采集窥探测试。

## 复现和逐实验报告

'''+ '\n'.join(f'- [seed{seed}](seed_{seed}/report.md)' for seed in FOLDS)+f'''

协议 `{REPO}/docs/CWMJ_M6_DATA_EXPANSION_PLAN_ZH.md`，生成器prepare_cwmj_m6.py，采集启动器work/run_cwmj_m6.sh，读取器cwmj_m6_samples.py。源SHA与采集退出记录见各result.json及source_manifest.json，完整data_manifest.json为不可覆盖快照。原竞赛代码/checkpoint只读；退出释放自有GPU租约，不重启Strata。
'''
 (root/'report.md').write_text(text);dest=Path('/home/admin001/lingbot-tactile-artifacts/cwmj-m6');dest.mkdir(exist_ok=True)
 for name in ['report.md','coverage.png','data_manifest.json']:shutil.copy2(root/name,dest/name)
 print(json.dumps({'new_train':train,'new_validation':val,'passed':gate['passed']}),flush=True)
if __name__=='__main__':report()
