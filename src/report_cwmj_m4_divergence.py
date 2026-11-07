"""CPU paired trajectory audit; differences after first divergence aren't causal local effects."""
from pathlib import Path
import json,shutil
import h5py,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1]
def report():
 out=R/'data/reports/cwmj_m4_divergence';out.mkdir(exist_ok=True)
 records=[];fig,axes=plt.subplots(2,2,figsize=(10,6),layout='constrained')
 for seed,ax in zip(range(100051,100055),axes.flat):
  paths={a:R/f'data/eval/cwmj_m4_{seed}_{a}' for a in ('zero','true')}
  if not all((p/'result.json').exists() for p in paths.values()):ax.set_title(f'{seed}: pending');continue
  if not all((p/'rollout.hdf5').exists() for p in paths.values()):ax.set_title(f'{seed}: unavailable');continue
  with h5py.File(paths['zero']/'rollout.hdf5') as a,h5py.File(paths['true']/'rollout.hdf5') as b:
   n=min(len(a['applied/command_14']),len(b['applied/command_14']));delta=np.abs(a['applied/command_14'][:n]-b['applied/command_14'][:n]).max(1);idx=np.flatnonzero(delta>0)
   first=int(idx[0]) if len(idx) else None
   item={'seed':seed,'first_different_action_zero_based':first,'common_compared_actions':n}
   if first is not None:
    req=int(b['applied/request_index'][first]);item['first_divergence_request']=req
    item['equal_pre_request_rgb_and_state']=all(np.array_equal(a[k][req],b[k][req]) for k in ['state_14','images/head','images/left','images/right'])
    item['history_max_abs_at_divergence']={arm:float(np.abs(h['tactile/history'][req]).max()) for arm,h in [('zero',a),('true',b)]}
   ax.plot(np.arange(n),delta,label='Max decoded command difference');ax.set(xlabel='Action index (0-based)',ylabel='Mixed-unit max delta',title=str(seed));ax.set_yscale('symlog',linthresh=.001)
   for arm,c in [('zero','gray'),('true','teal')]:
    r=json.loads((paths[arm]/'result.json').read_text());q=next((q['request_index'] for q in r['requests'] if q['past_contact']),None);item[arm+'_first_contact_request']=q
    if q is not None:ax.axvline(q*10,color=c,linestyle='--',label=f'{arm}: first past contact')
   ax.legend(fontsize=7);records.append(item)
 fig.savefig(out/'divergence.png',dpi=160);plt.close(fig)
 (out/'summary.json').write_text(json.dumps(records,indent=2))
 text='''# M4闭环轨迹分叉诊断

## 方法

只读取已结束同seed两臂HDF5，逐元素比较实际14维动作、请求前RGB/状态和历史力。动作含关节角与夹爪命令，最大差只是定位分叉的混合单位诊断量，不是物理误差或综合性能评分。虚线是各自轨迹第一次输入过去接触的请求，不代表两条轨迹此后状态相同。

![动作分叉与接触时间](divergence.png)

## 审计结果

'''+ '\n'.join(f"- seed{x['seed']}：首个不同动作（0起始）{x['first_different_action_zero_based']}；该请求前图像/状态一致={x.get('equal_pre_request_rgb_and_state')}；两臂历史数值最大绝对值={x.get('history_max_abs_at_divergence')}。" for x in records)+'''

## 解释与下一步

若首处分叉在同一观测、零触觉历史下出现，说明不同训练权重已经改变零力输入时的策略输出，不能把后续成功差直接解释为在线接触反馈收益。首次传感器窗口无效时适配器门关闭；后续窗口有效但数值零时，训练出的偏置/网络仍可影响动作。应执行M5同true权重输入消融，覆盖全部四个seed；这是探索性机制分析，不能代替独立泛化验证。

代码 `src/report_cwmj_m4_divergence.py`；数据为M4固定51–54的只读策略rollout，未加入训练。诊断不训练、不申请GPU，未改源回合。尚未完成的配对显示pending，完整批次后重新生成汇总。
'''
 (out/'report.md').write_text(text);dest=Path('/home/admin001/lingbot-tactile-artifacts/cwmj-m4-divergence');shutil.copytree(out,dest,dirs_exist_ok=True)
 print(json.dumps(records))
if __name__=='__main__':report()
