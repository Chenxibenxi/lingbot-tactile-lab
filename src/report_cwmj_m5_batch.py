"""Exploratory same-weight runtime input intervention, all four reused test seeds."""
from pathlib import Path
import json,math,shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
R=Path(__file__).resolve().parents[1]
def report():
 out=R/'data/reports/cwmj_m5_closed_loop';rows=[];pairs=[];matrix=np.full((3,4),-1);texts={}
 for j,seed in enumerate(range(100051,100055)):
  for i,(stage,arm,label) in enumerate([('m4','zero','zero-trained / zero-input'),('m4','true','true-trained / true-input'),('m5','zero','true-trained / zero-input')]):
   r=json.loads((R/f'data/eval/cwmj_{stage}_{seed}_{arm}/result.json').read_text());full=r['status']=='completed' and r.get('termination') in ('task_success','official_action_limit')
   matrix[i,j]=int(r['success']) if full else -1;texts[i,j]=('PASS' if full and r['success'] else 'FAIL' if full else 'EXCLUDED')+f"\n{r['actions']} actions"
   rows.append({'seed':seed,'condition':label,'full':full,'success':r.get('success'),'actions':r['actions'],'termination':r.get('termination',r.get('exception'))})
  pairs.append(json.loads((out/f'paired_{seed}.json').read_text()))
 valid=[]
 for p in pairs:
  relevant=[x for x in rows if x['seed']==p['seed'] and x['condition'].startswith('true-trained')]
  keys=['initial_rgb_and_state_equal','first10_actions_equal','invalid_tactile_first_both','same_adapter_sha256','same_scenario','before_first_contact_actions_equal']
  if len(relevant)==2 and all(x['full'] for x in relevant) and all(p.get(k,False) for k in keys):valid.append(p)
 wins=sum(p['outcomes']['true'] and not p['outcomes']['zero'] for p in valid);losses=sum(p['outcomes']['zero'] and not p['outcomes']['true'] for p in valid);n=wins+losses
 pval=min(1.,2*sum(math.comb(n,k) for k in range(min(wins,losses)+1))/2**n) if n else 1.
 summary={'rows':rows,'valid_complete_pairs':len(valid),'true_input_only_success':wins,'zero_input_only_success':losses,'exploratory_exact_two_sided_p':pval,'independent_confirmation':False,'pairs':pairs}
 counts={c:sum(r['full'] and r['success'] for r in rows if r['condition']==c) for c in {r['condition'] for r in rows}}
 summary['successes_by_condition']=counts
 (out/'summary.json').write_text(json.dumps(summary,indent=2))
 fig,ax=plt.subplots(figsize=(10,4),layout='constrained');ax.imshow(matrix,cmap=ListedColormap(['#bfc2c8','#e1b38a','#80b49a']),vmin=-1,vmax=1);ax.set_xticks(range(4),range(100051,100055));ax.set_yticks(range(3),['Zero-trained / zero-input','True-trained / true-input','True-trained / zero-input'])
 for (i,j),v in texts.items():ax.text(j,i,v,ha='center',va='center')
 ax.set_title('M5 same-weight input ablation | reused M4 test seeds');fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
 table='\n'.join(f"| {r['seed']} | {r['condition']} | {r['success']} | {r['actions']} | {r['termination']} |" for r in rows)
 text=f'''# M5 同权重触觉输入消融：完整图文报告

## 问题、预登记时点与结果

本实验在看到M4部分结果后制定，重用全部四个M4测试seed，属于探索性机制分析。固定M4 true的最后360步权重，推理history置零，保留真实mask、有效性和年龄。第一行M4无力训练策略仅供背景比较；第二、三行是同权重输入干预。

真实输入成功{counts['true-trained / true-input']}/4，同权重置零输入成功{counts['true-trained / zero-input']}/4。当前四对成功/失败结果一致，未观察到在线触觉的成功率提升；唯一共同成功seed100051分别用71与245动作。这是单回合的动作效率线索，不是独立确认的效率收益，也不能把400步失败当作成功完成用时。

通过完整回合与全部配对门的样本数{len(valid)}；真实输入独有成功{wins}，零输入独有成功{losses}。不一致配对双侧精确二项检验p={pval:.4f}，探索性且未作多重校正；四seed不能证明稳定总体收益。

![三条件对照](comparison.png)

| seed | 条件 | 成功 | 动作 | 终止 |
|---|---|---|---:|---|
{table}

## 技术配对与方法

同一true权重SHA、同资产manifest/factory、相同起始三路RGB/14维状态、相同首10动作。进一步核验首次触觉接触前所有实际动作逐元素相同，防止加载/推理路径差异被误算成触觉作用。固定400动作、每10动作反馈、10步去噪、seed与请求编号相同噪声，原任务成功判据不变；原154800权重及仿真代码只读。

置零作用于模型实际history输入；HDF5仍记录真实传感器历史，不能把物理测量与干预后的输入混为一谈。运行时哈希及source audit见每回合result.json。初始化异常/资源截断不计完整失败。

## 解释边界与失败分析

M4不同权重在零触觉下已经产生动作分叉，所以单看M4成功率无法隔离在线触觉作用。M5在同权重下测试这种依赖，仍可能受到置零分布偏移影响；若真实输入更好，仅支持该固定策略在这些回合依赖在线触觉，并非证明学到正确滑移/力学机制。若相同或更差，则不能声称在线触觉有效。离线主效用门没有通过的事实保留，不因某回合成功翻转。

只有两个训练seed、一个验证seed、一个训练随机初始化和单种替换瓶几何。轨迹失败不能当专家恢复标签，后续扩展需独立数据、多个训练随机初始化、独立确认seed，以及失败恢复监督和时序/错配负对照。

## 每回合图文报告与复现

'''+ '\n'.join(f'- [seed{seed} / true权重零输入]({seed}_zero/report.md)' for seed in range(100051,100055))+f'''

协议 `{R}/docs/CWMJ_M5_INPUT_ABLATION_PROTOCOL_ZH.md`；运行器 `src/run_cwmj_m5_closed_loop.py`，启动器 `work/run_cwmj_m5_closed_loop_batch.sh`，汇总 `src/report_cwmj_m5_batch.py`。输出各自独立，不覆盖M4数据和权重。每回合自有GPU租约退出释放，不自动启动Strata。
'''
 (out/'report.md').write_text(text);dest=Path('/home/admin001/lingbot-tactile-artifacts/cwmj-m5-closed-loop');dest.mkdir(exist_ok=True)
 for name in ('summary.json','comparison.png','report.md'):shutil.copy2(out/name,dest/name)
 print(json.dumps({k:v for k,v in summary.items() if k not in ('rows','pairs')}))
if __name__=='__main__':report()
