"""Complete fixed four-seed paired report, preserving all failures/exclusions."""
from pathlib import Path
import json,shutil,math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
R=Path(__file__).resolve().parents[1];ROOT=R/'data/reports/cwmj_m4_closed_loop';DEST=Path('/home/admin001/lingbot-tactile-artifacts/cwmj-m4-closed-loop')
def report():
    rows=[];pairs=[];matrix=np.full((2,4),-1,dtype=int);texts={};derived=[]
    for j,seed in enumerate(range(100051,100055)):
        for i,arm in enumerate(['zero','true']):
            p=R/f'data/eval/cwmj_m4_{seed}_{arm}/result.json';r=json.loads(p.read_text());rows.append(r)
            full=r['status']=='completed' and r['termination'] in ['official_action_limit','task_success']
            matrix[i,j]=int(r['success']) if full else -1;texts[i,j]=('SUCCESS' if full and r['success'] else 'FAIL' if full else 'EXCLUDED')+f"\n{r['actions']} actions"
            d={'seed':seed,'arm':arm,'full_episode':full,'success':r.get('success'),'reason':r.get('termination',r.get('exception')),'actions':r['actions'],'physics_steps':r['physics_steps'],'contact_requests':sum(q['past_contact'] for q in r['requests'])}
            if 'final_bottle_points' in r:
                for side,pose,target in zip(['left','right'],r['final_bottle_points'],[[-.06,-.105],[.06,-.105]]):
                    d[side+'_xy_max_error_m']=float(np.max(np.abs(np.asarray(pose[:2])-target)));d[side+'_final_z_m']=pose[2]
            derived.append(d)
        pairs.append(json.loads((ROOT/f'paired_{seed}.json').read_text()))
    fig,ax=plt.subplots(figsize=(9,3),layout='constrained');ax.imshow(matrix,cmap=ListedColormap(['#bfc2c8','#e1b38a','#80b49a']),vmin=-1,vmax=1)
    ax.set_xticks(range(4),[str(s) for s in range(100051,100055)]);ax.set_yticks([0,1],['No force adapter','True tactile adapter'])
    for (i,j),t in texts.items():ax.text(j,i,t,ha='center',va='center',fontsize=10)
    ax.set_title('Fixed CWMJ closed-loop pairs | original task success criterion');fig.savefig(ROOT/'paired_outcomes.png',dpi=160);plt.close(fig)
    paired_valid=[p for p in pairs if all(p['outcomes'][a] is not None for a in ['zero','true']) and all(p.get(k,False) for k in ['initial_rgb_and_state_equal','first10_actions_equal','invalid_tactile_first_both'])]
    complete_seeds={seed for seed in range(100051,100055) if all(d['full_episode'] for d in derived if d['seed']==seed)}
    paired_complete=[p for p in paired_valid if p['seed'] in complete_seeds]
    true_only=sum(p['outcomes']['true'] and not p['outcomes']['zero'] for p in paired_complete)
    zero_only=sum(p['outcomes']['zero'] and not p['outcomes']['true'] for p in paired_complete)
    discordant=true_only+zero_only
    exact_p=min(1.,2*sum(math.comb(discordant,k) for k in range(min(true_only,zero_only)+1))/2**discordant) if discordant else 1.
    successes={arm:sum(d['full_episode'] and d['success'] for d in derived if d['arm']==arm) for arm in ['zero','true']}
    full_counts={arm:sum(d['full_episode'] for d in derived if d['arm']==arm) for arm in ['zero','true']}
    summary={'successes':successes,'full_episode_counts':full_counts,'paired_initial_valid':len(paired_valid),'paired_complete_count':len(paired_complete),'true_only_success':true_only,'zero_only_success':zero_only,'exploratory_exact_p_two_sided':exact_p,'paired_results':pairs,'episode_diagnostics':derived,'general_improvement_proven':False}
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2))
    table='\n'.join(f"| {d['seed']} | {d['arm']} | {d['success']} | {d['reason']} | {d['actions']} | {d['physics_steps']} | {d['contact_requests']} |" for d in derived)
    pose_table='\n'.join(f"| {d['seed']} | {d['arm']} | {d.get('left_xy_max_error_m',float('nan')):.3f} | {d.get('left_final_z_m',float('nan')):.3f} | {d.get('right_xy_max_error_m',float('nan')):.3f} | {d.get('right_final_z_m',float('nan')):.3f} |" for d in derived)
    text=f'''# M4 CWMJ独立成对闭环：完整图文报告

## 主要结果

固定seed100051–100054全部按协议运行，每seed zero/true各一次。无力组成功{successes['zero']}/{full_counts['zero']}个完整回合；真实触觉组成功{successes['true']}/{full_counts['true']}。通过初始RGB/状态/首10动作逐元素一致性检查的配对数：{len(paired_valid)}。

完整且通过配对门的回合对数为{len(paired_complete)}；真实触觉独有成功{true_only}，无力组独有成功{zero_only}。仅以不一致配对计算的双侧精确二项检验值为{exact_p:.4f}（探索性，未作多重检验校正；小样本不显著也不证明等效）。

**四个固定seed仍是探索性结果，不能证明稳定总体收益；不按结果换seed或调步数。** 初始化异常和资源截断不计为完整失败。原20瓶场景与本CWMJ单瓶几何场景不混算。

![所有配对结果](paired_outcomes.png)

| seed | arm | 任务成功 | 终止/异常 | 真实动作 | 物理步 | 含过去接触请求 |
|---|---|---|---|---:|---:|---:|
{table}

## 原目标各分量

两个瓶都必须满足各自水平目标误差严格小于0.1m且功能点高度严格大于0.89m。表是最终功能点，不是actor原点。只说明终态，不据此断言全过程从未抬起或具体碰撞原因。

| seed | arm | 左水平最大误差m | 左高度m | 右水平最大误差m | 右高度m |
|---|---|---:|---:|---:|---:|
{pose_table}

## 方法与对照可信度

使用M4四组训练中预先指定zero/true的最后360步checkpoint，未根据验证MAE选模型。原400动作/10步去噪/10动作重规划，按seed和请求编号固定噪声；原任务成功判据不修改。每次模型加载核验六片154800 SHA、恢复非持久buffers、使用可复现MoE路径；两组相同CWMJ资产factory/manifest。只输入当前RGB/状态与过去触觉，首个接触处做同噪声置零影子预测但不执行。

每个回合独立校验动作/HDF5/力物理时钟。初始无触觉窗口两组输出相同是必要配对门；通过仅说明技术对照可靠，不说明触觉有效。完整逐请求影子差异与全部终态见各报告及summary.json。

## 逐实验图文报告

'''+ '\n'.join(f"- [seed{seed} / {arm}](%s/report.md)"%f'{seed}_{arm}' for seed in range(100051,100055) for arm in ['zero','true'])+f'''

## 局限与下一步

只有2个训练seed、1个验证seed、1个训练随机初始化和4个闭环seed；成功专家数据缺少失败恢复监督。原始100ms触觉窗口、每10动作重规划可能不足以捕获短暂滑移，需另设触觉响应频率/恢复数据实验，不能由本次终态直接确认原因。后续先基于视频、接触时钟和目标分量定位失败阶段，再冻结下一批恢复协议与负对照，保留本次结果不覆盖。

本轮离线结果与此闭环结论分别报告，动作MAE变化不等价成功率。原始回合在 `{R}/data/eval/cwmj_m4_*`，权重、模型代码、资产SHA和报告链可追溯。GPU释放自身租约，不自动启动Strata。
'''
    (ROOT/'report.md').write_text(text)
    for name in ['summary.json','paired_outcomes.png','report.md']:shutil.copy2(ROOT/name,DEST/name)
    print(json.dumps({'successes':successes,'full_counts':full_counts,'paired_initial_valid':len(paired_valid)}))
if __name__=='__main__':report()
