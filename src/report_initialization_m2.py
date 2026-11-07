import json,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1];ROOT=R/'data/reports/initialization_m2';DEST=Path('/home/admin001/lingbot-tactile-artifacts/initialization-m2')
def report(out):
    result=json.loads((out/'result.json').read_text());trace=json.loads((out/'trace.json').read_text())
    xyzq=np.asarray([t['poses'] for t in trace]);measure=[]
    fig,axes=plt.subplots(1,2,figsize=(10,3),layout='constrained')
    for j,side in enumerate(['left','right']):
        q=xyzq[:,j,3:];q=q/np.linalg.norm(q,axis=1,keepdims=True);angle=np.degrees(2*np.arccos(np.clip(np.abs(q@q[-1]),0,1)))
        measure.append({'side':side,'bottle_id':result['bottle_ids'][j],'max_rotation_last200_degrees':float(angle[-200:].max()),'z_final_m':float(xyzq[-1,j,2]),'z_range_last200_m':float(np.ptp(xyzq[-200:,j,2]))})
        axes[0].plot(xyzq[:,j,2],label=f"{side} ID{result['bottle_ids'][j]}")
        axes[1].plot(angle,label=side)
    axes[0].set_ylabel('Actor origin height (m)');axes[1].set_ylabel('Rotation to final orientation (deg)')
    axes[1].axhline(3,color='#b34c43',ls='--',label='Original 3 deg gate');axes[1].set_ylim(0,max(6,min(180,axes[1].get_ylim()[1])))
    for ax in axes:ax.axvspan(len(trace)-200,len(trace),color='#e5e7eb',alpha=.4);ax.set_xlabel('Stability-check physical step');ax.legend(fontsize=8)
    fig.suptitle(f"Seed {result['seed']} | unchanged setup gate: {result['setup']}");fig.savefig(out/'pose_trace.png',dpi=150);plt.close(fig)
    (out/'measurements.json').write_text(json.dumps(measure,indent=2))
    table='\n'.join(f"| {m['side']} | {m['bottle_id']} | {m['max_rotation_last200_degrees']:.4f} | {m['z_final_m']:.5f} | {m['z_range_last200_m']:.6f} |" for m in measure)
    experiment='M3 CWMJ资产初始化' if 'asset_replacement' in result else 'M2 初始化诊断'
    replacement_note=('本轮两瓶实际均为CWMJ酒瓶；瓶ID仅保留原随机抽样序号，不代表实际几何。原位置/朝向抽样保持，几何/质量/惯量及外观整套替换，属于新场景。资产manifest：'+str(result['asset_replacement'])+'。' if 'asset_replacement' in result else '本轮保留原瓶子资产。')
    text=f'''# {experiment}：seed{result['seed']}

## 目的和方法

对 M1 的同一 seed 做 setup-only 诊断重放，记录原稳定性检查期间的瓶子姿态。不是新的独立实验样本，不执行专家/模型动作，也不产生训练标签。只在原 check_stable 的每个 scene.step 后增加姿态读取，原2000步等待、500步检查、最后200步及3°旋转阈值完全保留。未修改原仓库。{replacement_note}

## 结果

原判据：**{result['setup']}**。瓶ID：{result.get('bottle_ids')}；记录{len(trace)}个检查物理步。

| 侧 | 瓶ID | 最后200步最大旋转° | 末态原点高度m | 最后200步高度极差m |
|---|---:|---:|---:|---:|
{table}

![姿态与原判据](pose_trace.png)

![初始化后头部相机](head.png)

## 解释边界和下一步

旋转相对最后一帧计算，与原稳定性函数的几何意义一致。这里是actor原点高度，不是瓶顶/功能点高度。灰色区域为原判据实际考察窗口；此前转动大不必然导致拒绝。异常：`{result.get('error','无')}`。相机异常：`{result.get('observation_error','无')}`。

本实验只定位初始化失效，不说明策略能力或触觉效果。下一步结合碰撞网格、质量/惯量及姿态分布分析，必要时按独立协议验证CWMJ资产；不得直接关闭门或把本次诊断接入监督数据。

## 复现

原检查与观察版源码SHA、模型ID及资产SHA见 result.json；完整2500步姿态在 trace.json。原始目录 `{out}`。GPU使用独立租约，不自动启动Strata。
'''
    (out/'report.md').write_text(text);DEST.mkdir(exist_ok=True);shutil.copytree(out,DEST/out.name,dirs_exist_ok=True)
    return measure
if __name__=='__main__':
    for out in sorted(ROOT.glob('seed_*')):
        if (out/'result.json').exists() and not (out/'report.md').exists():
            trace=json.loads((out/'trace.json').read_text())
            if len(trace)==2500:print(out.name,report(out),flush=True)
