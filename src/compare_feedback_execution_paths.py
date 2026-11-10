"""Compare complete execution paths without conflating request/action indices."""
import json
from pathlib import Path
import numpy as np
from report_feedback_block_intervention import require_complete,R

def compare_actions(left,right):
    def first_difference(key):
        for i,(a,b) in enumerate(zip(left,right)):
            if not np.array_equal(key(a),key(b)):return i
        return None
    q=lambda a:np.concatenate([a['after'][s]['arm_actual_qpos'] for s in ('left','right')])
    target=lambda a:np.concatenate([a['after'][s]['arm_drive_targets'] for s in ('left','right')])
    fields={'command':lambda a:a['command'],'physics_interval':lambda a:[a['physics_before'],a['physics_after']],
            'actual_qpos':q,'drive_targets':target}
    differences={k:first_difference(fn) for k,fn in fields.items()}
    return dict(left_actions=len(left),right_actions=len(right),common_actions=min(len(left),len(right)),
                same_length=len(left)==len(right),first_difference_zero_based=differences,
                exact_full_execution_match=len(left)==len(right) and all(v is None for v in differences.values()))

def read_actions(path):
    rows=[json.loads(s) for s in path.read_text().splitlines()]
    if rows[-1]!=dict(event='terminal',complete=True):raise ValueError('complete trace required')
    return [x for x in rows if x['event']=='action']

def main():
    require_complete(R/'data/reports/feedback_block_intervention_v1')
    comparisons=[]
    for seed in (100145,100146):
        for arm in ('live','history_zero'):
            new=R/f'data/eval/feedback_block_intervention_v1/s{seed}_{arm}_cached_block'
            old=R/f'data/eval/fixed_weight_controls_v1/s{seed}_{arm}_c10'
            nv=json.loads((new/'result.json').read_text());ov=json.loads((old/'result.json').read_text())
            for v in (nv,ov):
                if v['status']!='completed' or v['resource_censored']:raise ValueError('uncensored completion required')
            for key in ('seed','arm','adapter_sha256','initial_scene','denoise_steps','temporal_ensemble','training_steps','repeat_index'):
                if nv[key]!=ov[key]:raise ValueError('unmatched comparison '+key)
            comparisons.append(dict(seed=seed,arm=arm,comparison=compare_actions(read_actions(new/'block_trace.jsonl'),read_actions(old/'prediction_trace.jsonl'))))
    out=R/'data/reports/feedback_execution_path_comparison_v1';out.mkdir(exist_ok=False)
    (out/'summary.json').write_text(json.dumps(comparisons,indent=2))
    table='\n'.join(f"|{x['seed']}|{x['arm']}|{x['comparison']['exact_full_execution_match']}|{x['comparison']['first_difference_zero_based']}|" for x in comparisons)
    (out/'report.md').write_text('''# 缓存执行与旧c10轨迹对齐

固定seed、触觉条件、适配器及初始场景，比较完整执行指令、物理区间、动作后真实qpos与驱动目标。索引为零基实际动作，不是模型请求。精确比较不使用容差；不同长度即不构成完整一致。全部8场完成前禁止运行。

|seed|触觉条件|完整执行一致|首次差异动作|
|---|---|---|---|
'''+table+'''

一致仅说明这些已记录字段未发现影响，不证明所有仿真内部状态相同。若存在差异，需要继续比较块边界输入与噪声；不能直接归因为触觉或额外渲染。本报告没有把两个开发seed扩充成独立样本。

复现：src/compare_feedback_execution_paths.py。原始日志只读，输出不覆盖旧报告。
''')
    print(out)
if __name__=='__main__':main()
