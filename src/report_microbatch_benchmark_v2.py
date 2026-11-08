"""Read-only experiment audit plus fresh illustrated summary; no GPU imports."""
import argparse
import json
from pathlib import Path
import torch
from microbatch_comparison import compare
from validate_expert_data import sha256


def report(root):
    root=Path(root);base=root/'data/reports/microbatch_benchmark_v2'
    screen=root/'datasets/microbatch_numeric_screen_v1.json'
    limits=json.loads(screen.read_text())['limits']
    results={m:json.loads((base/f'micro{m}/result.json').read_text()) for m in (1,2,4)}
    comparisons={};reference=None;reference_grad=None
    for m,result in results.items():
        if result['status']!='completed':
            comparisons[m]={'passed':False,'reason':result.get('error','failed')};continue
        path=base/f'micro{m}'
        info=json.loads((path/'comparison.json').read_text())
        if info['identity']['numeric_screen_sha256']!=sha256(screen):raise ValueError('numeric screen changed')
        if info['identity']['plan_sha256']!=sha256(root/'datasets/microbatch_throughput_windows_v1.json'):
            raise ValueError('sample plan changed')
        for name,digest in result['code_sha256'].items():
            if sha256(root/name)!=digest:raise ValueError('benchmark code drift')
        grads=torch.load(path/'comparison_gradients.pt',map_location='cpu',weights_only=True)
        if m==1:
            reference=info;reference_grad=grads
        if reference is None:
            comparisons[m]={'passed':False,'reason':'micro1 reference failed'};continue
        if info['identity']!=reference['identity'] or info['router_tokens_per_sample']!=reference['router_tokens_per_sample']:
            comparisons[m]={'passed':False,'reason':'starting state, inputs or router shapes differ'};continue
        comparisons[m]=compare(reference_grad,grads,reference['loss'],info['loss'],limits)
    out=base/'summary';out.mkdir(exist_ok=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,key,title,scale in zip(axes,['samples_per_second','peak_allocated_bytes'],['Training samples / second','Peak allocated GiB'],[1,1024**3]):
        for m,r in results.items():
            if r['status']=='completed':
                ax.bar(m,r[key]/scale,color='#3b82f6' if comparisons[m]['passed'] else '#f59e0b')
            else:ax.text(m,0,'FAILED',ha='center')
        ax.set_xticks([1,2,4]);ax.set_title(title);ax.set_xlabel('Micro-batch (global batch = 4)')
    fig.tight_layout();fig.savefig(out/'throughput.png',dpi=160);plt.close(fig)
    summary=dict(results=results,comparisons=comparisons,limits=limits,
                 task_benefit_claim=False,report_code_sha256=sha256(Path(__file__)),
                 source_sha256={str(p.relative_to(root)):sha256(p) for p in base.glob('micro*/*') if p.is_file()})
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    lines=['# Micro-batch 1/2/4 吞吐基准','',
        '## 方法与数据','固定32个唯一训练窗口，global-batch=4，分别累积4/2/1次；每个配置独立进程，目标为8次计时更新；失败配置未必进入计时。所有配置统一关闭precompute_grid_thw，防止单样本网格缓存复用于多样本。预热后恢复相同适配器状态并使用新AdamW。固定noise/time；teacher、起始参数和输入清单必须匹配。原始仓库与checkpoint只读。','',
        '## 结果','![吞吐与峰值显存](throughput.png)','',
        '| Micro | 状态 | 样本/秒 | allocated GiB | 数值核验 |','|---|---|---:|---:|---|']
    for m,r in results.items():
        speed=f"{r['samples_per_second']:.3f}" if r['status']=='completed' else '—'
        memory=f"{r['peak_allocated_bytes']/1024**3:.2f}" if r['status']=='completed' else '—'
        lines.append(f"| {m} | {r['status']} | {speed} | {memory} | {comparisons[m]['passed']} |")
    lines+=['','## 失败分析与局限',
        '数值核验按运行前固定的loss、整体/逐参数梯度差和方向阈值执行。失败配置不推荐用于提速；错误与分项指标完整保存在summary.json。8步短基准只能衡量本机当前配置，不能代表长期稳态吞吐，也不构成触觉任务收益证据。CPU预处理与teacher生成耗时单列，图中仅表示计时训练段。',
        '', '## 复现与下一步',
        '固定输入见datasets/microbatch_throughput_windows_v1.json；阈值见datasets/microbatch_numeric_screen_v1.json。scripts/run_microbatch_benchmark_v2.sh依次接收1、2、4，各自获取自有GPU租约。输出目录为独占新目录，不覆盖历史试验。下一步仅在数值核验通过后考虑采用更高吞吐配置，并另行固定正式训练的曝光量和学习率；不修改本次阈值以追求通过。']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);a=p.parse_args();print(report(a.root))
