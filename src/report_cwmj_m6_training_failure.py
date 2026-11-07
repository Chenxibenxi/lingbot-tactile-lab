"""Preserve failed training infrastructure attempts without claiming efficacy."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--repeat',type=int,required=True);a=p.parse_args();out=R/f'data/reports/cwmj_m6_training_r{a.repeat}'
def read(name,default):return json.loads((out/name).read_text()) if (out/name).exists() else default
failure=read('failure.json',{'type':'launcher_or_preflight_failure','error':'see full launch log'});logs=read('training_log.json',[]);dest=Path(f'/home/admin001/lingbot-tactile-artifacts/cwmj-m6-training/repeat_{a.repeat}/failure');dest.mkdir(parents=True,exist_ok=False)
fig,ax=plt.subplots(figsize=(8,3),layout='constrained')
if logs:ax.plot([r['loss'] for r in logs]);ax.set(xlabel='Available completed-arm log entries',ylabel='Masked loss')
else:ax.text(.5,.5,'No complete arm loss log available\nThis does not prove zero optimizer steps.',ha='center',va='center');ax.axis('off')
fig.savefig(dest/'available_training.png',dpi=160);plt.close(fig)
(dest/'report.md').write_text(f'''# M6训练重复{a.repeat}异常报告

## 方法与终止

按冻结M6三臂720步协议启动，原基座/checkpoint只读；启动器或训练过程非零退出，后续重复已停止，未将不完整输出作为效益结果。

```json
{json.dumps(failure,ensure_ascii=False,indent=2)}
```

![可用训练记录](available_training.png)

## 解释与后续

已保存完整arm日志条数{len(logs)}；日志只在arm结束落盘，不能由缺失日志断言未执行优化。查看原目录 `{out}` 和日志 `{out}.log` 区分资源预检、模型基础设施和训练数值错误。此前完成的arm报告保留，失败模型不进入确认。先诊断修复，任何重新运行使用独立输出目录并显式记录，不能覆盖这次失败或改数据门。异常不是触觉性能失败。
''')
print('M6_TRAINING_FAILURE_REPORTED',a.repeat)
