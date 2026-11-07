"""An interrupted/unscorable attempt gets a report, not a fabricated task outcome."""
from pathlib import Path
import argparse,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--repeat',type=int,required=True);p.add_argument('--arm',required=True);p.add_argument('--code',type=int,required=True);a=p.parse_args()
name=f'r{a.repeat}_{a.seed}_{a.arm}';out=Path('/home/admin001/lingbot-tactile-artifacts/cwmj-m6-confirmation')/name;out.mkdir(parents=True,exist_ok=True)
fig,ax=plt.subplots(figsize=(7,3),layout='constrained');ax.text(.5,.5,'Unscorable infrastructure interruption\nNo task-success outcome inferred.',ha='center',va='center');ax.axis('off');fig.savefig(out/'interruption.png',dpi=160);plt.close(fig)
(out/'infrastructure_failure.md').write_text(f'''# M6独立确认基础设施异常：{name}

单回合启动器退出码{a.code}；缺少完整终态或可用报告，整批已停止。此状态不计作策略失败，不生成假的成功率或补跑结果。

![异常状态](interruption.png)

原始输出 `{R}/data/eval/cwmj_m6_confirm_{name}`，日志 `{R}/data/reports/cwmj_m6_confirmation/{name}.log`。保留全部部分文件，先检查存活进程、退出原因、GPU租约和物理时钟，再决定恢复方式；不能仅因观察超时重启或覆盖。原比赛仓库/checkpoint只读，未自动重启Strata。
''')
print('CONFIRMATION_INFRASTRUCTURE_REPORT',name)
