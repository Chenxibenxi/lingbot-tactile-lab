"""Partial batch index; never presents pending attempts as results."""
from pathlib import Path
import json
R=Path(__file__).resolve().parents[1]
def main():
 rows=[];accepted=0
 for seed in range(100055,100071):
  p=R/f'data/reports/cwmj_m6/seed_{seed}/result.json'
  if p.exists():
   r=json.loads(p.read_text());accepted+=r['status']=='accepted_success_source_verified';rows.append(f"| {seed} | {r['fold']} | {r['status']} | {r['valid_windows']} | {r['contact_windows']} | [图文报告](seed_{seed}/report.md) |")
  else:rows.append(f"| {seed} | {'train' if seed<100067 else 'validation'} | 待结束/尚未启动 | — | — | — |")
 text=f'''# M6固定采集批次：进度索引

这是未完成批次的状态快照，不能作为最终数据门结论。目前通过同源完整成功审计的回合：{accepted}。固定12训练/4验证各一次；训练数据门需至少4新训练、2新验证且各>=4接触/4非接触窗口。不会因失败补seed。

| seed | fold | 状态 | 有效窗口 | 接触窗口 | 报告 |
|---|---|---|---:|---:|---|
'''+ '\n'.join(rows)+'\n\n全部结束后生成report.md与data_manifest.json；当前不启动模型拟合。\n'
 dest=Path('/home/admin001/lingbot-tactile-artifacts/cwmj-m6');dest.mkdir(exist_ok=True);(dest/'progress.md').write_text(text)
 print('M6_ACCEPTED_SO_FAR',accepted)
if __name__=='__main__':main()
