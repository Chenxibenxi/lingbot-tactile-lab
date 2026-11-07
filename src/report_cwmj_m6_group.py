"""Save four-condition technical gate without changing experiment outcomes."""
from pathlib import Path
import json,argparse,shutil
from audit_cwmj_m6_pair import audit_pair,CONDITIONS
R=Path(__file__).resolve().parents[1]
def report(seed,repeat):
 paths={a:R/f'data/eval/cwmj_m6_confirm_r{repeat}_{seed}_{a}' for a in CONDITIONS};result=audit_pair(paths,seed,repeat);result.update(seed=seed,repeat=repeat)
 root=R/'data/reports/cwmj_m6_confirmation';root.mkdir(exist_ok=True);p=root/f'pair_r{repeat}_{seed}.json';p.write_text(json.dumps(result,indent=2));dest=Path('/home/admin001/lingbot-tactile-artifacts/cwmj-m6-confirmation');dest.mkdir(exist_ok=True);shutil.copy2(p,dest/p.name)
 if result['checks'] and not all(result['checks'].values()):raise ValueError('technical pairing failed; stop confirmation matrix')
 print('M6_GROUP_AUDIT',seed,repeat,result.get('eligible_complete_pair',False),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--repeat',type=int,required=True);a=p.parse_args();report(a.seed,a.repeat)
