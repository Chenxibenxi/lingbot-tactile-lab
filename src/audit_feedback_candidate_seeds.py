"""Read-only candidate-seed search in local JSON provenance; no execution approval."""
import argparse
import json
import re
from pathlib import Path
from validate_expert_data import sha256

CANDIDATES=set(range(100129,100145))
PATTERN=re.compile(r'(?<!\d)(100129|10013\d|10014[0-4])(?!\d)')


def seed_mentions(value,path=()):
    hits=[]
    if isinstance(value,dict):
        for key,child in value.items():hits.extend(seed_mentions(child,path+(str(key),)))
    elif isinstance(value,list):
        for i,child in enumerate(value):hits.extend(seed_mentions(child,path+(str(i),)))
    elif any('seed' in key.lower() or 'trial' in key.lower() for key in path):
        for match in PATTERN.finditer(str(value)):
            hits.append(dict(seed=int(match.group()),json_path='.'.join(path),value=str(value)[:300]))
    return hits


def audit(root,storage):
    roots=[Path(root)/'datasets',Path(root)/'data',Path(root)/'models',Path(storage)]
    files={}
    for base in roots:
        if not base.is_dir():raise ValueError('required audit root missing: '+str(base))
        for p in base.rglob('*.json'):
            # Resolve duplicates due to the explicit same-machine storage symlink.
            if p.is_file():files[str(p.resolve())]=p.resolve()
    evidence=[];hits=[];errors=[];path_hits=[]
    for p in sorted(files.values()):
        if PATTERN.search(str(p)):path_hits.append(str(p))
        try:
            raw=p.read_bytes();value=json.loads(raw)
            import hashlib
            evidence.append(dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
            for hit in seed_mentions(value):hits.append(dict(file=str(p),**hit))
        except (ValueError,OSError) as exc:errors.append(dict(file=str(p),error=type(exc).__name__+': '+str(exc)))
    return dict(schema='candidate_seed_json_provenance_audit_v1',candidate_seeds=sorted(CANDIDATES),
        roots=list(map(str,roots)),scanned_json_files=len(evidence),files=evidence,
        seed_or_trial_field_hits=hits,path_hits=path_hits,errors=errors,
        json_scope_clear=not hits and not errors and not path_hits,
        execution_approved=False,training_approval=False,
        limitation='Searches JSON seed/trial fields and JSON paths; not arbitrary logs, JSONL/HDF5 contents, remote historical machines or undocumented experiments. A clean result alone is not execution approval.')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);a=parser.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    result=audit(Path(__file__).resolve().parents[1],'/home/admin001/lingbot-tactile-storage')
    result['auditor_sha256']=sha256(Path(__file__))
    with a.out.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k not in ('files','seed_or_trial_field_hits')},ensure_ascii=False))
    print('Field hits:',len(result['seed_or_trial_field_hits']))
