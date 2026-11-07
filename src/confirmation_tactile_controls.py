"""Fixed runtime controls for M6 held-out confirmation; no target fields read."""
import torch
from terminal_h50_contract import tactile_arm
CONDITIONS=('zero','true','mismatch','true_input_zero')
WEIGHT_ARM={'zero':'zero','true':'true','mismatch':'mismatch','true_input_zero':'true'}

def runtime_payload(current,condition,request_index,donor_pools):
    if condition not in CONDITIONS or type(request_index) is not int or request_index<0:raise ValueError('fixed condition and request index required')
    if condition=='true':return tactile_arm(current,'true'),None
    if condition in ('zero','true_input_zero'):return tactile_arm(current,'zero'),None
    # First invalid history must not be replaced with a valid historical donor.
    if not bool(current['tactile_valid'][0]):return tactile_arm(current,'true'),None
    contact=bool((current['tactile_history'][...,:2]>0).any());seeds=sorted(donor_pools)
    if not seeds:raise ValueError('frozen training donors required')
    source_seed=seeds[request_index%len(seeds)];group=donor_pools[source_seed][contact]
    if not group:raise ValueError('missing predeclared contact stratum')
    item=group[request_index%len(group)];meta=item['meta'];payload=item['tactile']
    if meta['fold']!='train' or meta['seed']!=source_seed or bool(meta['contact'])!=contact or not bool(payload['tactile_valid'][0]):raise ValueError('donor provenance or valid-window mismatch')
    return {k:v.clone() for k,v in payload.items()},{k:meta[k] for k in ('seed','frame_index','fold','contact')}

def load_frozen_training_donors(repo):
    import json
    from cwmj_m4_samples import M4TerminalEpisode
    from cwmj_m6_samples import M6TerminalEpisode
    from prepare_cwmj_m6_training_manifest import validate_split
    manifest=json.loads((repo/'data/reports/cwmj_m6/training_manifest.json').read_text())
    validate_split(manifest['train_seeds'],manifest['validation_seeds'],manifest['new_train_seeds'],manifest['new_validation_seeds'])
    pools={};mapping=[]
    for seed in sorted(manifest['train_seeds']):
        ep=M4TerminalEpisode(seed) if seed in (100045,100046) else M6TerminalEpisode(seed)
        expected=next(x['source_sha256'] for x in manifest['sources'] if x['seed']==seed)
        if ep.fold!='train' or ep.hashes!=expected:raise ValueError('runtime donor source changed')
        pools[seed]={False:[],True:[]}
        for i in range(len(ep)):
            s=ep.sample(i,False);x=s['input'];meta={k:s['metadata'][k] for k in ('seed','frame_index','fold','contact')}
            payload={'tactile_history':torch.as_tensor(x['tactile_history']).unsqueeze(0),'tactile_mask':torch.as_tensor(x['tactile_mask']).unsqueeze(0),'tactile_age_ms':torch.tensor([x['tactile_age_ms']]),'tactile_valid':torch.tensor([True])}
            pools[seed][bool(meta['contact'])].append({'meta':meta,'tactile':payload});mapping.append(meta)
    return pools,{'source_manifest':str(repo/'data/reports/cwmj_m6/training_manifest.json'),'donors_in_order':mapping,'selection':'seed=request_index mod training_seed_count, frame=request_index mod same-contact group size'}
