"""Explicitly authorize fixed M6 research fitting after complete new-data gate."""
import json
from cwmj_m6_samples import M6TerminalEpisode
from cwmj_m4_samples import M4TerminalEpisode
from prepare_cwmj_m6 import FOLDS,REPO
from validate_expert_data import sha256

def validate_split(train,val,new_train,new_val):
    if set(train)&set(val) or len(train)!=len(set(train)) or len(val)!=len(set(val)):raise ValueError('cross-fold or duplicate seed')
    if len(new_train)<4 or len(new_val)<2:raise ValueError('new data gate not met')
    if set(train)!=set(new_train)|{100045,100046} or set(val)!=set(new_val)|{100049}:raise ValueError('explicit prior/new merge required')
    if any(FOLDS.get(s)!='train' for s in new_train) or any(FOLDS.get(s)!='validation' for s in new_val):raise ValueError('heldout or wrong fold in new data')

def prepare():
    root=REPO/'data/reports/cwmj_m6';gate=json.loads((root/'data_manifest.json').read_text())
    if gate['passed'] is not True or gate['collection_protocol_sha256']!=sha256(REPO/'docs/CWMJ_M6_DATA_EXPANSION_PLAN_ZH.md'):raise ValueError('completed data gate/protocol required')
    nt=gate['new_train_seeds'];nv=gate['new_validation_seeds'];train=sorted(nt+[100045,100046]);val=sorted(nv+[100049]);validate_split(train,val,nt,nv)
    old=json.loads((REPO/'data/reports/cwmj_m4/training_manifest.json').read_text());sources=[]
    for seed in train+val:
        ep=(M4TerminalEpisode(seed) if seed in (100045,100046,100049) else M6TerminalEpisode(seed))
        source=next(x for x in (old if seed<100055 else gate)['sources'] if x['seed']==seed)
        if ep.hashes!=source['source_sha256']:raise ValueError('source changed since gate')
        samples=[ep.sample(i,False) for i in range(len(ep))];nc=sum(s['metadata']['contact'] for s in samples)
        if nc<4 or len(samples)-nc<4:raise ValueError('contact strata changed')
        sources.append({'seed':seed,'fold':ep.fold,'samples':len(samples),'contact_samples':nc,'source_sha256':ep.hashes})
    manifest={'schema':'cwmj_m6_training_manifest_v1','training_approval':True,'authorization':'user authorized GPU training and local simulation; approved scope fixed M6 protocol only','train_seeds':train,'validation_seeds':val,'new_train_seeds':nt,'new_validation_seeds':nv,'sources':sources,'data_gate_sha256':sha256(root/'data_manifest.json'),'training_protocol_sha256':sha256(REPO/'docs/CWMJ_M6_TRAINING_PROTOCOL_ZH.md'),'repetitions':[0,1,2],'arms':['zero','true','mismatch'],'steps_per_arm':720,'test_seeds':list(range(100071,100087))}
    with (root/'training_manifest.json').open('x') as f:json.dump(manifest,f,indent=2)
    print(json.dumps({'train':train,'validation':val,'training_samples':sum(s['samples'] for s in sources if s['fold']=='train'),'new_validation_count':len(nv)}))
if __name__=='__main__':prepare()
