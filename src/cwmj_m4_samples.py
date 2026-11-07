"""Read-only same-run source audit and reader for preregistered 20261106 trials."""
from pathlib import Path
import json
from align_temporal import align
from h50_online_samples import OnlineH50Candidate
from h50_contact_readiness import assess
from validate_expert_data import read_jsonl,sha256,validate
from register_dataset import build_card
from record_online_expert_demo import audit_same_run_online_clock
from policy_highlevel_chunks import METHOD,ROLE,SOURCE_SHA,select_policy_h50
from preregister_online_expert_trial import SOURCE,TASK_SOURCE,ROBOT_SOURCE,TASK_SOURCE_SHA,ROBOT_SOURCE_SHA
from patch_online_expert_stride20_pilot import REPO

from prepare_cwmj_m4 import FOLDS
PROTOCOL={s:(20,f) for s,f in FOLDS.items()}

class M4Candidate(OnlineH50Candidate):
    def __init__(self,seed):
        if type(seed) is not int or seed not in PROTOCOL:raise ValueError('unregistered seed')
        stride,fold=PROTOCOL[seed];self.seed=seed;self.fold=fold
        self.trial_id=f'relay_pickcwmj_s{seed}_online_h50_stride{stride}_{fold}_expert0'
        self.trial=REPO/'data/paired'/self.trial_id
        if any(sha256(p)!=h for p,h in ((SOURCE,SOURCE_SHA),(TASK_SOURCE,TASK_SOURCE_SHA),(ROBOT_SOURCE,ROBOT_SOURCE_SHA))):
            raise ValueError('original simulator changed')
        if stride==20:
            from prepare_cwmj_m4 import generate
            private=REPO/f'work/cwmj_m4_{seed}'
        else:
            from prepare_contact_h50_batch import generate
            private=REPO/f'work/contact_h50_20261106_{seed}'
        for name,text in generate(seed).items():
            if (private/name).is_symlink() or (private/name).read_text()!=text:raise ValueError('collector source drift')
        if stride==10:
            manifest=json.loads((private/'source_manifest.json').read_text())
            if manifest['seed']!=seed or manifest['fold']!=fold or manifest['stride']!=stride:
                raise ValueError('frozen protocol manifest drift')
            for name,digest in manifest['files'].items():
                if sha256(private/name)!=digest:raise ValueError('generated collector hash drift')
            for name,digest in manifest['source_files'].items():
                if sha256(REPO/'src'/name)!=digest:raise ValueError('original planning source drift')
        raws=[p for p in (self.trial/'force').glob('*.jsonl') if not p.name.endswith(('.obs.jsonl','.frames.jsonl'))]
        if len(raws)!=1:raise ValueError('unique raw stream required')
        raw=raws[0];obs=raw.with_suffix('.obs.jsonl');frames=raw.with_suffix('.frames.jsonl')
        self.hdf5=self.trial/'demonstration/data/episode0.hdf5'
        self._sources={'raw':raw,'obs':obs,'frames':frames,'hdf5':self.hdf5,
            'online_actions':self.trial/'online_actions.jsonl','pilot_result':self.trial/'pilot_result.json',
            'completion':self.trial/'completion.json','success_gate':self.trial/'expert_success_gate.json'}
        scenario=self.trial/'scenario.json'
        asset=json.loads(scenario.read_text())
        from cwmj_bottle_factory import ASSET
        if (asset['manifest']!=str(ASSET/'manifest.json') or
            asset['manifest_sha256']!=sha256(ASSET/'manifest.json') or
            asset['factory_sha256']!=sha256(REPO/'src/cwmj_bottle_factory.py') or
            asset['actual_geometry']!='same_cwmj_wine_bottle_both_sides'):
            raise ValueError('CWMJ scenario provenance mismatch')
        self._sources['scenario']=scenario
        hashes={k:sha256(p) for k,p in self._sources.items()}
        source=validate(raw,obs,frames,self.hdf5,self._sources['completion'])
        card=build_card(self.trial,REPO)
        result=json.loads(self._sources['pilot_result'].read_text())
        complete=json.loads(self._sources['completion'].read_text());gate=json.loads(self._sources['success_gate'].read_text())
        header,*self.actions=read_jsonl(self._sources['online_actions'])
        if (source['seed']!=seed or card['seed']!=seed or header['seed']!=seed or header['trial_id']!=self.trial_id
            or result.get('planning_path_stride_indices')!=stride or result.get('goal_ok') is not True
            or result.get('planner_ok') is not True or result.get('success_candidate_not_training_approved') is not True
            or result.get('actual_take_action_calls')!=len(self.actions)
            or complete.get('online_action_sha256')!=hashes['online_actions']
            or gate.get('online_action_sha256')!=hashes['online_actions']
            or header.get('source_sha256')!={'original_base_task':SOURCE_SHA,'original_bottle_task':TASK_SOURCE_SHA,'original_robot_planner':ROBOT_SOURCE_SHA}):
            raise ValueError('source identity/controller/success gate mismatch')
        audit_same_run_online_clock(raw,self.actions,result['physics_steps'])
        self.frames=read_jsonl(frames)[1:]
        self.windows={w['observation_marker_index']:w for w in align(raw,obs)}
        prepared=[dict(a,trial_id=self.trial_id,completion_sha256=hashes['completion'],
                       original_base_task_sha256=SOURCE_SHA,source_model_role=ROLE) for a in self.actions]
        anchors=[]
        for i in range(len(self.actions)-49):
            frame=self.frames[i];w=self.windows[frame['obs_index']]
            if not w['quality']['window_valid']:continue
            select_policy_h50(prepared,{'trial_id':self.trial_id,'completion_sha256':hashes['completion'],
                'controller_method':METHOD,'source_model_role':ROLE,'original_base_task_sha256':SOURCE_SHA,
                'current_rgb_hdf5_sha256':hashes['hdf5'],'tactile_window_valid':True,'first_action_index':i,
                'capture_physics_step':self.actions[i]['before_physics_step'],
                'past_tactile_max_physics_step':w['quality']['last_source_step']},
                expected_trial=self.trial_id,expected_completion_sha=hashes['completion'],
                source_physics_steps=result['physics_steps'],fold=fold)
            anchors.append(i)
        self.anchors=tuple(anchors)
        self.report={'schema':'lingbot_cwmj_m4_source_audit_v1','seed':seed,'fold':fold,'stride':stride,
            'physics_steps':result['physics_steps'],'actual_take_action_calls':len(self.actions),
            'anchors':anchors,'source_file_sha256':hashes,'training_approved':False}
    def sample(self,index,*,include_images=True):
        s=super().sample(index,include_images=include_images)
        s['metadata'].update(seed=self.seed,trial_id=self.trial_id,split=self.fold+'_candidate',
            source_card=None,source_file_sha256=self.report['source_file_sha256'])
        return s

def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=PROTOCOL,required=True)
    a=p.parse_args();r=M4Candidate(a.seed)
    report=dict(r.report,contact_coverage=assess(r.sample(i,include_images=False) for i in range(len(r))))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()


from terminal_h50_samples import TerminalEpisode
class M4TerminalEpisode(TerminalEpisode):
    def __init__(self,seed):
        self.reader=M4Candidate(seed);self.seed=seed;self.fold=FOLDS[seed]
        r=self.reader
        self.anchors=tuple(i for i,f in enumerate(r.frames) if r.windows[f['obs_index']]['quality']['window_valid'])
        self.hashes={k:sha256(p) for k,p in r._sources.items()}
        self.trial_id=r.trial.name
