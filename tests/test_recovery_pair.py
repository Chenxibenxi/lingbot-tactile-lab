import json
from pathlib import Path
import tempfile
import unittest
import h5py
import numpy as np
from audit_recovery_pair import audit
from validate_expert_data import sha256


class RecoveryPair(unittest.TestCase):
    def test_prefix_change_fails_pair(self):
        with tempfile.TemporaryDirectory() as d:
            paths=[]
            for condition in ('expert_control','left_open_once_recovery'):
                p=Path(d)/condition;(p/'demonstration/data').mkdir(parents=True);paths.append(p)
                (p/'result.json').write_text(json.dumps(dict(seed=100119,condition=condition,status='completed')))
                (p/'scenario.json').write_text('{"synthetic":true}')
                with h5py.File(p/'demonstration/data/episode0.hdf5','w') as h:
                    h.create_dataset('joint_action/vector',data=np.zeros((1,14)))
                    for cam in ('head_camera','left_camera','right_camera'):h.create_dataset(f'observation/{cam}/rgb',data=np.zeros((1,2,2,3),dtype=np.uint8))
                events=[]
                for i in range(4):
                    row=dict(action_index=i,source_role='external_intervention' if condition!='expert_control' and i==2 else 'expert',
                        command_state_14=[0.]*14,before_physics_step=i*10,after_physics_step=(i+1)*10)
                    events.extend([dict(event='before_action',**row),dict(event='after_action',**row)])
                (p/'action_journal.jsonl').write_text('\n'.join(json.dumps(e) for e in events))
                (p/'force').mkdir()
                sources={'hdf5':p/'demonstration/data/episode0.hdf5'}
                for key,suffix in [('raw','.jsonl'),('obs','.obs.jsonl'),('frames','.frames.jsonl')]:
                    sources[key]=p/'force'/('synthetic'+suffix);sources[key].write_text('synthetic source fixture')
                result=json.loads((p/'result.json').read_text())
                result.update(action_journal_sha256=sha256(p/'action_journal.jsonl'),source_audit={'sha256':{k:sha256(q) for k,q in sources.items()}})
                (p/'result.json').write_text(json.dumps(result))
            self.assertTrue(audit(*paths,100119)['technical_comparable'])
            p=paths[1]/'action_journal.jsonl';events=[json.loads(x) for x in p.read_text().splitlines()]
            original=p.read_text()
            for event in events:event['source_role']='expert'
            p.write_text('\n'.join(json.dumps(e) for e in events))
            result_path=paths[1]/'result.json';result=json.loads(result_path.read_text())
            result['action_journal_sha256']=sha256(p);result_path.write_text(json.dumps(result))
            untriggered=audit(*paths,100119)
            self.assertTrue(untriggered['technical_comparable'])
            self.assertFalse(untriggered['intervention_applied'])
            self.assertEqual(untriggered['compared_prefix_actions'],4)
            p.write_text(original)
            result['action_journal_sha256']=sha256(p);result_path.write_text(json.dumps(result))
            events=[json.loads(x) for x in original.splitlines()]
            events[1]['command_state_14'][0]=1.
            p.write_text('\n'.join(json.dumps(e) for e in events))
            with self.assertRaisesRegex(ValueError,'journal changed'):audit(*paths,100119)
            result_path=paths[1]/'result.json';result=json.loads(result_path.read_text())
            result['action_journal_sha256']=sha256(p);result_path.write_text(json.dumps(result))
            self.assertFalse(audit(*paths,100119)['checks']['pre_intervention_commands_equal'])
