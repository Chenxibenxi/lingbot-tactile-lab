"""World points must match raw physics, SHA-bound same-run RGB and time."""
import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import h5py
import numpy as np
from validate_contact_geometry import validate_pair,build_card


class GeometryValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.repo=Path(self.tmp.name);self.trial=self.repo/'data/negative/trial'
        (self.trial/'force').mkdir(parents=True);(self.trial/'geometry').mkdir()
        (self.trial/'demonstration/data').mkdir(parents=True)
        self.raw=self.trial/'force/raw.jsonl'
        self.geometry=self.trial/'geometry/raw.geometry.jsonl'
        self.header={'schema':'phizju_force_probe_v1','task':'pick_diverse_bottles',
                     'seed':100000,'episode':0,'dt_s':.004,
                     'clock_origin':'first_instrumented_expert_step_after_setup_stability',
                     'side_links':{'left':['fl_link7','fl_link8'],'right':['fr_link7','fr_link8']}}
        self.gh={'schema':'lingbot_private_contact_geometry_v1','probe_file':'raw.jsonl',
                 'task':self.header['task'],'seed':self.header['seed'],'episode':0,
                 'dt_s':.004,'clock_origin':self.header['clock_origin']}
        self.raw_rows=[self.physics_row(1,1.),self.physics_row(2,0.)]
        self.geometry_rows=[{'physics_step':1,'points':[{'side':'left','finger':'fl_link7',
                'position_world_m':[0.,0.,1.],'load_proxy_n':1.}]},
                            {'physics_step':2,'points':[]}]
        self.save()

    @staticmethod
    def physics_row(step,proxy):
        return {'physics_step':step,'left':{'points':int(proxy>0),'load_proxy_n':proxy,
                'fingers':{f:{'points':int(proxy>0) if f=='fl_link7' else 0,
                              'load_proxy_n':proxy if f=='fl_link7' else 0.}
                           for f in ('fl_link7','fl_link8')}},
                'right':{'points':0,'load_proxy_n':0.,
                         'fingers':{f:{'points':0,'load_proxy_n':0.} for f in ('fr_link7','fr_link8')}}}

    def save(self):
        for path,rows in ((self.raw,[self.header]+self.raw_rows),
                          (self.geometry,[self.gh]+self.geometry_rows)):
            path.write_text(''.join(json.dumps(x,allow_nan=True)+'\n' for x in rows))

    def test_valid_private_points_match_each_physical_step_and_finger(self):
        pair=validate_pair(self.raw,self.geometry)
        self.assertEqual(pair['steps'],2)
        self.assertEqual(pair['point_counts'],{'left':1,'right':0})
        self.assertEqual(pair['step_points'][1],[])

    def test_gap_falsified_position_force_and_source_are_rejected(self):
        self.geometry_rows[1]['physics_step']=3;self.save()
        with self.assertRaisesRegex(ValueError,'gap'):validate_pair(self.raw,self.geometry)
        self.geometry_rows[1]['physics_step']=2
        self.geometry_rows[0]['points'][0]['position_world_m']=[0.,float('nan'),1.];self.save()
        with self.assertRaisesRegex(ValueError,'nonfinite'):validate_pair(self.raw,self.geometry)
        self.geometry_rows[0]['points'][0]['position_world_m']=[0.,0.,1.]
        self.geometry_rows[0]['points'][0]['load_proxy_n']=9.;self.save()
        with self.assertRaisesRegex(ValueError,'proxy disagrees'):validate_pair(self.raw,self.geometry)
        self.geometry_rows[0]['points'][0]['load_proxy_n']=1.
        self.gh['probe_file']='another_trial.jsonl';self.save()
        with self.assertRaisesRegex(ValueError,'another physics'):validate_pair(self.raw,self.geometry)

    def test_bound_parent_and_same_frame_hdf5_projection_is_not_occlusion_claim(self):
        parent={'task':'pick_diverse_bottles','seed':100000,'episode':0,
                'source_machine_role':'relay_single_rtx3090'}
        card=self.repo/'datasets/negative_diagnostics/trial.json';card.parent.mkdir(parents=True)
        card.write_text(json.dumps(parent))
        (self.trial/'force/raw.frames.jsonl').write_text('{}\n'+json.dumps({'physics_step':1})+'\n')
        K=np.array([[200.,0.,160.],[0.,200.,120.],[0.,0.,1.]])
        E=np.array([[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.]])
        with h5py.File(self.trial/'demonstration/data/episode0.hdf5','w') as f:
            for cam in ('head_camera','left_camera','right_camera'):
                group=f.require_group('observation/'+cam)
                group.create_dataset('rgb',data=np.zeros(1,dtype='S1'))
                group.create_dataset('intrinsic_cv',data=K[None])
                group.create_dataset('extrinsic_cv',data=E[None])
        with patch('validate_contact_geometry.negative_card',return_value=parent):
            record=build_card(self.repo,self.trial)
            self.assertEqual(record['same_step_in_frustum_candidates_by_camera'],
                             {'head_camera':1,'left_camera':1,'right_camera':1})
            self.assertEqual(record['status'],'private_same_run_frustum_projection_only_occlusion_unknown_not_training_or_eval')
            self.assertFalse(record['occlusion_validated'])
            self.assertFalse(record['accepted_for_expert_training'])
            # A camera captured at setup step 0 may not borrow step-1 future contact.
            (self.trial/'force/raw.frames.jsonl').write_text('{}\n'+json.dumps({'physics_step':0})+'\n')
            before_contact=build_card(self.repo,self.trial)
            self.assertEqual(before_contact['same_step_in_frustum_candidates_by_camera'],
                             {'head_camera':0,'left_camera':0,'right_camera':0})
            card.write_text('{}')
            with self.assertRaisesRegex(ValueError,'parent card differs'):
                build_card(self.repo,self.trial)


if __name__=='__main__':unittest.main()
