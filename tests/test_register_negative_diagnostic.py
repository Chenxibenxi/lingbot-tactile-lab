"""Rejected same-run relay trajectories never become positive expert training data."""
import json
from pathlib import Path
import socket
import tempfile
import unittest
from register_negative_diagnostic import build_negative_card


def write_lines(path,items):
    path.write_text(''.join(json.dumps(item)+'\n' for item in items))


class RejectedTrialTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.repo=Path(self.tmp.name);self.trial=self.repo/'data/paired/failed0'
        (self.trial/'force').mkdir(parents=True)
        self.failure={'schema':'phizju_expert_trial_rejected_v1','accepted_for_training':False,
                      'source_machine_role':'relay_single_rtx3090','source_hostname':socket.gethostname(),
                      'planner_ok':False,'goal_ok':False,'probe_failed':False,'saved_frame_markers':2,
                      'task':'lift_pot','seed':0,'episode':0,'raw_basename':'trial0.jsonl'}
        (self.trial/'failure.json').write_text(json.dumps(self.failure))
        base={'task':'lift_pot','seed':0,'episode':0}
        self.raw=self.trial/'force/trial0.jsonl'
        write_lines(self.raw,[base,
            {'physics_step':1,'left':{'points':0},'right':{'points':0}},
            {'physics_step':2,'left':{'points':0},'right':{'points':0}}])
        write_lines(self.raw.with_suffix('.obs.jsonl'),[base,
            {'obs_index':0,'physics_step':0},{'obs_index':1,'physics_step':2}])
        write_lines(self.raw.with_suffix('.frames.jsonl'),[base,
            {'frame_index':0,'obs_index':0,'physics_step':0},
            {'frame_index':1,'obs_index':1,'physics_step':2}])

    def test_rejected_planner_card_keeps_no_rgb_or_training_claim(self):
        c=build_negative_card(self.trial,self.repo)
        self.assertEqual(c['status'],'quarantined_diagnostic_not_training_or_eval')
        self.assertEqual(c['physics_steps'],2)
        self.assertEqual(c['contact_physics_steps_by_side'],{'left':0,'right':0})
        self.assertFalse(c['has_validated_rgb_hdf5'])
        self.assertFalse(c['accepted_for_training'])
        self.assertFalse(c['suitable_as_grasp_failure_example'])
        self.assertEqual(set(c['source_sha256']),{'failure','raw','obs','frames'})

    def test_success_markers_lxc_or_noncontiguous_physics_refused(self):
        (self.trial/'completion.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'success markers'):
            build_negative_card(self.trial,self.repo)
        (self.trial/'completion.json').unlink()
        self.failure['source_machine_role']='LXC';(self.trial/'failure.json').write_text(json.dumps(self.failure))
        with self.assertRaisesRegex(ValueError,'local relay'):
            build_negative_card(self.trial,self.repo)
        self.failure['source_machine_role']='relay_single_rtx3090'
        (self.trial/'failure.json').write_text(json.dumps(self.failure))
        rows=[{'task':'lift_pot','seed':0,'episode':0},
              {'physics_step':1,'left':{'points':0},'right':{'points':0}},
              {'physics_step':3,'left':{'points':0},'right':{'points':0}}]
        write_lines(self.raw,rows)
        with self.assertRaisesRegex(ValueError,'noncontiguous'):
            build_negative_card(self.trial,self.repo)


if __name__=='__main__':unittest.main()
