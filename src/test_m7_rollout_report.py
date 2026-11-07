import unittest,tempfile,json,subprocess,sys
from pathlib import Path
from unittest.mock import patch
import h5py,numpy as np
import report_cwmj_m7_confirmation as reporter
class ReportTest(unittest.TestCase):
 def test_contact_state_cli_and_no_contact_caption(self):
  help_text=subprocess.check_output([sys.executable,str(Path(reporter.__file__)),'--help'],text=True)
  self.assertIn('contact_state',help_text);self.assertNotIn('true_input_zero',help_text)
  with tempfile.TemporaryDirectory() as d:
   r=Path(d);trial=r/'data/eval/cwmj_m7_confirm_r0_100087_contact_state';trial.mkdir(parents=True)
   (trial/'result.json').write_text(json.dumps(dict(status='completed',seed=100087,arm='contact_state',success=False,termination='official_action_limit',actions=400,physics_steps=10,requests=[dict(request_index=0,past_contact=False)])))
   with h5py.File(trial/'rollout.hdf5','w') as h:
    h['state_14']=np.zeros((1,14));h['applied/command_14']=np.zeros((400,14));h['images/head']=np.zeros((1,16,16,3),dtype='uint8')
   labels=[];original=reporter.plt.Axes.set_title
   def capture(ax,label,*args,**kwargs):labels.append(label);return original(ax,label,*args,**kwargs)
   with patch.object(reporter,'R',r),patch.object(reporter,'DEST',r/'published'),patch.object(reporter.plt.Axes,'set_title',capture):reporter.report(100087,'contact_state',0)
   self.assertTrue(any('No past-contact' in s for s in labels));self.assertFalse(any('First past-contact' in s for s in labels))
   p=r/'published/r0_100087_contact_state';text=(p/'report.md').read_text()
   self.assertIn('contact_state',text);self.assertNotIn('true_input_zero',text)
   self.assertTrue((p/'camera_frames.png').exists());self.assertTrue((p/'rollout_trace.png').exists())
if __name__=='__main__':unittest.main()
