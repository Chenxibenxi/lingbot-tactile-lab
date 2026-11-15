import copy,numpy as np
import weighted_selection as w
from mainline_first_target_contract import GRIPPER_REASON
h=[]
for t in range(9):
 a=np.zeros((50,14));a[:,0]=np.arange(50)+t*100;a[:,6]=np.arange(50)*.01+t*.08;a[:,13]=1.2-t*.1
 h.append({'tick':t,'decoded':{'status':'DECODE_REJECTED','reason':GRIPPER_REASON,'raw_commands_14d':a.tolist()}})
before=copy.deepcopy(h)
for win in [3,9]:
 for tick in range(9):
  x=w.select_weighted(h[:tick+1],tick,{'window':win,'decay':.85},lambda x,**kw:{'action':x});assert x['accepted'];src=x['ensemble']['sources'];weights=x['ensemble']['weights'];expect=np.zeros(14)
  for s,wt in zip(src,weights):
   a=np.array(h[s['tick']]['decoded']['raw_commands_14d'][tick-s['tick']]);a[[6,13]]=np.clip(a[[6,13]],0,1);expect+=wt*a
  assert np.allclose(x['command'],expect);assert 0<=x['command'][6]<=1 and 0<=x['command'][13]<=1
assert h==before
bad=copy.deepcopy(h);bad[-1]['decoded']['raw_commands_14d'][0][6]=float('nan');assert not w.select_weighted(bad,8,{'window':9,'decay':.85},lambda x,**k:{})['accepted']
print('18 actual decoder/projection/weighted gripper combinations + immutable sources/nonfinite rejection passed')
