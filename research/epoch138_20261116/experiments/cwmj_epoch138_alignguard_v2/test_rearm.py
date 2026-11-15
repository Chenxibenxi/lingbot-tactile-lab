import numpy as np
from alignment_guard import AlignmentGuard
# State-machine tests isolate timing and validity from kinematics.
g=AlignmentGuard.__new__(AlignmentGuard);g.contacted=False;g.no_contact_ticks=0;g.phase_start=0;g.rearms=0;g.prefix='fl';g.offset=0;g.fk=lambda *args:(None,np.zeros(3),np.eye(3));cmd=[0.]*14
valid=lambda x:{'sensor_valid':True,'per_finger':{'fl_link7':{'max_normal_N':x},'fl_link8':{'max_normal_N':0.}}}
g.apply(cmd,cmd,valid(.1),0);assert g.contacted
for t in range(1,5):g.apply(cmd,cmd,valid(0.),t);assert g.contacted
# invalid interval interrupts streak and cannot be interpreted as no contact
g.apply(cmd,cmd,{},5);assert g.contacted and g.no_contact_ticks==0
for t in range(6,10):g.apply(cmd,cmd,valid(0.),t);assert g.contacted
g.apply(cmd,cmd,valid(0.),10);assert not g.contacted and g.phase_start==10 and g.rearms==1
for t in range(11,20):g.apply(cmd,cmd,valid(0.),t)
assert g.rearms==1 and g.phase_start==10
g.apply(cmd,cmd,valid(.1),20);assert g.contacted
for t in range(21,26):g.apply(cmd,cmd,valid(0.),t)
assert not g.contacted and g.rearms==2 and g.phase_start==25
print('5-interval rearm, invalid interruption, reacquisition, nonrepeated resets passed')
