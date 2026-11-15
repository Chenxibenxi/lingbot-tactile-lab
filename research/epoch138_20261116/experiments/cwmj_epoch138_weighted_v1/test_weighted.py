import copy,numpy as np
import weighted_selection as w
# Isolate aggregation from already-versioned decode contract.
def admit(d,validate):
 a=np.asarray(d['commands_14d']);return {'accepted':True,'command':a[0].tolist()}
w.admit_projected_first_target=admit
validate=lambda a,**kw:{'action':a}
h=[]
for t in range(9):
 a=np.array([[100*t+r]*14 for r in range(50)],float);h.append({'tick':t,'decoded':{'status':'DECODED','commands_14d':a.tolist()}})
before=copy.deepcopy(h)
for win in [1,3,5,9]:
 for decay in [.5,.85]:
  for tick in range(9):
   x=w.select_weighted(h[:tick+1],tick,{'window':win,'decay':decay},validate);e=x['ensemble'];n=min(win,tick+1);ages=list(range(n-1,-1,-1));weights=np.array([decay**a for a in ages]);weights/=weights.sum();expected=sum(weights[i]*(100*(tick-age)+age) for i,age in enumerate(ages))
   assert np.isclose(x['command'][0],expected);assert x['command'][6]==100*tick;assert x['command'][13]==100*tick;assert all(s['source_tick']+s['source_row']==tick for s in [{'source_tick':v['tick'],**v} for v in e['sources']]);assert np.isclose(sum(e['weights']),1)
assert before==h
for bad in [h[:-1],h[:7]+h[8:],h+[h[7]]]:
 try:w.select_weighted(bad,8,{'window':9,'decay':.5},validate)
 except ValueError:pass
 else:raise AssertionError('invalid history accepted')
print('72 aligned weighted combinations + mutation/gripper/history checks passed')
