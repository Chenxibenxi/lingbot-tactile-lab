"""Independent source-index arithmetic plus HDF5 executed-command audit."""
import json
from pathlib import Path
import numpy as np
import h5py

def require(ok,message):
    if not ok:raise ValueError(message)

def audit(trace,hdf5):
    rows=[json.loads(s) for s in Path(trace).read_text().splitlines()]
    require(len(rows)>=4 and rows[0].get('schema')=='feedback_block_trace_v1' and rows[-1]==dict(event='terminal',complete=True),'complete schema')
    header=rows[0];mode=header['mode'];size=header['block_size']
    require(mode in ('fresh_first','cached_block') and type(size) is int and 1<=size<=50,'mode/block size')
    body=rows[1:-1];require(len(body)%2==0,'paired events')
    blocks=[];step=0
    with h5py.File(hdf5) as h:
        require(bool(h.attrs.get('complete',False)),'incomplete HDF5')
        n=len(body)//2
        for key in ('request/physics_step','applied/command_14','applied/request_index','applied/index_in_request','applied/physics_step_before','applied/physics_step_after'):
            require(len(h[key])==n,'HDF5 count '+key)
        for i in range(n):
            p,a=body[2*i:2*i+2]
            require(p['event']=='prediction' and a['event']=='action' and p['index']==i,'event sequence')
            b=np.asarray(p['commands']);require(b.shape==(50,14) and np.isfinite(b).all(),'prediction shape/finite');blocks.append(b)
            src=i if mode=='fresh_first' else i//size*size
            offset=0 if mode=='fresh_first' else i%size
            expected=dict(execution_source=mode,action_index=i,current_request_index=i,
                          source_request_index=src,source_prediction_offset=offset,block_size=size)
            require(a['source']==expected,'source provenance')
            require(np.array_equal(a['command'],blocks[src][offset]),'source command')
            require(p['physics_step']==step==a['physics_before']==int(h['request/physics_step'][i])==int(h['applied/physics_step_before'][i]),'before clock')
            require(a['physics_after']>step and a['physics_after']==int(h['applied/physics_step_after'][i]),'after clock')
            require(int(h['applied/request_index'][i])==i and int(h['applied/index_in_request'][i])==0,'HDF5 request mapping')
            require(np.array_equal(np.asarray(a['command'],dtype=h['applied/command_14'].dtype),h['applied/command_14'][i]),'HDF5 command')
            step=a['physics_after']
    return dict(status='passed',actions=n,mode=mode,block_size=size,training_approval=False)
