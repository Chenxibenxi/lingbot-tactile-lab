"""Observe TOPP calls without changing arguments, outputs, or planner exceptions.

Logging errors are retained separately; caller must stop the trial after the
current action and classify it as a technical error, never silently accept it.
"""
from contextlib import contextmanager
import time
import numpy as np


class Trace:
    def __init__(self,emit,context):
        self.emit=emit;self.context=context;self.errors=[];self.calls=0
    def record(self,event):
        try:self.emit(event)
        except Exception as exc:self.errors.append(type(exc).__name__+': '+str(exc))
    def capture(self):
        try:return dict(self.context())
        except Exception as exc:
            self.errors.append(type(exc).__name__+': '+str(exc));return {'context_unavailable':True}
    def require_valid(self):
        if self.errors:raise RuntimeError('TOPP diagnostic capture failed: '+str(self.errors))


@contextmanager
def observe_topp(robot,emit,context):
    trace=Trace(emit,context);installed=[]
    try:
        owners=[robot.left_mplib_planner,robot.right_mplib_planner]
        if owners[0] is owners[1]:raise ValueError('distinct left/right planners required')
        for side,owner in zip(('left','right'),owners):
            original=owner.TOPP;had_local='TOPP' in vars(owner);local=vars(owner).get('TOPP')
            def wrapped(*args,_original=original,_side=side,**kwargs):
                call=trace.calls;trace.calls+=1
                start=time.monotonic();ctx=trace.capture()
                event=dict(event='topp_start',call=call,side=_side,context=ctx)
                try:
                    path=args[0] if args else kwargs['path']
                    value=np.asarray(path)
                    event.update(path=value.tolist(),path_dtype=str(value.dtype),path_shape=list(value.shape),
                        step=float(args[1] if len(args)>1 else kwargs.get('step',.1)))
                except Exception as exc:trace.errors.append(type(exc).__name__+': '+str(exc))
                trace.record(event)
                try:result=_original(*args,**kwargs)
                except Exception as exc:
                    trace.record(dict(event='topp_exception',call=call,side=_side,context=ctx,
                        exception=type(exc).__name__,error=str(exc),wall_seconds=time.monotonic()-start))
                    raise
                event=dict(event='topp_return',call=call,side=_side,context=ctx,wall_seconds=time.monotonic()-start)
                try:event.update(samples=int(result[1].shape[0]),duration_seconds=float(result[4]))
                except Exception as exc:trace.errors.append(type(exc).__name__+': '+str(exc))
                trace.record(event)
                return result
            owner.TOPP=wrapped;installed.append((owner,had_local,local))
        yield trace
    finally:
        for owner,had_local,local in reversed(installed):
            if had_local:owner.TOPP=local
            else:del owner.TOPP


def observed_action(task, command, *, emit, context):
    """Run the unchanged controller; surface recorder failures outside its fallback."""
    with observe_topp(task.robot, emit, context) as trace:
        result = task.take_action(command, action_type='qpos')
        trace.require_valid()
        return result
