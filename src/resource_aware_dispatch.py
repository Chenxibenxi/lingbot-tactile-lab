"""Fail-closed sequential dispatch for future protocols, independent of live queues.

Callbacks must preserve exclusive attempt output and the shell's atomic lease
acquisition/recheck. A preflight snapshot is not a GPU reservation.
"""

def dispatch(items, *, snapshot, launch, save, owner):
    completed=[]
    for item in items:
        state=snapshot()
        lease=state.get('lease_status',{})
        waiting=lease.get('waiting',[])
        if (state.get('probe_ok') is not True or state.get('compute_pids')
                or any(w.get('owner')!=owner for w in waiting)
                or lease.get('state') not in ('free','stale')):
            result=dict(status='paused',reason='resource_preflight',pending_item=item,
                        snapshot=state,completed=completed)
            save(result);return result
        attempt=launch(item)
        completed.append(dict(item=item,attempt=attempt))
        # Treat every technical error as actionable, never consume the rest of
        # the matrix trying to guess whether the shared resource has recovered.
        if (attempt.get('returncode')!=0 or attempt.get('status')!='completed'
                or attempt.get('resource_censored') is not False):
            result=dict(status='paused',reason='attempt_technical_error',failed_item=item,
                        snapshot=snapshot(),completed=completed)
            save(result);return result
    result=dict(status='completed',completed=completed);save(result);return result
