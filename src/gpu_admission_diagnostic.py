"""Pure admission diagnostics for future queues; never acquires or modifies leases."""
def reasons(info, *, owner, ancestor_pids, minimum_seconds=600):
    lease=info.get('lease') or {};issues=[]
    if info.get('state')!='held':issues.append('lease_not_held')
    if info.get('holder')!=owner:issues.append('owner_mismatch')
    if info.get('pid_alive') is not True:issues.append('holder_pid_not_alive')
    if lease.get('pid') not in ancestor_pids:issues.append('holder_not_ancestor')
    if info.get('remaining_s',0)<minimum_seconds:issues.append('insufficient_ttl')
    if any(x.get('owner')!=owner for x in info.get('waiting',[])):issues.append('other_gpu_waiter')
    return issues
