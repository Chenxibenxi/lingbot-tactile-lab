"""Only a contiguous technically complete prefix can be skipped on resume."""
def remaining(matrix, receipts, started):
    matrix=list(matrix)
    if len(set(matrix))!=len(matrix):raise ValueError('unique registered matrix required')
    if set(receipts)-set(matrix) or set(started)-set(matrix):raise ValueError('unregistered attempt')
    completed=0
    for item in matrix:
        receipt=receipts.get(item)
        if receipt is None:break
        if (receipt.get('returncode')!=0 or receipt.get('status')!='completed'
                or receipt.get('resource_censored') is not False):
            raise ValueError('technical failure requires separate recovery protocol, never silent retry')
        if item not in started:raise ValueError('receipt without started attempt')
        completed+=1
    pending=matrix[completed:]
    if any(item in receipts or item in started for item in pending):
        raise ValueError('incomplete started attempt or noncontiguous receipts; inspect without retry')
    return pending
