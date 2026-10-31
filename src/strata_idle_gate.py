"""Check Strata's actual busy/queued fields, not its stale phase label.

Source serve/server.py sets busy=False after completion but retains the last
phase='thinking'; /v1/unload separately holds FIFO and atomically refuses a
new active/queued request. Run AFTER taking a GPU lease, before atomic unload.
"""
from __future__ import annotations
import json
import sys

def require_idle(status):
    if not isinstance(status,dict):raise ValueError('Strata status must be an object')
    if status.get('busy') is not False or type(status.get('queued')) is not int or status['queued']!=0:
        raise ValueError('Strata has an active or queued request or malformed status')
    # Never gate on 'phase': Strata does not clear it after setting busy=False.
    # This read-only check is NOT a substitute for /v1/unload's atomic FIFO gate.
    return True


def main():
    try:
        require_idle(json.load(sys.stdin))
    except (ValueError,TypeError,json.JSONDecodeError) as exc:
        print('STRATA_IDLE_GATE_REFUSED',str(exc),file=sys.stderr)
        return 4
    print('STRATA_IDLE_GATE_OK')
    return 0


if __name__=='__main__':sys.exit(main())
