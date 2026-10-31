"""Fail closed when Strata still reports an active generation phase.

busy=false and queued=0 alone are insufficient if phase='thinking'. This
module must run AFTER a GPU lease and BEFORE any service unload/stop.
"""
from __future__ import annotations
import json
import sys

SAFE_PHASES={None,'idle','ready','loaded','unloaded'}


def require_idle(status):
    if not isinstance(status,dict):raise ValueError('Strata status must be an object')
    if status.get('busy') is not False or type(status.get('queued')) is not int or status['queued']!=0:
        raise ValueError('Strata has an active or queued request or malformed status')
    if status.get('phase') not in SAFE_PHASES:
        raise ValueError(f'Strata phase {str(status.get("phase"))[:40]!r} is not safely idle')
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
