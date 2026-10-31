"""Generate, but NEVER install, opt-in tactile RoboTwin client from known source.

Output belongs in this repository's ignored work/ folder. It is not ready to
run with the old inference server: TACTILE_POLICY_INFER=1 requires server
metadata advertising tactile_v1, and the model/server still need explicit
training and inference wiring. No official source file is modified here.
"""
from __future__ import annotations
import argparse
from pathlib import Path

CONNECT='    model = WebsocketClientPolicy(port=usr_args[\'port\'])'
INFER='            ret = model.infer(formatted_observation) #(TASK_ENV, model, observation)'


def patch_client(source):
    if source.count(CONNECT) != 1 or source.count(INFER) != 1:
        raise ValueError('unsupported RoboTwin eval client anchors; refuse blind patch')
    source=source.replace(CONNECT,CONNECT+'''
    if os.environ.get("TACTILE_POLICY_INFER") == "1":
        from tactile_rpc import require_tactile_capability
        require_tactile_capability(model.get_server_metadata())''')
    source=source.replace(INFER,'''            if os.environ.get("TACTILE_POLICY_INFER") == "1":
                from shadow_forward import build_shadow_request
                probe = getattr(TASK_ENV, "_force_probe", None)
                if probe is None or probe.policy_window() is None:
                    raise RuntimeError("TACTILE_POLICY_INFER needs FORCE_PROBE_DIR + TACTILE_LIVE_WINDOW=1")
                probe.record_policy_request()
                formatted_observation = build_shadow_request(formatted_observation, probe, enabled=True)
''' + INFER)
    compile(source,'eval_policy_client_lingbotvla.py.orig','exec')
    return source


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('/mnt/relaydata/lingbot/RoboTwin/script/eval_policy_client_lingbotvla.py.orig'))
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--create',action='store_true',help='write new file only in this repo work/')
    args=p.parse_args()
    source=args.source.resolve(strict=True)
    repo=Path(__file__).resolve().parents[1]
    out=args.out.resolve()
    if repo/'work' not in out.parents or source==out:
        raise ValueError('output must be a new file inside this repo work/')
    result=patch_client(source.read_text())
    if args.create:
        out.parent.mkdir(parents=True,exist_ok=True)
        with out.open('x',encoding='utf-8') as file:
            file.write(result)
        print('CREATED_EXPERIMENTAL_CLIENT',out)
    else:
        print('DRY_RUN_PATCH_OK; original untouched; pass --create to write',out)


if __name__=='__main__':main()
