"""Generate an isolated visual-only RoboTwin client with opt-in self-rollout HDF5.

Never patch the baseline client. The new stream logs input RGB and actual
applied actions, NOT expert labels and NOT tactile-conditioned model inference.
Only a separately created writable RoboTwin overlay may load this copy.
"""
from pathlib import Path
import argparse

SOURCE=Path('/mnt/relaydata/lingbot/RoboTwin/script/eval_policy_client_lingbotvla.py.orig')
START='    st_seed = 100000 * (1 + seed)'
INFER='            ret = model.infer(formatted_observation) #(TASK_ENV, model, observation)'
APPLY_CHUNK='                    TASK_ENV.take_action(act)\n                    if TASK_ENV.eval_success:'
APPLY_SINGLE='                TASK_ENV.take_action(action)\n                if TASK_ENV.eval_success:'
FINISH='        now_id += 1\n        TASK_ENV.close_env(clear_cache=((succ_seed + 1) % clear_cache_freq == 0))'


def once(source,old,new):
    if source.count(old)!=1:
        raise ValueError(f'unsupported eval-client source anchor: {old[:65]!r} count={source.count(old)}')
    return source.replace(old,new)


def patched(source):
    source=once(source,START,'''    # Only the isolated replay may override the starting physical task seed.
    st_seed = int(os.environ.get("RELAY_REPLAY_START_SEED", 100000 * (1 + seed)))''')
    source=once(source,'        while TASK_ENV.take_action_cnt<TASK_ENV.step_lim and not succ:',
                '''        _rollout_writer = None
        while TASK_ENV.take_action_cnt<TASK_ENV.step_lim and not succ:''')
    source=once(source,INFER,'''            if os.environ.get("POLICY_SELF_ROLLOUT_CAPTURE_DIR"):
                from pathlib import Path
                from policy_rollout_capture import PolicyRolloutWriter
                probe = getattr(TASK_ENV, "_force_probe", None)
                if probe is None or probe.policy_window() is None:
                    raise RuntimeError("capture requires FORCE_PROBE_DIR and TACTILE_LIVE_WINDOW=1")
                capture_dir = Path(os.environ["POLICY_SELF_ROLLOUT_CAPTURE_DIR"]).resolve()
                eval_data = Path("/mnt/relaydata/lingbot-tactile-lab/data/eval").resolve()
                if eval_data not in capture_dir.parents or probe.output_dir.resolve().parent != capture_dir:
                    raise RuntimeError("capture/force directories must share one private relay trial root")
                if _rollout_writer is None:
                    name = f"policy_{probe.task}_{probe.seed}_{probe.episode}.hdf5"
                    _rollout_writer = PolicyRolloutWriter(capture_dir / name,
                        task=probe.task, seed=probe.seed, episode=probe.episode)
                marker = probe.record_policy_request()
                _rollout_writer.begin_request(marker, formatted_observation, probe.policy_window())
''' + INFER)
    source=once(source,APPLY_CHUNK,'''                    _before_step = probe.step if _rollout_writer is not None else None
                    TASK_ENV.take_action(act)
                    if _rollout_writer is not None:
                        _rollout_writer.applied_action(act,
                            physics_step_before=_before_step, physics_step_after=probe.step)
                    if TASK_ENV.eval_success:''')
    source=once(source,APPLY_SINGLE,'''                _before_step = probe.step if _rollout_writer is not None else None
                TASK_ENV.take_action(action)
                if _rollout_writer is not None:
                    _rollout_writer.applied_action(action,
                        physics_step_before=_before_step, physics_step_after=probe.step)
                if TASK_ENV.eval_success:''')
    source=once(source,FINISH,'''        if _rollout_writer is not None:
            _rollout_writer.close(observed_success=succ)
        now_id += 1
        TASK_ENV.close_env(clear_cache=((succ_seed + 1) % clear_cache_freq == 0))''')
    compile(source,'eval_policy_client_lingbotvla.py.orig','exec')
    return source


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=SOURCE)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--create',action='store_true')
    a=p.parse_args()
    source=a.source.resolve(strict=True);out=a.out.resolve()
    if Path(__file__).resolve().parents[1]/'work' not in out.parents or out==source:
        raise ValueError('patched client output must be new isolated work/ file')
    result=patched(source.read_text())
    if a.create:
        out.parent.mkdir(parents=True,exist_ok=True)
        with out.open('x') as f:f.write(result)
        print('CREATED_ISOLATED_POLICY_CAPTURE_CLIENT',out)
    else:print('DRY_RUN_CLIENT_AST_OK',out)


if __name__=='__main__':main()
