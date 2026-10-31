"""Create an isolated eval launcher that refuses curobo asset/.pth rewrites.

Only for copying the previously successful LingBot-VLA baseline into work/;
never run the stock launcher against the source-linked tactile overlay. Read-only
reference assets remain at their original absolute curobo locations.
"""
from __future__ import annotations
import argparse
from pathlib import Path

SOURCE=Path('/mnt/relaydata/lingbot/lingbot-vla-v2/experiment/robotwin/start_robotwin_infer_and_eval.sh')
START='# ===== Ensure curobo absolute paths point at this RoboTwin checkout ====='
END='total_start_time=$(date +%s)'
SAFE='''# ===== Isolated eval: verify baseline curobo paths; NEVER rewrite assets/.pth =====
_original_rt=/mnt/relaydata/lingbot/RoboTwin
for _asset_dir in embodiments objects background_texture; do
    [ -L "$eval_workdir/assets/$_asset_dir" ] &&
    [ "$(readlink -f "$eval_workdir/assets/$_asset_dir")" = "$(readlink -f "$_original_rt/assets/$_asset_dir")" ] || { echo 'UNSAFE eval asset symlink target'; exit 4; }
done
[ -f "$eval_workdir/script/eval_policy_client_lingbotvla.py.orig" ] || { echo 'UNSAFE missing original client'; exit 4; }
[ -f "$eval_workdir/script/deploy/msgpack_numpy.py" ] || { echo 'UNSAFE missing packer'; exit 4; }
[ -f "$eval_workdir/assets/embodiments/aloha-agilex/curobo_left.yml" ] || { echo 'UNSAFE missing curobo YAML'; exit 4; }
grep -qF "$_original_rt/assets" "$eval_workdir/assets/embodiments/aloha-agilex/curobo_left.yml" || { echo 'UNSAFE original curobo asset path absent'; exit 4; }
_env_site="$( source ${conda_sh} && conda activate ${sim_env} && python -c 'import site;print(site.getsitepackages()[0])' )"
_curobo_pth="$_env_site/__editable__.nvidia_curobo-0.7.8.pth"
[ -f "$_curobo_pth" ] && [ "$(head -1 "$_curobo_pth")" = "$_original_rt/envs/curobo/src" ] || { echo 'UNSAFE curobo .pth differs; abort without writes'; exit 4; }

'''


def patched(source):
    if source.count(START)!=1 or source.count(END)!=1:
        raise ValueError('unexpected launcher revision; refuse blind patch')
    before, rest=source.split(START,1)
    _, after=rest.split(END,1)
    result=before+SAFE+END+after
    if result.count('update_embodiment_config_path.py') or result.count('> "$_curobo_pth"'):
        raise ValueError('unsafe rewrite code remains')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=SOURCE)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--create',action='store_true')
    a=p.parse_args()
    src=a.source.resolve(strict=True); dest=a.out.resolve()
    if Path(__file__).resolve().parents[1]/'work' not in dest.parents or src==dest:
        raise ValueError('only write new isolated work/ launchers')
    result=patched(src.read_text())
    if a.create:
        dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open('x') as file:file.write(result)
        print('CREATED_ISOLATED_EVAL_LAUNCHER',dest)
    else:print('DRY_RUN_ISOLATED_EVAL_LAUNCHER_OK',dest)


if __name__=='__main__':main()
