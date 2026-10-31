"""Create a *separate* opt-in RoboTwin source overlay; never patch the original.

Run from the overlay as cwd with FORCE_PROBE_DIR set. All top-level source assets
are symlinks; only envs/_base_task.py is copied and instrumented. Do not publish
or submit this experimental overlay as an official benchmark implementation.
"""
from __future__ import annotations

import argparse
from pathlib import Path

INIT_ANCHOR = "        self.load_actors()\n"
INIT_INSERT = '''        self.load_actors()
        self._force_probe = None
        if os.environ.get("FORCE_PROBE_DIR"):
            from force_probe import ForceProbe
            dt = self.scene.get_timestep() if hasattr(self.scene, "get_timestep") else 1 / 250
            self._force_probe = ForceProbe(os.environ["FORCE_PROBE_DIR"],
                task=self.task_name, seed=kwags.get("seed", 0), episode=self.ep_num,
                dt=dt, robot=self.robot)
'''
STEP_ANCHOR = "            self.scene.step()\n            self._update_render()\n                \n            if self.check_success():"
STEP_INSERT = '''            self.scene.step()
            if self._force_probe is not None:
                self._force_probe.sample(self.scene, self.robot, self.take_action_cnt)
            self._update_render()

            if self.check_success():'''
OBS_ANCHOR = "        self.now_obs = deepcopy(pkl_dic)\n        return pkl_dic"
OBS_INSERT = '''        if getattr(self, "_force_probe", None) is not None:
            self._force_probe.record_observation(pkl_dic, self.take_action_cnt)
            pkl_dic["tactile_force"] = self._force_probe.observation()
        self.now_obs = deepcopy(pkl_dic)
        return pkl_dic'''
CLOSE_ANCHOR = "    def close_env(self, clear_cache=False):\n        if clear_cache:"
CLOSE_INSERT = '''    def close_env(self, clear_cache=False):
        if getattr(self, "_force_probe", None) is not None:
            self._force_probe.close()
            self._force_probe = None
        if clear_cache:'''


def patched_source(source: str) -> str:
    for old, new in ((INIT_ANCHOR, INIT_INSERT), (STEP_ANCHOR, STEP_INSERT),
                     (OBS_ANCHOR, OBS_INSERT), (CLOSE_ANCHOR, CLOSE_INSERT)):
        if source.count(old) != 1:
            raise RuntimeError(f"Expected exactly one known RoboTwin anchor, got {source.count(old)}: {old[:70]!r}")
        source = source.replace(old, new)
    compile(source, "_base_task.py", "exec")
    return source


def create_overlay(source: Path, dest: Path, transform=patched_source):
    source, dest = source.resolve(strict=True), dest.resolve()
    if not (source / "envs/_base_task.py").is_file():
        raise ValueError("Source has no RoboTwin envs/_base_task.py")
    if dest == source or source in dest.parents:
        raise ValueError("Overlay must be outside the original RoboTwin tree")
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError("Refusing non-empty overlay destination")
    # Validate the exact patch target BEFORE touching the destination.
    result = transform((source / "envs/_base_task.py").read_text())
    dest.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        if item.name == "envs":
            env = dest / "envs"; env.mkdir()
            for child in item.iterdir():
                target = env / child.name
                if child.name == "_base_task.py":
                    target.write_text(result)
                else:
                    target.symlink_to(child, target_is_directory=child.is_dir())
        elif item.name not in {"eval_result", "eval_runs", "logs", "data"}:
            (dest / item.name).symlink_to(item, target_is_directory=item.is_dir())
    (dest / "eval_result").mkdir(exist_ok=True)
    return dest / "envs/_base_task.py"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--create", action="store_true", help="Explicitly create overlay")
    args = parser.parse_args()
    if args.create:
        print("Created:", create_overlay(args.source, args.out))
    else:
        source = args.source.resolve(strict=True)
        patched_source((source / "envs/_base_task.py").read_text())
        print("DRY RUN: anchors valid; no files changed. Pass --create to build", args.out)


if __name__ == "__main__":
    main()
