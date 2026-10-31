"""Create a fresh, isolated RoboTwin overlay for paired EXPERT tactile demonstrations.

Keeps original task motion, physics stepping and success checks unchanged. Extends
E009's policy-only opt-in probe with the two expert physics loops and markers
at the exact successful _take_picture() pickle write. No baseline tree edits.
"""
import argparse
from pathlib import Path
from make_overlay import create_overlay, patched_source as basic_patch, INIT_INSERT


def replace_once(text, old, new):
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"RoboTwin source changed: expected exactly one anchor, got {count}: {old[:90]!r}")
    return text.replace(old, new)


def patched_source(source):
    text = basic_patch(source)
    # E009 creates its probe after load_actors but before check_stable; move
    # initialization AFTER check_stable. Pre-demo settling then remains outside
    # the expert timeline, with an explicit clock origin in the file header.
    text = replace_once(text, INIT_INSERT, "        self.load_actors()\n")
    stable = '        if self.eval_mode:\n            with open(os.path.join(CONFIGS_PATH, "_eval_step_limit.yml"), "r") as f:'
    initialized = '''        self._force_probe = None
        if self.save_data and os.environ.get("FORCE_PROBE_EXPERT_DIR"):
            from force_probe import ForceProbe
            dt = self.scene.get_timestep() if hasattr(self.scene, "get_timestep") else 1 / 250
            self._force_probe = ForceProbe(os.environ["FORCE_PROBE_EXPERT_DIR"],
                task=self.task_name, seed=kwags.get("seed", 0), episode=self.ep_num,
                dt=dt, robot=self.robot)

'''
    text = replace_once(text, stable, initialized + stable)
    # Start a new motion ID before any beginning-of-segment image is saved.
    together = '        save_freq = self.save_freq if save_freq == -1 else save_freq\n        if self.need_plan:'
    text = replace_once(text, together, '''        save_freq = self.save_freq if save_freq == -1 else save_freq
        if getattr(self, "_force_probe", None) is not None:
            self._force_probe.begin_motion("expert_together")
        if self.need_plan:''')
    dense = '        save_freq = self.save_freq if save_freq == -1 else save_freq\n        if save_freq != None:'
    text = replace_once(text, dense, '''        save_freq = self.save_freq if save_freq == -1 else save_freq
        if getattr(self, "_force_probe", None) is not None:
            self._force_probe.begin_motion("expert_dense")
        if save_freq != None:''')
    text = replace_once(text,
        '            self.scene.step()\n            if self.render_freq and i % self.render_freq == 0:',
        '''            self.scene.step()
            if getattr(self, "_force_probe", None) is not None:
                self._force_probe.sample(self.scene, self.robot, self._force_probe.motion_index)
            if self.render_freq and i % self.render_freq == 0:''')
    text = replace_once(text,
        '            self.scene.step()\n\n            if self.render_freq and control_idx % self.render_freq == 0:',
        '''            self.scene.step()
            if getattr(self, "_force_probe", None) is not None:
                self._force_probe.sample(self.scene, self.robot, self._force_probe.motion_index)

            if self.render_freq and control_idx % self.render_freq == 0:''')
    text = replace_once(text,
        '        pkl_dic = self.get_obs()\n        save_pkl(self.folder_path["cache"] + f"{self.FRAME_IDX}.pkl", pkl_dic)  # use cache\n        self.FRAME_IDX += 1',
        '''        pkl_dic = self.get_obs()
        # The training HDF5 retains the original baseline payload. Tactile data
        # and capture timestamps live ONLY in paired sidecar files.
        pkl_dic.pop("tactile_force", None)
        save_pkl(self.folder_path["cache"] + f"{self.FRAME_IDX}.pkl", pkl_dic)  # use cache
        if getattr(self, "_force_probe", None) is not None:
            self._force_probe.record_saved_frame(self.FRAME_IDX, pkl_dic, self.robot)
        self.FRAME_IDX += 1''')
    compile(text, "_base_task.py", "exec")
    return text


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--create",action="store_true")
    args=parser.parse_args()
    if args.create:
        print("CREATED",create_overlay(args.source,args.out,transform=patched_source))
    else:
        path=args.source.resolve(strict=True)/"envs/_base_task.py"
        patched_source(path.read_text())
        print("DRY_RUN_OK source untouched; use --create for new overlay")


if __name__=="__main__": main()
