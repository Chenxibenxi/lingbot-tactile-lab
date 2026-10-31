"""CPU-only SAPIEN read-only contact logging overhead benchmark (box scene)."""
import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace as NS


def make_scene():
    import sapien
    scene = sapien.Scene(); scene.set_timestep(.004)
    b = scene.create_actor_builder(); b.add_box_collision(half_size=[.1, .1, .1]);
    base = b.build_kinematic(name="probe_finger_box")
    base.set_pose(sapien.Pose(p=[0, 0, .1]))
    b = scene.create_actor_builder(); b.add_box_collision(half_size=[.05, .05, .05]);
    top = b.build(name="probe_target_box")
    top.set_pose(sapien.Pose(p=[0, 0, .25]))
    return scene


def fake_robot():
    return NS(left_gripper=[(NS(child_link=NS(get_name=lambda: "probe_finger_box")), 1, 0)],
              right_gripper=[(NS(child_link=NS(get_name=lambda: "unused")), 1, 0)],
              left_fix_gripper_name=[], right_fix_gripper_name=[])


def run(steps, record, out):
    from force_probe import ForceProbe
    scene = make_scene(); robot = fake_robot()
    probe = ForceProbe(out, task="box", seed=0, episode=0, dt=.004, robot=robot) if record else None
    timings=[]
    for i in range(steps):
        t0=time.perf_counter()
        scene.step()
        if probe:
            probe.sample(scene, robot, 1)
        timings.append((time.perf_counter()-t0)*1e3)
    if probe: probe.close()
    return timings


def main():
    p=argparse.ArgumentParser(); p.add_argument("--steps",type=int,default=1000)
    p.add_argument("--out",type=Path,required=True); args=p.parse_args()
    with tempfile.TemporaryDirectory() as td:
        baseline = run(args.steps, False, td)
        recording = run(args.steps, True, td)
        result = {"schema":"phizju_force_probe_benchmark_v1", "steps":args.steps,
                  "baseline_ms_mean":statistics.mean(baseline), "record_ms_mean":statistics.mean(recording),
                  "baseline_ms_p95":sorted(baseline)[int(.95*(len(baseline)-1))],
                  "record_ms_p95":sorted(recording)[int(.95*(len(recording)-1))],
                  "jsonl_bytes":sum(f.stat().st_size for f in Path(td).glob("*.jsonl")),
                  "caveat":"CPU box test, not full RoboTwin model, different run order/physics states"}
    args.out.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__": main()
