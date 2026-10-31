"""Load actual Aloha-AgileX URDF into a tiny CPU scene and verify gripper mapping.

No VLA inference, no training, no task evaluation, no GPU rendering. An optional
probe cube placed at a finger's frame validates a live SAPIEN gripper contact.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace as NS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--urdf", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    import sapien
    from force_probe import gripper_links, sample_contacts
    scene = sapien.Scene()
    dt = 1 / 250
    scene.set_timestep(dt)
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    robot_entity = loader.load(str(args.urdf.resolve(strict=True)))
    if robot_entity is None:
        raise RuntimeError("URDF loader returned no articulation")
    def joint(name):
        result = robot_entity.find_joint_by_name(name)
        if result is None:
            raise RuntimeError("Missing gripper joint " + name)
        return (result, 1, 0)
    robot = NS(left_gripper=[joint("fl_joint7"), joint("fl_joint8")],
               right_gripper=[joint("fr_joint7"), joint("fr_joint8")],
               left_fix_gripper_name=[], right_fix_gripper_name=[],
               left_entity=robot_entity, right_entity=robot_entity)
    links = gripper_links(robot)
    # Place a collision cube at the actual left finger frame; this is ONLY a
    # contact/API smoke, not a grasp performance evaluation.
    finger = robot.left_gripper[0][0].child_link
    scene.step()  # populate articulation link world poses before positioning probe
    builder = scene.create_actor_builder()
    builder.add_box_collision(half_size=[.035, .035, .035])
    probe_cube = builder.build_kinematic(name="probe_cube")
    probe_cube.set_pose(sapien.Pose(p=finger.get_entity_pose().p))
    initial_pose = {"finger": list(map(float, finger.get_entity_pose().p)),
                    "cube": list(map(float, probe_cube.get_pose().p))}
    seen = 0; peak = 0.; raw_pairs = set()
    robot_links = {link.get_name() for link in robot_entity.get_links()}
    for _ in range(12):
        scene.step()
        raw_pairs.update((c.bodies[0].entity.name, c.bodies[1].entity.name) for c in scene.get_contacts())
        value = sample_contacts(scene, links, dt, robot_links=robot_links)
        if "probe_cube" in value["left"]["objects"]:
            seen += 1
        peak = max(peak, value["left"]["load_proxy_n"])
    result = {"schema": "phizju_robot_force_probe_smoke_v1",
              "links": {k: sorted(v) for k,v in links.items()},
              "dt_s": dt, "steps": 12, "left_cube_contact_steps": seen,
              "left_peak_load_proxy_n": peak, "initial_pose": initial_pose,
              "raw_contact_body_pairs": sorted([list(v) for v in raw_pairs]),
              "note": "artificial kinematic cube at URDF finger link; NOT a calibrated gripper-force measurement"}
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not seen:
        raise SystemExit("FAIL: robot gripper contact was not observable; inspect cube placement")


if __name__ == "__main__":
    main()
