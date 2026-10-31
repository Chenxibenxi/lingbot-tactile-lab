"""Actual SAPIEN contact-impulse smoke, CPU physics, no robot or VLA inference.

Use the pinned RoboTwin Python. A box stands in for a finger for API/units
verification ONLY; success does not prove a real GelSight or real robot works.
"""
import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    import sapien
    from force_probe import sample_contacts

    scene = sapien.Scene()
    dt = 1 / 100.0
    scene.set_timestep(dt)
    builder = scene.create_actor_builder()
    builder.add_box_collision(half_size=[.1, .1, .1])
    finger = builder.build_kinematic(name="probe_finger_box")
    finger.set_pose(sapien.Pose(p=[0, 0, .1]))
    builder = scene.create_actor_builder()
    builder.add_box_collision(half_size=[.05, .05, .05])
    target = builder.build(name="probe_target_box")
    target.set_pose(sapien.Pose(p=[0, 0, .26]))
    readings = []
    for i in range(100):
        scene.step()
        item = sample_contacts(scene, {"left": {"probe_finger_box"}, "right": {"unused"}}, dt)
        readings.append(item["left"]["load_proxy_n"])
    result = {"schema": "phizju_contact_api_smoke_v1", "sapien_version": getattr(sapien, "__version__", "unknown"),
              "dt_s": dt, "steps": len(readings), "nonzero_steps": sum(x > 0 for x in readings),
              "peak_load_proxy_n": max(readings), "tail10_mean_load_proxy_n": sum(readings[-10:]) / 10,
              "note": "box-on-box contact impulse/dt only; NOT a real gripper force measurement"}
    with open(args.out, "w", encoding="utf8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    if result["nonzero_steps"] < 10 or result["tail10_mean_load_proxy_n"] <= 0:
        raise SystemExit("FAIL: no stable contacts or finite force proxy")


if __name__ == "__main__":
    main()
