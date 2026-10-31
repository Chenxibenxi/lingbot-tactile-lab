"""Fail-closed validation of same-run RoboTwin expert HDF5+physics+saved-frame sidecars."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1<<20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(raw, obs, frames, hdf5, completion=None):
    import h5py
    import numpy as np
    raw,obs,frames,hdf5=map(Path,(raw,obs,frames,hdf5))
    rr,oo,ff=(read_jsonl(p) for p in (raw,obs,frames))
    if min(map(len,(rr,oo,ff)))<2:
        raise ValueError("empty physics/observation/frame stream")
    rh,oh,fh=rr[0],oo[0],ff[0]
    if (rh.get("schema"),oh.get("schema"),fh.get("schema")) != (
            "phizju_force_probe_v1","phizju_obs_marker_v1","phizju_saved_frames_v1"):
        raise ValueError("unsupported input stream schemas")
    if oh.get("probe_file")!=raw.name or fh.get("probe_file")!=raw.name:
        raise ValueError("sidecar references different physics stream")
    for key in ("task","seed","episode"):
        if rh[key]!=oh[key] or rh[key]!=fh[key]:
            raise ValueError("task/seed/episode mismatch")
    dt=float(rh["dt_s"])
    if not math.isfinite(dt) or dt<=0 or any(not math.isclose(float(h["dt_s"]),dt,abs_tol=1e-7) for h in (oh,fh)):
        raise ValueError("physics timestep mismatch")
    for n,row in enumerate(rr[1:],start=1):
        if row["physics_step"]!=n:
            raise ValueError(f"physics step gap or duplicate at {n}")
        if not math.isclose(float(row["sim_time_s"]),round(n*dt,8),abs_tol=2e-7):
            raise ValueError("physics timestamp drift")
        for side in ("left","right"):
            fingers=row.get(side,{}).get("fingers",{})
            if set(fingers)!=set(rh["side_links"][side]):
                raise ValueError("per-finger telemetry incomplete")
    markers={}
    last_obs_step=-1
    for n,row in enumerate(oo[1:]):
        if row.get("obs_index")!=n or row["physics_step"]<last_obs_step or row["physics_step"]>len(rr)-1:
            raise ValueError("observation index/step invalid")
        markers[n]=row
        last_obs_step=row["physics_step"]
    last_frame_step=-1
    for n,row in enumerate(ff[1:]):
        if row.get("frame_index")!=n:
            raise ValueError("noncontiguous saved frame index")
        marker=markers.get(row.get("obs_index"))
        if marker is None or row["physics_step"]!=marker["physics_step"]:
            raise ValueError("saved frame not linked to its camera observation")
        if row["physics_step"]<last_frame_step or row["physics_step"]>len(rr)-1:
            raise ValueError("saved frame out of physics timeline")
        if len(row.get("command_state_14",[]))!=14:
            raise ValueError("saved command state is not 14D")
        for side in ("left","right"):
            actual=row.get("actual_joints",{}).get(side,{})
            if not actual.get("names") or len(actual["names"])!=len(actual.get("qpos",[])) or len(actual["names"])!=len(actual.get("qvel",[])):
                raise ValueError("actual qpos/qvel missing or unmapped")
        last_frame_step=row["physics_step"]
    if not any(row["physics_step"]>0 for row in ff[1:]):
        raise ValueError("no saved frame contains instrumented expert motion")
    with h5py.File(hdf5,"r") as f:
        size=len(f["joint_action/vector"])
        if size != len(ff)-1:
            raise ValueError(f"HDF5/frame marker count mismatch: {size} != {len(ff)-1}")
        for cam in ("head_camera","left_camera","right_camera"):
            if len(f[f"observation/{cam}/rgb"]) != size:
                raise ValueError(f"HDF5 {cam} frame count mismatch")
        for i,row in enumerate(ff[1:]):
            if not np.allclose(f["joint_action/vector"][i],row["command_state_14"],rtol=0,atol=1e-5):
                raise ValueError(f"HDF5 command state != frame marker {i}")
    paths={"raw":raw,"obs":obs,"frames":frames,"hdf5":hdf5}
    hashes={name:sha256(path) for name,path in paths.items()}
    if completion is not None:
        manifest=json.loads(Path(completion).read_text())
        if manifest.get("schema")!="phizju_expert_episode_complete_v1" or not manifest.get("success"):
            raise ValueError("no successful completion manifest")
        if manifest.get("sha256")!=hashes or manifest.get("seed")!=rh["seed"] or manifest.get("episode")!=rh["episode"]:
            raise ValueError("completion manifest hashes/identity mismatch")
    return {"task":rh["task"],"seed":rh["seed"],"episode":rh["episode"],"physics_steps":len(rr)-1,
            "camera_observations":len(oo)-1,"saved_frames":len(ff)-1,"dt_s":dt,
            "saved_first_step":ff[1]["physics_step"],"saved_last_step":ff[-1]["physics_step"],"sha256":hashes}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("raw","obs","frames","hdf5"):
        p.add_argument("--"+key,required=True)
    p.add_argument("--completion")
    args=p.parse_args()
    print(json.dumps(validate(args.raw,args.obs,args.frames,args.hdf5,args.completion),indent=2))


if __name__=="__main__":main()
