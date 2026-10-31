"""Seed-grouped relay tactile split and TRAIN-ONLY feature normalization.

These functions prepare a study; they do not approve candidate trials for
training, start a GPU job, or infer timestamps from LeRobot's packaging FPS.
"""
from __future__ import annotations

from collections import defaultdict
import random

ROLE = "relay_single_rtx3090"
HOST = "admin001-MS-7D90"


def make_seed_splits(cards, *, random_seed=42, min_seeds_per_task=5):
    """Keep every trial with the same (task,actual_seed) in one split."""
    if min_seeds_per_task < 3:
        raise ValueError("need at least three independent seeds per task")
    cards = list(cards)
    if not cards:
        raise ValueError("no relay dataset cards")
    groups = defaultdict(lambda: defaultdict(list))
    seen_ids = set()
    for card in cards:
        if card.get("schema") != "lingbot_tactile_dataset_card_v1" or (
                card.get("source_machine_role"), card.get("source_hostname")) != (ROLE, HOST):
            raise ValueError("non-relay or unverified dataset card")
        task, seed, trial = card.get("task"), card.get("seed"), card.get("trial_id")
        if not isinstance(task, str) or not isinstance(seed, int) or not isinstance(trial, str):
            raise ValueError("invalid task/actual-seed/trial identity")
        if trial in seen_ids:
            raise ValueError("duplicate trial id")
        seen_ids.add(trial)
        groups[task][seed].append(trial)
    result = {"schema":"lingbot_tactile_seed_split_plan_v1", "status":"candidate_not_approved",
              "random_seed":random_seed, "source_machine_role":ROLE,
              "train":[], "validation":[], "test":[], "task_seed_groups":{}}
    for task, by_seed in sorted(groups.items()):
        seeds=sorted(by_seed)
        if len(seeds) < min_seeds_per_task:
            raise ValueError(f"{task}: {len(seeds)} distinct seeds < {min_seeds_per_task} required")
        random.Random(f"{random_seed}:{task}").shuffle(seeds)
        # Keep a minimum of one held-out seed per task for each split.
        n_holdout = max(1, round(len(seeds) * .2))
        if len(seeds) - 2*n_holdout < 1:
            n_holdout = 1
        val, test, train = seeds[:n_holdout], seeds[n_holdout:2*n_holdout], seeds[2*n_holdout:]
        if not train or not val or not test:
            raise ValueError("empty train/validation/test split")
        result["task_seed_groups"][task] = {"train":train,"validation":val,"test":test}
        for label, selection in (("train",train),("validation",val),("test",test)):
            for seed in selection:
                result[label].extend(sorted(by_seed[seed]))
    for label in ("train","validation","test"):
        result[label].sort()
    return result


def fit_train_only_stats(samples, train_trial_ids):
    """Robust median/IQR of log1p force and contact fraction on TRAIN trials.

    Samples must have the PairedEpisode.sample(include_images=False) structure.
    This function deliberately ignores provided validation/test samples.
    """
    import numpy as np
    train_ids = set(train_trial_ids)
    if not train_ids:
        raise ValueError("empty training split")
    values = []
    observed_trials = set()
    for sample in samples:
        trial = sample["metadata"]["trial_id"]
        if trial not in train_ids:
            continue
        x=sample["input"]
        if not x["tactile_window_valid"]:
            continue
        history=np.asarray(x["tactile_history"],dtype=np.float64)
        mask=np.asarray(x["tactile_mask"],dtype=np.float64)
        if history.shape != (5,4,3) or mask.shape != (5,4):
            raise ValueError("invalid tactile feature/mask shape")
        if not np.isfinite(history).all() or not np.isfinite(mask).all() or np.any(history < 0):
            raise ValueError("nonfinite/negative tactile training value")
        if np.any((mask < 0) | (mask > 1)) or np.any(history[:,:,2] > 1):
            raise ValueError("coverage/contact fraction out of bounds")
        transformed=history.copy()
        transformed[:,:,:2]=np.log1p(transformed[:,:,:2])
        for t in range(5):
            for f in range(4):
                if mask[t,f] == 1.0:  # partial bins stay masked, never imputed into statistics
                    values.append((f,transformed[t,f].copy()))
        observed_trials.add(trial)
    if observed_trials != train_ids:
        raise ValueError("not every requested train trial has a valid tactile sample")
    med=np.zeros((4,3)); iqr=np.ones((4,3)); counts=[]
    for finger in range(4):
        rows=np.asarray([v for f,v in values if f==finger],dtype=np.float64)
        if not len(rows):
            raise ValueError(f"finger {finger} has no complete tactile bins")
        med[finger]=np.median(rows,axis=0)
        q25,q75=np.percentile(rows,[25,75],axis=0)
        iqr[finger]=np.maximum(q75-q25,.1)
        counts.append(len(rows))
    return {"schema":"lingbot_tactile_train_norm_v1", "status":"fit_on_train_split_only",
            "source_trials":sorted(train_ids),"n_bins_by_finger":counts,
            "pretransform":"log1p on mean/peak proxy-N only; contact fraction unchanged",
            "median":med.tolist(),"scale_iqr_floor_0p1":iqr.tolist()}
