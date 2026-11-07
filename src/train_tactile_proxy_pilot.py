"""Pre-registered bottle-v2 offline proxy, NOT VLA training or deployment approval.

Only reads same-run relay expert train and one validation episode. Never constructs
or samples the test episode. All artifacts are private, ignored by Git.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
from paired_samples import CAMERAS, PairedEpisode
from register_dataset import build_card
from validate_expert_data import sha256

TASK = 'pick_diverse_bottles'
TRAIN_SEEDS = (100000, 100002, 100007, 100013, 100015)
VALID_SEED = 100012
TEST_SEED = 100009
STATUS = 'exploratory_proxy_not_vla_not_approved'
STEPS = 120
BATCH = 64
LR = 1e-3
SEED = 590
MAX_SECONDS = 20 * 60
INSTRUCTION = 'Pick and place the two diverse bottles'
GRID = 8
VIS_DIM = len(CAMERAS) * GRID * GRID * 3
TOUCH_DIM = 5 * 4 * 3 + 5 * 4 + 1
JOINTS = (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12)
GRIPPERS = (6, 13)


def _json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def _safe_out(repo, out):
    path = Path(out).resolve()
    if path.parent not in ((repo / 'models').resolve(), (repo / 'data/reports').resolve()):
        raise ValueError('output must be a fresh direct child of ignored models/ or data/reports/')
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', path.name) or path.exists():
        raise ValueError('output must have a fresh safe basename')
    # Must be ignored even if repository ignore rules have changed.
    import subprocess
    result = subprocess.run(['git', 'check-ignore', '-q', str(path)], cwd=repo, check=False)
    if result.returncode != 0:
        raise ValueError('output directory is not Git-ignored')
    return path


def verify_sources(repo):
    """Check SHA-frozen index, original holdout and live same-run expert cards.

    HDF5 SHA is checked by build_card/PairedEpisode on train/validation only.
    The test trial is NEVER opened, even to check its card.
    """
    ds = repo / 'datasets'
    reg = _json(ds / 'provenance_v1.json')
    v1_path = ds / 'relay_pickdiverse_seed_split_candidate_v1.json'
    v2_path = ds / 'relay_pickdiverse_seed_split_candidate_v2.json'
    card_path = ds / 'relay_paired_training_index_candidate_v2.json'
    old_path = ds / 'relay_paired_training_index_candidate_v1.json'
    private = repo / 'data/reports/relay_paired_training_index_candidate_v2.jsonl'
    old_private = repo / 'data/reports/relay_paired_training_index_candidate_v1.jsonl'
    norm_path = repo / 'configs/relay_pickdiverse_train_norm_candidate_v2.json'
    v1, v2, index, old, norm = map(_json, (v1_path, v2_path, card_path, old_path, norm_path))
    if (reg.get('schema') != 'lingbot_tactile_dataset_registry_v1' or
            reg.get('approved_paired_episodes') != [] or
            index.get('schema') != 'lingbot_relay_paired_training_index_candidate_v1' or
            index.get('variant') != 'bottle_v2_add_train_keep_v1_heldout' or
            index.get('training_approval') is not False or
            index.get('status') != 'candidate_source_verified_not_training_approved' or
            index.get('source_machine_role') != 'relay_single_rtx3090' or
            index.get('provenance_registry_sha256') != sha256(ds / 'provenance_v1.json') or
            index.get('private_index_sha256') != sha256(private) or
            old.get('private_index_sha256') != sha256(old_private) or
            index.get('bottle_v1_split_sha256') != sha256(v1_path) or
            v2.get('parent_v1_split_sha256') != sha256(v1_path) or
            index['tasks'][TASK].get('split_sha256') != sha256(v2_path) or
            index['tasks'][TASK].get('train_only_normalization_sha256') != sha256(norm_path)):
        raise ValueError('source card/private index/old holdout SHA mismatch')
    if (v1.get('status') != 'candidate_not_approved' or
            v2.get('status') != 'candidate_not_approved' or
            v2.get('evaluation_claims_approved') is not False or
            norm.get('training_approval') is not False or
            norm.get('source_trials') != v2.get('train') or
            norm.get('split_candidate_sha256') != hashlib.sha256(
                json.dumps(v2, sort_keys=True).encode()).hexdigest()):
        raise ValueError('candidate approval/split/norm mismatch')
    groups = v2.get('task_seed_groups', {}).get(TASK, {})
    if (set(groups.get('train', [])) != set(TRAIN_SEEDS) or
            groups.get('validation') != [VALID_SEED] or groups.get('test') != [TEST_SEED] or
            len(v2.get('train', [])) != 5 or len(v2.get('validation', [])) != 1 or
            len(v2.get('test', [])) != 1 or
            sorted(v2['train']) != sorted(v1['train'] + v2.get('new_unassigned_parent_experts', [])) or
            v2['validation'] != v1['validation'] or v2['test'] != v1['test'] or
            v2['expert_card_sha256'][v2['test'][0]] != v1['expert_card_sha256'][v1['test'][0]] or
            v2['expert_card_sha256'][v2['validation'][0]] != v1['expert_card_sha256'][v1['validation'][0]]):
        raise ValueError('frozen heldout or train5 seeds changed')
    trials = v2['train'] + v2['validation']
    if any(not re.fullmatch(r'relay_pickdiverse_s\d+(?:_onsetpilot)?_expert\d+', t) for t in trials):
        raise ValueError('non-expert or malformed trial')
    for trial in trials:
        card_file = ds / 'cards' / f'{trial}.json'
        card = _json(card_file)
        if (sha256(card_file) != v2['expert_card_sha256'].get(trial) or
                card.get('trial_id') != trial or card.get('task') != TASK or
                card.get('seed') != int(trial.split('_s')[1].split('_')[0]) or
                card.get('seed') not in (TRAIN_SEEDS if trial in v2['train'] else (VALID_SEED,)) or
                build_card(repo / 'data/paired' / trial, repo) != card):
            raise ValueError('same-run expert card/HDF5 SHA mismatch')
    expected = index['tasks'][TASK]['folds']
    if expected['train']['pairs'] != 590 or expected['validation']['pairs'] != 126:
        raise ValueError('pre-registered 590/126 pair counts changed')
    rows = {'train': {}, 'validation': {}}
    # Hash of entire private index verified above; do not construct any test episode.
    with private.open(encoding='utf-8') as fp:
        for line in fp:
            # SHA covers all bytes, but test rows are not even deserialized.
            if re.search(r'"fold"\s*:\s*"test"', line):
                continue
            row = json.loads(line)
            if row.get('task') != TASK or row.get('fold') not in rows:
                continue
            fold = row['fold']; trial = row['trial_id']
            if (row.get('schema') != 'lingbot_private_relay_pair_index_v1' or
                    trial not in v2[fold] or row.get('seed') not in groups[fold] or
                    row.get('target_semantics') != 'next_saved_frame_expert_command_14_not_future_force_input'):
                raise ValueError('private row provenance/fold mismatch')
            key = (trial, row['current_frame_index'])
            if key in rows[fold]:
                raise ValueError('duplicate private index row')
            rows[fold][key] = row
    if any(len(rows[fold]) != expected[fold]['pairs'] for fold in rows):
        raise ValueError('private index train/validation pair count mismatch')
    return v2, rows, {'split_sha256': sha256(v2_path), 'private_index_sha256': sha256(private),
                       'old_private_index_sha256': sha256(old_private), 'old_holdout_sha256': sha256(v1_path),
                       'source_card_sha256': sha256(card_path), 'normalization_candidate_sha256': sha256(norm_path)}


def _pool(image):
    im = np.asarray(image)
    if im.shape != (240, 320, 3) or im.dtype != np.uint8:
        raise ValueError('invalid current RGB frame')
    return im.reshape(GRID, 30, GRID, 40, 3).mean(axis=(1, 3)).reshape(-1).astype(np.float32) / 255.


def load_fold(repo, v2, rows, fold):
    visual, state, touch, target, contact = [], [], [], [], []
    for trial in v2[fold]:
        episode = PairedEpisode(repo, trial, instruction=INSTRUCTION)
        if episode.card != _json(repo / 'datasets/cards' / f'{trial}.json'):
            raise ValueError('episode source changed since card verification')
        for i in range(len(episode)):
            sample = episode.sample(i, include_images=True)
            inp, meta = sample['input'], sample['metadata']
            key = (trial, meta['frame_index']); row = rows[fold].pop(key, None)
            if (row is None or meta['trial_id'] != trial or meta['task'] != TASK or
                    meta['seed'] != row['seed'] or meta['target_frame_index'] != row['next_expert_frame_index'] or
                    inp['capture_physics_step'] != row['capture_physics_step'] or
                    inp['capture_physics_step'] + meta['next_frame_step_delta'] != row['next_expert_physics_step'] or
                    not inp['tactile_window_valid'] or meta['next_frame_step_delta'] <= 0):
                raise ValueError('sample diverged from private same-run index')
            h = np.asarray(inp['tactile_history'], dtype=np.float32)
            mask = np.asarray(inp['tactile_mask'], dtype=np.float32)
            age = float(inp['tactile_age_ms'])
            s = np.asarray(inp['command_state_14'], dtype=np.float32)
            y = np.asarray(sample['target']['action_command_14'], dtype=np.float32)
            if (h.shape != (5, 4, 3) or mask.shape != (5, 4) or s.shape != (14,) or y.shape != (14,) or
                    not all(np.isfinite(x).all() for x in (h, mask, s, y)) or
                    not np.isfinite(age) or age < 0 or not np.all((mask >= 0) & (mask <= 1)) or
                    set(inp['images_rgb']) != set(CAMERAS)):
                raise ValueError('invalid causal feature or target')
            c = bool(np.any((h[..., :2] > 0).any(axis=-1) & (mask > 0)))
            if c != row['history_nonzero_contact']:
                raise ValueError('past-contact marker changed')
            visual.append(np.concatenate([_pool(inp['images_rgb'][cam]) for cam in CAMERAS]))
            state.append(s); touch.append((h, mask, age)); target.append(y); contact.append(c)
    if rows[fold]:
        raise ValueError('private index contains unmatched rows')
    return np.stack(visual), np.stack(state), touch, np.stack(target), np.asarray(contact, dtype=bool)


def make_features(train, val):
    """Fit normalization exclusively on train; keep touch groups separable for permutation."""
    x, s, t, y, c = train
    vx, vs, vt, vy, vc = val
    sm, ss = s.mean(0), np.maximum(s.std(0), 1e-3)
    ym, ys = y.mean(0), np.maximum(y.std(0), 1e-3)
    raw = np.stack([item[0] for item in t]); ages = np.array([item[2] for item in t])
    if np.any(raw[..., :2] < 0):
        raise ValueError('negative proxy force')
    lograw = raw.copy(); lograw[..., :2] = np.log1p(raw[..., :2])
    scale = np.maximum(lograw.std((0, 1)), .1)
    age_scale = max(float(np.max(ages)), 1.)

    def transform(batch, state_rows, tactile):
        hist = np.stack([item[0] for item in tactile]); masks = np.stack([item[1] for item in tactile])
        if np.any(hist[..., :2] < 0):
            raise ValueError('negative proxy force')
        hist = hist.copy(); hist[..., :2] = np.log1p(hist[..., :2])
        hist = np.clip(hist / scale, 0, 10) * masks[..., None]
        age = np.clip(np.array([item[2] for item in tactile])[:, None] / age_scale, 0, 10)
        features = np.concatenate((batch, np.clip((state_rows-sm)/ss, -10, 10)), axis=1)
        touch_features = np.concatenate((hist.reshape(len(batch), -1), masks.reshape(len(batch), -1), age), axis=1)
        if not np.isfinite(features).all() or not np.isfinite(touch_features).all():
            raise ValueError('nonfinite model input')
        return features.astype(np.float32), touch_features.astype(np.float32)

    tr, tt = transform(x, s, t); va, tv = transform(vx, vs, vt)
    return tr, tt, ((y-ym)/ys).astype(np.float32), va, tv, vy, vc, ym, ys


def permute_past(touch, seed=SEED):
    """One fixed derangement across validation samples; never permutes labels/RGB/state."""
    if len(touch) < 2:
        raise ValueError('need at least two validation samples for sham control')
    # A cyclic shift is a true permutation and excludes each row's own history.
    shift = int(np.random.default_rng(seed).integers(1, len(touch)))
    return np.roll(touch, shift, axis=0), shift


def _metrics(predictions, labels, contacts):
    errors = {name: np.abs(pred-labels) for name, pred in predictions.items()}
    result = {}
    for name, mask in (('past_contact', contacts), ('no_past_contact', ~contacts)):
        if not mask.any():
            raise ValueError('missing pre-registered contact stratum')
        def summary(values):
            return {'mae_14d': float(values.mean()), 'mae_12_joints': float(values[:, list(JOINTS)].mean()),
                    'mae_2_grippers': float(values[:, list(GRIPPERS)].mean())}
        result[name] = {'n': int(mask.sum()), 'arms': {arm: summary(err[mask]) for arm, err in errors.items()},
                        'paired_tactile_minus_visual': summary((errors['tactile']-errors['visual'])[mask]),
                        'paired_shuffled_minus_tactile': summary((errors['shuffled_past']-errors['tactile'])[mask])}
    return result


def run(repo, out, device='cuda', *, steps=STEPS):
    start = time.monotonic()
    repo = Path(repo).resolve(strict=True)
    out = _safe_out(repo, out)
    if steps < 1 or steps > STEPS:
        raise ValueError('optimizer step bound violated')
    if device not in ('cuda', 'cpu'):
        raise ValueError('device must be cuda or cpu')
    v2, rows, evidence = verify_sources(repo)
    train = load_fold(repo, v2, rows, 'train')
    val = load_fold(repo, v2, rows, 'validation')
    tr, tt, y, va, tv, vy, vc, ym, ys = make_features(train, val)
    if len(tr) != 590 or len(va) != 126 or int(vc.sum()) != 70:
        raise ValueError('pre-registered fold count or contact stratum mismatch')
    import torch
    from torch import nn
    torch.manual_seed(SEED)
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no implicit CPU fallback')
    # Identical architecture and initial parameters; visual arm has zeroed touch channels,
    # not an early-return adapter that skips trainable parameter paths.
    def network():
        return nn.Sequential(nn.Linear(tr.shape[1] + TOUCH_DIM, 128), nn.ReLU(),
                             nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 14))
    initial = network().state_dict()
    models = {}
    batch_order = np.random.default_rng(SEED).integers(0, len(tr), size=(steps, BATCH))
    labels = torch.from_numpy(y).to(device)
    for arm in ('visual', 'tactile'):
        net = network().to(device); net.load_state_dict(copy.deepcopy(initial))
        optimizer = torch.optim.Adam(net.parameters(), lr=LR)
        features = np.concatenate((tr, np.zeros_like(tt) if arm == 'visual' else tt), axis=1)
        data = torch.from_numpy(features).to(device)
        net.train()
        for indices in batch_order:
            if time.monotonic()-start > MAX_SECONDS:
                raise TimeoutError('20-minute exploratory proxy cap reached')
            ix = torch.as_tensor(indices, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = ((net(data[ix])-labels[ix])**2).mean()
            if not bool(torch.isfinite(loss).item()):
                raise ValueError('nonfinite loss')
            loss.backward(); optimizer.step()
        models[arm] = net.eval()
    shuffled, shift = permute_past(tv)
    predictions = {}
    with torch.no_grad():
        for arm, model, touch in (('visual', models['visual'], np.zeros_like(tv)),
                                  ('tactile', models['tactile'], tv),
                                  ('shuffled_past', models['tactile'], shuffled)):
            values = np.concatenate((va, touch), axis=1)
            predictions[arm] = model(torch.from_numpy(values).to(device)).cpu().numpy()*ys+ym
    if any(not np.isfinite(v).all() for v in predictions.values()):
        raise ValueError('nonfinite validation predictions')
    report = {'status': STATUS, 'task': TASK, 'not_vla': True, 'training_approval': False,
              'closed_loop_claim': False, 'test_episode_loaded': False, 'validation_scored_once': True,
              'train_seed_count': 5, 'train_pairs': 590, 'validation_seed': VALID_SEED,
              'validation_pairs': 126, 'optimizer_steps_per_arm': steps, 'single_seed': SEED,
              'batch_size': BATCH, 'learning_rate': LR, 'architecture': 'identical 128-64 MLP',
              'input_dimension_per_arm': int(tr.shape[1]+TOUCH_DIM), 'past_touch_dimension': TOUCH_DIM,
              'fixed_past_permutation_shift': shift, 'source_evidence_sha256': evidence,
              'validation': _metrics(predictions, vy, vc),
              'limitations': 'One validation physical seed, adjacent frames dependent; offline next-command proxy only. No LingBot/VLA, closed-loop benefit or deployment approval.'}
    if time.monotonic()-start > MAX_SECONDS:
        raise TimeoutError('20-minute exploratory proxy cap reached')
    out.mkdir(parents=False, exist_ok=False)
    torch.save({arm: {k: v.cpu() for k, v in model.state_dict().items()} for arm, model in models.items()}, out / 'proxy_weights.pt')
    (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True, help='fresh ignored directory directly within models/ or data/reports/')
    p.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    args = p.parse_args()
    report = run(args.repo, args.out, device=args.device)
    print(report['status'], 'validation', report['validation'])


if __name__ == '__main__':
    main()
