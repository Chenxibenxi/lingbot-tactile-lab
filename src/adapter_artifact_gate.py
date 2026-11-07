"""CPU-only *negative* deployment preflight: no fake tactile_v1 from a SHA.

A future approved training run must supply private safetensors and a locally
verifiable run manifest. This cannot itself prove policy efficacy or replace
human research approval; currently all REAL relay splits/norms remain candidate.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re
import numpy as np
from safetensors import safe_open

_HEX64=re.compile(r'[0-9a-f]{64}\Z')


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def _scoped(repo,name,scope):
    if not isinstance(name,str):raise ValueError('artifact path must be a string')
    path=Path(name)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('artifact paths must be repo-relative and nonescaping')
    result=repo/path
    if result.is_symlink() or not result.is_file() or not result.resolve(strict=True).is_relative_to(repo/scope):
        raise ValueError('missing or out-of-scope artifact file')
    return result


def _verify_train_expert_sources(repo,folds,train):
    """Rebuild expert SHA gates: registry strings alone are not train evidence."""
    from register_dataset import build_card as build_expert_card
    card_hashes=folds.get('expert_card_sha256')
    if not isinstance(card_hashes,dict):
        raise ValueError('approved split lacks SHA-bound expert cards')
    actual_train_seed_groups={}
    for trial_id in train:
        if not isinstance(trial_id,str) or not re.fullmatch(r'relay_[A-Za-z0-9_-]+_expert[0-9]+',trial_id):
            raise ValueError('train contains nonexpert or malformed trial id')
        card_path=_scoped(repo,f'datasets/cards/{trial_id}.json','datasets')
        card=json.loads(card_path.read_text())
        if (card_hashes.get(trial_id)!=sha256(card_path) or
                card.get('trial_id')!=trial_id or
                card.get('source_machine_role')!='relay_single_rtx3090' or
                card.get('data_path_relative')!=f'data/paired/{trial_id}' or
                not isinstance(card.get('task'),str) or type(card.get('seed'))!=int or
                build_expert_card(repo/'data/paired'/trial_id,repo)!=card):
            raise ValueError('approved train expert card or live RGB/physics/success gate changed')
        actual_train_seed_groups.setdefault(card['task'],set()).add(card['seed'])
    groups=folds.get('task_seed_groups')
    if not isinstance(groups,dict) or set(groups)!=set(actual_train_seed_groups):
        raise ValueError('approved split lacks matching actual train task seed groups')
    for task,by_fold in groups.items():
        if not isinstance(by_fold,dict):raise ValueError('invalid task seed groups')
        sets=[]
        for name in ('train','validation','test'):
            values=by_fold.get(name)
            if not isinstance(values,list) or any(type(x)!=int for x in values) or len(set(values))!=len(values):
                raise ValueError('duplicate/malformed actual seed group')
            sets.append(set(values))
        if sets[0]!=actual_train_seed_groups[task] or any(sets[i]&sets[j] for i,j in ((0,1),(0,2),(1,2))):
            raise ValueError('actual physical seed disagrees with train expert cards or leaked between folds')


def verify_deployment_artifact(repo,checkpoint,manifest,expected_sha):
    repo=Path(repo).resolve(strict=True)
    if not isinstance(expected_sha,str) or not _HEX64.fullmatch(expected_sha):
        raise ValueError('actual adapter SHA256 required')
    weights=_scoped(repo,checkpoint,'models');record_path=_scoped(repo,manifest,'models')
    if weights.suffix!='.safetensors' or not 0<weights.stat().st_size<100_000_000:
        raise ValueError('only bounded private safetensors accepted')
    if sha256(weights)!=expected_sha:raise ValueError('adapter bytes differ from expected SHA256')
    record=json.loads(record_path.read_text())
    if (record.get('schema')!='lingbot_tactile_deployment_evidence_v1' or
            record.get('status')!='approved_for_deployment' or
            record.get('source_machine_role')!='relay_single_rtx3090' or
            record.get('tactile_conditioned') is not True or
            type(record.get('trained_optimizer_steps'))!=int or record['trained_optimizer_steps']<=0 or
            record.get('adapter_sha256')!=expected_sha or
            not isinstance(record.get('base_checkpoint_sha256'),str) or
            not _HEX64.fullmatch(record['base_checkpoint_sha256'])):
        raise ValueError('no approved relay tactile training/adapter evidence')
    provenance=_scoped(repo,'datasets/provenance_v1.json','datasets')
    split=_scoped(repo,record.get('split_path'),'datasets')
    norm=_scoped(repo,record.get('normalization_path'),'configs')
    if any(record.get(key)!=sha256(path) for key,path in
           (('provenance_sha256',provenance),('split_sha256',split),('normalization_sha256',norm))):
        raise ValueError('approved training source bytes differ from manifest')
    registry=json.loads(provenance.read_text());folds=json.loads(split.read_text());stats=json.loads(norm.read_text())
    train=folds.get('train');approved=registry.get('approved_paired_episodes')
    if (registry.get('schema')!='lingbot_tactile_dataset_registry_v1' or
            folds.get('schema')!='lingbot_tactile_seed_split_plan_v1' or
            folds.get('source_machine_role')!='relay_single_rtx3090' or
            folds.get('status')!='approved_for_training' or
            not isinstance(train,list) or not train or len(train)!=len(set(train)) or
            not isinstance(approved,list) or not set(train).issubset(set(approved)) or
            any(x in set(folds.get('validation',[]))|set(folds.get('test',[])) for x in train) or
            stats.get('schema') not in ('lingbot_tactile_train_norm_v1','lingbot_tactile_contact_norm_v2') or
            stats.get('status')!='fit_on_train_split_only' or
            stats.get('training_approval') is not True or
            stats.get('source_trials')!=train or
            stats.get('split_candidate_sha256')!=sha256(split)):
        raise ValueError('train-only approved registry/split/normalization missing')
    _verify_train_expert_sources(repo,folds,train)
    # No pickle; the actual trained output projection must not be all zeros.
    with safe_open(str(weights),framework='np') as f:
        names=set(f.keys())
        required={'to_state.weight','temporal.weight_ih_l0','temporal.weight_hh_l0',
                  'train_median','train_scale','norm_mode','norm_clip_abs'}
        if not required.issubset(names):raise ValueError('incomplete tactile adapter state_dict')
        output=f.get_tensor('to_state.weight')
        ih=f.get_tensor('temporal.weight_ih_l0');hh=f.get_tensor('temporal.weight_hh_l0')
        median=f.get_tensor('train_median');scale=f.get_tensor('train_scale')
        mode=f.get_tensor('norm_mode');clip=f.get_tensor('norm_clip_abs')
        hidden=output.shape[1]-1 if output.ndim==2 else -1
        if (output.ndim!=2 or output.shape[0]<=0 or hidden<=0 or
                ih.shape!=(3*hidden,16) or hh.shape!=(3*hidden,hidden) or
                not np.isfinite(output).all() or not np.any(output!=0) or
                median.shape!=(4,3) or scale.shape!=(4,3) or
                not np.isfinite(median).all() or not np.isfinite(scale).all() or np.any(scale<=0)):
            raise ValueError('untrained/nonfinite adapter or invalid persisted train normalization')
        expected_median=np.asarray(stats.get('median'),dtype=np.float32)
        expected_scale=np.asarray(stats.get('scale_iqr_floor_0p1'),dtype=np.float32)
        expected_mode=int(stats.get('schema')=='lingbot_tactile_contact_norm_v2')
        if (expected_median.shape!=(4,3) or expected_scale.shape!=(4,3) or
                not np.isfinite(expected_median).all() or not np.isfinite(expected_scale).all() or
                np.any(expected_scale<=0) or not np.allclose(median,expected_median,rtol=1e-5,atol=1e-6) or
                not np.allclose(scale,expected_scale,rtol=1e-5,atol=1e-6) or
                mode.shape!=() or int(mode)!=expected_mode or clip.shape!=()):
            raise ValueError('adapter persisted buffers disagree with approved train-only normalization')
        if expected_mode and (not np.isfinite(clip).all() or not np.isclose(float(clip),float(stats.get('normalized_force_clip_abs',float('nan'))),rtol=1e-5)):
            raise ValueError('adapter clipping disagrees with approved contact normalization')
    return {'adapter_sha256':expected_sha,'trained_optimizer_steps':record['trained_optimizer_steps'],
            'approved_train_episode_count':len(train),'base_checkpoint_sha256':record['base_checkpoint_sha256']}
