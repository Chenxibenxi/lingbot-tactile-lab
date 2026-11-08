"""Strict CPU collator for fixed-shape transformed benchmark inputs.

No padding, augmentation, target completion, device transfer or fold selection.
Call prepare_censored_flow separately with matching frozen teacher predictions.
"""
import torch

FIELDS = {'images', 'img_masks', 'state', 'lang_tokens', 'lang_masks',
          'actions', 'joint_mask', 'action_is_pad', 'image_grid_thw'}
TACTILE = {'tactile_history', 'tactile_mask', 'tactile_age_ms', 'tactile_valid'}


def collate(items):
    if not items:
        raise ValueError('nonempty batch required')
    for item in items:
        b = item['batch']
        if set(b) != FIELDS or set(item['tactile']) != TACTILE:
            raise ValueError('unexpected transformed fields')
        if item['meta']['fold'] != 'train':
            raise ValueError('throughput benchmark accepts train fold only')
        if b['actions'].shape != (1,50,55) or b['joint_mask'].shape != (1,50,55):
            raise ValueError('single-sample H50 inputs required')
        if b['images'].ndim != 4 or b['images'].shape[0] != 1:
            raise ValueError('expected images [1,cameras,patches,features]')
        n = b['images'].shape[1]
        if b['image_grid_thw'].shape != (n,3) or b['img_masks'].shape != (1,n):
            raise ValueError('camera/grid mismatch')
        v = item['valid']
        if v.shape != (1,50) or v.dtype != torch.bool or not bool(v.any()) or bool((v[:,1:] & ~v[:,:-1]).any()):
            raise ValueError('valid targets must be a nonempty contiguous prefix')
        if not torch.equal(b['action_is_pad'], ~v):
            raise ValueError('target validity disagrees with padding mask')
    def combine(values, grid=False):
        first = values[0]
        if any(not isinstance(t,torch.Tensor) or t.device.type != 'cpu' or t.shape != first.shape or t.dtype != first.dtype for t in values):
            raise ValueError('fixed-shape CPU tensors with identical dtype required')
        if not grid and (first.ndim < 1 or first.shape[0] != 1):
            raise ValueError('singleton batch dimension required')
        return torch.stack(values) if grid else torch.cat(values,dim=0)
    batch = {k:combine([x['batch'][k] for x in items],k=='image_grid_thw') for k in FIELDS}
    tactile = {k:combine([x['tactile'][k] for x in items]) for k in TACTILE}
    return dict(batch=batch,tactile=tactile,valid=combine([x['valid'] for x in items]),
                metadata=[dict(x['meta']) for x in items])
