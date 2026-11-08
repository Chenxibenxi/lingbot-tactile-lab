import unittest
import torch
from microbatch_collator import collate
from terminal_h50_contract import prepare_censored_flow


def sample(index, length):
    valid=torch.arange(50)[None,:]<length
    joints=torch.zeros(1,50,55,dtype=torch.bool);joints[:,:,:14]=True
    b=dict(images=torch.full((1,3,8,12),float(index)),img_masks=torch.ones(1,3,dtype=torch.bool),
           image_grid_thw=torch.tensor([[1,2,4]]*3),state=torch.zeros(1,55),
           lang_tokens=torch.ones(1,10,dtype=torch.long),lang_masks=torch.ones(1,10,dtype=torch.bool),
           actions=torch.full((1,50,55),float(index)),joint_mask=joints,action_is_pad=~valid)
    tactile=dict(tactile_history=torch.full((1,8,2),float(index)),tactile_mask=torch.ones(1,8,dtype=torch.bool),
                 tactile_age_ms=torch.tensor([1.]),tactile_valid=torch.tensor([True]))
    return dict(batch=b,tactile=tactile,valid=valid,meta=dict(fold='train',frame_index=index))


class Collator(unittest.TestCase):
    def test_order_grid_and_censored_tail(self):
        items=[sample(i,n) for i,n in enumerate((1,5,23,50))]
        result=collate(items);b=result['batch']
        self.assertEqual(b['image_grid_thw'].shape,(4,3,3))
        self.assertEqual(b['images'][:,0,0,0].tolist(),[0,1,2,3])
        self.assertEqual(result['tactile']['tactile_history'][:,0,0].tolist(),[0,1,2,3])
        teacher=torch.full_like(b['actions'],-9.)
        filled,mask=prepare_censored_flow(b['actions'],b['joint_mask'],result['valid'],teacher)
        self.assertEqual(int(mask.sum()),14*(1+5+23+50))
        self.assertTrue(torch.equal(filled[~mask],teacher[~mask]))
        b['images'].zero_();self.assertEqual(float(items[3]['batch']['images'][0,0,0,0]),3.)

    def test_reject_heldout_variable_shape_and_inconsistent_mask(self):
        for mode in ('heldout','shape','mask'):
            items=[sample(0,5),sample(1,50)]
            if mode=='heldout':items[1]['meta']['fold']='validation'
            elif mode=='shape':items[1]['batch']['lang_tokens']=torch.ones(1,11,dtype=torch.long)
            else:items[0]['batch']['action_is_pad'].zero_()
            with self.assertRaises(ValueError):collate(items)
