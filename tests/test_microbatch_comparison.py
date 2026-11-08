import json
from pathlib import Path
import unittest
import torch
from microbatch_comparison import compare,tensor_digest
LIMITS=json.loads((Path(__file__).resolve().parents[1]/'datasets/microbatch_numeric_screen_v1.json').read_text())['limits']

class Comparison(unittest.TestCase):
    def test_screen_detects_direction_scale_and_local_change(self):
        a={'large':torch.ones(100),'small':torch.ones(4)*.001}
        self.assertTrue(compare(a,a,1.,1.,LIMITS)['passed'])
        for b in ({k:-v for k,v in a.items()},{k:v*1.1 for k,v in a.items()},
                  dict(a,small=a['small']*2)):
            self.assertFalse(compare(a,b,1.,1.,LIMITS)['passed'])
        self.assertFalse(compare(a,a,1.,1.1,LIMITS)['passed'])

    def test_hashes_and_invalid_inputs(self):
        a={'x':torch.tensor([1.,2.])}
        self.assertEqual(tensor_digest(a),tensor_digest({'x':a['x'].clone()}))
        self.assertNotEqual(tensor_digest(a),tensor_digest({'x':a['x'].reshape(1,2)}))
        for b in ({},{'x':a['x'].double()},{'x':a['x'].reshape(1,2)}, {'x':torch.tensor([float('nan'),2.])}):
            with self.assertRaises(ValueError):compare(a,b,1.,1.,LIMITS)
        with self.assertRaises(ValueError):compare({'x':torch.zeros(2)},{'x':torch.zeros(2)},1.,1.,LIMITS)
