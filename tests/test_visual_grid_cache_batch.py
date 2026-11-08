"""CPU reproduction using the exact frozen preview method and a fake visual encoder."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch

SOURCE=Path(__file__).resolve().parents[1]/'work/real_v2_meta_preview.py'


def method():
    tree=ast.parse(SOURCE.read_text())
    node=next(n for cls in tree.body if isinstance(cls,ast.ClassDef) for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='get_image_features')
    module=ast.Module(body=[node],type_ignores=[]);ast.fix_missing_locations(module)
    scope={'torch':torch};exec(compile(module,str(SOURCE),'exec'),scope)
    return scope['get_image_features']


class Visual:
    spatial_merge_size=2
    def preprcess_grid_thw(self,grid_thw):
        return torch.ones(1),torch.ones(1),torch.ones(1),[64]*len(grid_thw),256
    def __call__(self,pixels,**kwargs):
        x=torch.ones(len(pixels)*64,2)
        return x,[x.clone()]


class GridCache(unittest.TestCase):
    def make(self,enabled):
        return SimpleNamespace(config=SimpleNamespace(precompute_grid_thw=enabled),
            qwenvl=SimpleNamespace(visual=Visual()),pos_embeds=None,position_embeddings=None,
            cu_seqlens=None,visual_split_sizes=None,visual_max_seqlen=None)
    def test_existing_cache_reproduces_batch_two_failure(self):
        f=method();obj=self.make(True)
        f(obj,torch.zeros(3,256,1),torch.tensor([[1,16,16]]*3))
        with self.assertRaisesRegex(RuntimeError,'split_sizes'):
            f(obj,torch.zeros(6,256,1),torch.tensor([[1,16,16]]*6))
    def test_uncached_grid_supports_changes_in_batch_size(self):
        f=method();obj=self.make(False)
        for count in (3,6,12,3):
            images,deep=f(obj,torch.zeros(count,256,1),torch.tensor([[1,16,16]]*count))
            self.assertEqual(images.shape,(count,64,2));self.assertEqual(deep[0].shape,images.shape)
            self.assertIsNone(obj.visual_split_sizes)
