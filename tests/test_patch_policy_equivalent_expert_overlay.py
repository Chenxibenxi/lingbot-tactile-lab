"""CPU-only isolated overlay preview; never invoke real planner/take_action."""
import ast
import unittest
from patch_policy_equivalent_expert_overlay import SOURCE,ORIGINAL_SHA,ANCHOR,generate
from make_expert_overlay import patched_source as current_expert
from validate_expert_data import sha256


class ExpertControllerOverlayPreviewTests(unittest.TestCase):
    def test_actual_take_action_only_has_opt_in_policy_motion_probe(self):
        before=sha256(SOURCE);self.assertEqual(before,ORIGINAL_SHA)
        original=SOURCE.read_text()
        preview=generate(original)
        self.assertEqual(sha256(SOURCE),before)
        self.assertEqual(preview.count('begin_motion("policy_action")'),1)
        self.assertEqual(current_expert(original).count('begin_motion("policy_action")'),0)
        body=ast.parse(preview)
        cls=next(n for n in body.body if isinstance(n,ast.ClassDef) and n.name=='Base_Task')
        action=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='take_action')
        snippet=ast.get_source_segment(preview,action)
        self.assertIn('TACTILE_POLICY_EQUIVALENT_EXPERT',snippet)
        self.assertLess(snippet.index('begin_motion("policy_action")'),snippet.index('eval_video_freq'))
        # Existing rule-expert move does not call this opt-in high-level hook.
        move=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='move')
        self.assertNotIn('begin_motion("policy_action")',ast.get_source_segment(preview,move))

    def test_changed_take_action_anchor_cannot_be_silently_patched(self):
        changed=SOURCE.read_text().replace(ANCHOR,ANCHOR.replace('eval_video_freq = 1','eval_video_freq = 2'))
        with self.assertRaisesRegex(ValueError,'SHA/content changed'):
            generate(changed)
        silent_source_change=SOURCE.read_text().replace('self.step_lim = None','self.step_lim = 2')
        with self.assertRaisesRegex(ValueError,'SHA/content changed'):
            generate(silent_source_change)


if __name__=='__main__':unittest.main()
