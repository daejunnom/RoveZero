"""Pure paired parameter evidence checks; no initializer/NN execution."""
import copy
import hashlib
import unittest

from rz_pals_model.config import PROFILES
from rz_pals_model.controlled_initialization import compare_parameter_inventories, initialization_plan


def _parameter(name, bits=b'\x00\x00\x80\x3f'):
    return {"name": name, "shape": [1], "dtype": "fp32_le", "bytes": 4, "sha256": hashlib.sha256(bits).hexdigest()}


def _inventories():
    result = {profile: [_parameter('public_encoder.core.weight')] for profile in PROFILES}
    for profile in ('legacy_summary_v1', 'full_line_v2'):
        result[profile].append(_parameter('experts.proposer.policy_head.weight'))
    for profile in ('full_line_v2', 'full_line_interaction_v2'):
        result[profile].append(_parameter('public_encoder.line_encoder.weight'))
    for profile in ('interaction_head_v2', 'full_line_interaction_v2'):
        result[profile].append(_parameter('experts.proposer.interaction_policy_head.weight'))
    return result


class ControlledInitializationEvidenceTests(unittest.TestCase):
    def test_plan_distinguishes_four_profiles_and_keeps_registration_pending(self):
        plan = initialization_plan(23)
        self.assertEqual([value['profile'] for value in plan['profiles']], list(PROFILES))
        self.assertEqual([(value['full_line'], value['interaction_head']) for value in plan['profiles']],
                         [(False, False), (True, False), (False, True), (True, True)])
        self.assertTrue(plan['new_checkpoints_required'])
        self.assertFalse(plan['existing_diagnostic_relabel_allowed'])
        self.assertFalse(plan['arena_eligible'])
        self.assertFalse(plan['diagnostic_only'])
        self.assertEqual(plan['optimizer_steps'], 0)
        with self.assertRaises(ValueError): initialization_plan(True)

    def test_all_common_and_added_parameter_pairs_are_bit_compared(self):
        result = compare_parameter_inventories(_inventories())
        self.assertEqual(len(result['pairs']), 6)
        self.assertEqual(result['common_core_bytes'], 4)
        self.assertTrue(all(pair['equal_fp32_bits'] for pair in result['pairs']))
        self.assertEqual(len(result['common_core']), 1)

    def test_differing_common_added_bits_shape_or_duplicate_names_fail(self):
        for label in ('core', 'input', 'head', 'shape', 'bytes', 'duplicate', 'profile'):
            inventories = _inventories()
            selected = inventories['full_line_interaction_v2']
            if label == 'core': selected[0]['sha256'] = '22' * 32
            elif label == 'input': selected[1]['sha256'] = '22' * 32
            elif label == 'head': selected[2]['sha256'] = '22' * 32
            elif label == 'shape': selected[0]['shape'] = [2]; selected[0]['bytes'] = 8
            elif label == 'bytes': selected[0]['bytes'] = True
            elif label == 'duplicate': selected.append(copy.deepcopy(selected[0]))
            else: del inventories['legacy_summary_v1']
            with self.subTest(label=label), self.assertRaises(ValueError):
                compare_parameter_inventories(inventories)


if __name__ == '__main__':
    unittest.main()
