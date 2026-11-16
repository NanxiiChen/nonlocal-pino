"""Reject accidental cohort/budget substitution in the15k publication."""
import copy
import unittest

from scripts.electro_15k_results import stage
from scripts.publish_pino_three_method import validate_budget_extension


class PublicationGuards(unittest.TestCase):
    def pair(self):
        previous = dict(comparison_kind='time_fusion', example='electropolish2d', budget=10000,
                        steps=list(range(0,10001,250)), monitor_ids=list(range(64)), test_ids=list(range(64)),
                        fields=[dict(key='phi',label='Phase'),dict(key='c',label='Concentration')], seed=42,
                        reference=[f'/references/condition_{i:03d}/local101.npz' for i in range(64)],
                        arms=[dict(method='decoder_film'),dict(method='joint_query')])
        current = copy.deepcopy(previous)
        current.update(budget=15000,steps=list(range(0,15001,250)),reference='/references')
        return current,previous

    def test_explicit_extension_retains_cohort_reference_and_methods(self):
        current,previous = self.pair()
        validate_budget_extension(current,previous)
        for key,value in [('budget',20000),('test_ids',list(range(8))),('reference','/other'),
                          ('arms',[dict(method='decoder_film')]),('example','icemelting3d')]:
            changed = copy.deepcopy(current)
            changed[key] = value
            with self.subTest(key=key),self.assertRaises(ValueError):
                validate_budget_extension(changed,previous)

    def test_cold_stage_must_be_complete_and_unique(self):
        item = dict(name='train-online',status='passed',exit_code=0,seconds=1200.)
        self.assertEqual(stage(dict(stages=[item]),'train-online'),item)
        for rows in [[],[item,item],[dict(item,status='running')],[dict(item,exit_code=1)]]:
            with self.subTest(rows=rows),self.assertRaises(ValueError):
                stage(dict(stages=rows),'train-online')


if __name__ == '__main__':
    unittest.main()
