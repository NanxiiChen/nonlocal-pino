"""The expanded queue respects its dependency and never overlaps GPU jobs."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from docs.tasks import electro_15k_suite as suite


class ExpandedQueueTests(unittest.TestCase):
    def exercise(self, dependency_status, fail_train):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / 'suite'
            dependency = Path(directory) / 'dependency'
            bundle.mkdir()
            dependency.mkdir()
            suite.write(dependency / 'journal.json', dict(status=dependency_status, runs={}))
            state = dict(status='prepared', dependency=str(dependency), source=str(bundle),
                         gpu='test GPU', xla_flags='--xla_gpu_enable_command_buffer=', stages=[],
                         runs={arm: arm + '-run' for arm in suite.ARMS},
                         configs={arm: arm + '.py' for arm in suite.ARMS}, training_order=list(suite.ARMS))
            suite.write(bundle / 'journal.json', state)
            suite.write(bundle / 'preflight.json', dict(status='passed'))
            active, started = [], []

            class Child:
                def __init__(child, args, **kwargs):
                    self.assertFalse(active)
                    self.assertEqual(kwargs['env']['XLA_FLAGS'], state['xla_flags'])
                    started.append(args)
                    child.pid, child.code = len(started), None
                    active.append(child)

                def wait(child, timeout=None):
                    child.code = 1 if fail_train and child.pid == 1 else 0
                    active.remove(child)
                    return child.code

                def poll(child):
                    return child.code

            with (patch.object(suite, 'gpu', return_value='test GPU'),
                  patch.object(suite.subprocess, 'check_output', return_value=''),
                  patch.object(suite.subprocess, 'Popen', Child)):
                if fail_train or dependency_status == 'failed':
                    with self.assertRaises(RuntimeError):
                        suite.run(bundle)
                else:
                    suite.run(bundle)
            return suite.read(bundle / 'journal.json'), started

    def test_dependency_failure_prevents_all_gpu_work(self):
        state, started = self.exercise('failed', False)
        self.assertEqual(state['status'], 'blocked_dependency')
        self.assertEqual(started, [])

    def test_only_five_serial_trainers_with_checkpoint_final_and_audit(self):
        state, started = self.exercise('complete', False)
        self.assertEqual(state['status'], 'complete')
        self.assertEqual(len(started), 20)
        self.assertEqual([s['name'] for s in state['stages'] if s['name'].startswith('train-')],
                         ['train-' + arm for arm in suite.ARMS])
        for index, arm in enumerate(suite.ARMS):
            self.assertEqual([s['name'] for s in state['stages'][4*index:4*index+4]],
                             ['train-' + arm, f'evaluate-{arm}-10000',
                              f'evaluate-{arm}-15000', 'audit-' + arm])

    def test_training_failure_does_not_retry_or_launch_next_arm(self):
        state, started = self.exercise('complete', True)
        self.assertEqual(state['status'], 'failed')
        self.assertEqual(len(started), 1)


if __name__ == '__main__':
    unittest.main()
