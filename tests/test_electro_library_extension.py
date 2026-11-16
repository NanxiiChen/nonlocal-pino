"""Recovery owner must finish one trainer before starting the other."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from docs.tasks import electro_library_extension as owner


class SerialRecoveryTests(unittest.TestCase):
    def exercise(self, fail_first):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary)
            state = dict(status='prepared', gpu='test GPU', source=str(bundle),
                         execution='runtime_serial', **owner.execution_settings('runtime_serial'),
                         configs={'online': 'online.py', 'fixed32': 'fixed32.py'},
                         runs={'online': 'online-run', 'fixed32': 'fixed32-run'},
                         checkpoint_step=10000, stages=[])
            owner.write(bundle / 'journal.json', state)
            owner.write(bundle / 'preflight.json', dict(status='passed'))
            active, starts = [], []

            class Child:
                def __init__(child, args, **kwargs):
                    self.assertFalse(active, 'Serial recovery overlapped GPU processes')
                    self.assertEqual(kwargs['env']['XLA_FLAGS'], '--xla_gpu_enable_command_buffer=')
                    self.assertEqual(kwargs['env']['JAX_ENABLE_X64'], 'false')
                    starts.append(args)
                    active.append(child)
                    child.pid, child.code = len(starts), None

                def wait(child, timeout=None):
                    child.code = 1 if fail_first and child.pid == 1 else 0
                    active.remove(child)
                    return child.code

                def poll(child):
                    return child.code

            with (patch.object(owner, 'gpu', return_value='test GPU'),
                  patch.object(owner.subprocess, 'check_output', return_value=''),
                  patch.object(owner.subprocess, 'Popen', Child)):
                if fail_first:
                    with self.assertRaisesRegex(RuntimeError, 'train-online failed'):
                        owner.run(bundle)
                else:
                    owner.run(bundle)
            return owner.read(bundle / 'journal.json'), starts

    def test_serial_training_precedes_all_evaluation_and_audits(self):
        state, starts = self.exercise(False)
        self.assertEqual(state['status'], 'complete')
        self.assertEqual([stage['name'] for stage in state['stages']], [
            'train-online', 'train-fixed32', 'evaluate-fixed32-checkpoint',
            'evaluate-fixed32', 'evaluate-online-checkpoint', 'evaluate-online',
            'audit-fixed32', 'audit-online'])
        self.assertEqual(len(starts), 8)

    def test_failure_is_recorded_without_retry_or_second_trainer(self):
        state, starts = self.exercise(True)
        self.assertEqual(state['status'], 'failed')
        self.assertEqual(len(starts), 1)
        self.assertEqual(state['stages'][0]['exit_code'], 1)


if __name__ == '__main__':
    unittest.main()
