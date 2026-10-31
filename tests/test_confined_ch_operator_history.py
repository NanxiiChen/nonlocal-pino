"""Global histories retain parent stages without duplicate import evaluations."""
import json,tempfile,unittest
from pathlib import Path
from examples.confined_ch2d.operator_history import training_records


class HistoryTests(unittest.TestCase):
    def test_metrics_loss_and_cycle(self):
        with tempfile.TemporaryDirectory() as directory:
            parent=Path(directory)/'parent';child=Path(directory)/'child';parent.mkdir();child.mkdir()
            (parent/'config.json').write_text(json.dumps(dict(train=dict(iterations=3))))
            (child/'config.json').write_text(json.dumps(dict(train=dict(iterations=8,continuation=dict(parent_run=str(parent))))))
            (parent/'summary.json').write_text(json.dumps(dict(final_metrics=dict(step=3),wall_seconds=2.)))
            def write(path,name,rows):path.joinpath(name).write_text(''.join(json.dumps(row)+'\n' for row in rows))
            write(parent,'metrics.jsonl',[dict(step=0,elapsed_seconds=.1),dict(step=3,elapsed_seconds=1.9)])
            write(child,'metrics.jsonl',[dict(step=3,elapsed_seconds=.2),dict(step=6,elapsed_seconds=1.),dict(step=8,elapsed_seconds=2.)])
            write(parent,'loss.jsonl',[dict(step=1,loss=1.),dict(step=3,loss=.8)])
            write(child,'loss.jsonl',[dict(step=4,loss=.7),dict(step=8,loss=.5)])
            rows=training_records(child,'metrics.jsonl')
            self.assertEqual([r['step'] for r in rows],[0,3,6,8]);self.assertEqual(rows[-1]['elapsed_seconds'],4.)
            self.assertEqual([r['step'] for r in training_records(child,'loss.jsonl')],[1,3,4,8])
            (parent/'config.json').write_text(json.dumps(dict(train=dict(continuation=dict(parent_run=str(child))))))
            with self.assertRaisesRegex(ValueError,'Cyclic'):training_records(child,'metrics.jsonl')


if __name__=='__main__':unittest.main()
