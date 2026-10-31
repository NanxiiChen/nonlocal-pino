"""Read complete global-step histories across explicit direct-path continuations."""
import json
from pathlib import Path


def training_records(run,filename):
    if filename not in ('metrics.jsonl','loss.jsonl'):raise ValueError('Explicit metrics/loss history filename required')
    def read(path,seen):
        identity=path.resolve()
        if identity in seen:raise ValueError('Cyclic training continuation history')
        seen=seen|{identity}
        c=json.loads((path/'config.json').read_text())
        rows=[json.loads(line) for line in (path/filename).read_text().splitlines()]
        if not rows or any(b['step']<=a['step'] for a,b in zip(rows,rows[1:])):
            raise ValueError('Nonempty ordered stage history required')
        if 'continuation' not in c['train']:return rows
        parent=Path(c['train']['continuation']['parent_run'])
        if not parent.is_absolute():raise ValueError('Absolute parent history path required')
        previous=read(parent,seen);summary=json.loads((parent/'summary.json').read_text())
        start=summary['final_metrics']['step']
        if filename=='metrics.jsonl':
            if rows[0]['step']!=start:raise ValueError('Missing imported initial-field assessment')
            offset=summary['cumulative_wall_seconds'] if 'cumulative_wall_seconds' in summary else summary['wall_seconds']
            rows=[dict(row,elapsed_seconds=offset+row['elapsed_seconds']) for row in rows[1:]]
        elif rows[0]['step']<=start:raise ValueError('Loss stage must start after parent endpoint')
        return previous+rows
    return read(Path(run),set())
