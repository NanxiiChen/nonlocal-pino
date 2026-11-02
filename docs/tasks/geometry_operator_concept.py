"""Generate the short conceptual version from already verified real inputs."""
import json
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parents[2]


def main():
    detailed = ROOT/'docs/tasks/geometry-operator-explainer.html'
    data = json.loads(re.search(r'<script id="geometry-data" type="application/json">(.*?)</script>',
        detailed.read_text(), re.S).group(1))
    compact = dict(cases=[dict(index=c['index'], holes=c['holes'],
        boundary=[p[:2] for p in c['boundary']], domain=[p[:2] for p in c['domain']]) for c in data['cases']])
    template = (ROOT/'docs/tasks/geometry-operator-concept.html.in').read_text()
    if template.count('__CONCEPT_DATA__') != 1:
        raise ValueError('Exactly one conceptual data slot required')
    html = template.replace('__CONCEPT_DATA__', json.dumps(compact, separators=(',', ':')))
    output = ROOT/'docs/tasks/geometry-operator-concept.html'; output.write_text(html)
    display = Path('/root/.codex/visualizations/2026/10/24/01a16d71-2d47-7201-9441-ab74e0523152/geometry-operator-concept.html')
    shutil.copyfile(output, display)
    print(str(output))


if __name__ == '__main__':
    main()
