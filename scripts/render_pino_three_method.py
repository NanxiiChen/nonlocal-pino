"""Render declared derivative controls; Spinodal may explicitly omit Local AD."""
import argparse
import json
from pathlib import Path
import sys

from scripts import pino_comparison_data as data
from scripts.pino_comparison_plot import render


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',choices=('ceramic','electropolish','melting','spinodal','channel'),required=True)
    parser.add_argument('--comparison',type=Path)
    parser.add_argument('--baseline-comparison',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.case!='ceramic' and args.comparison is None:parser.error('--comparison is required for this case')
    if args.case=='channel' and args.baseline_comparison is None:parser.error('Channel requires the retained nonlocal/FEM audit')
    if args.case!='channel' and args.baseline_comparison is not None:parser.error('--baseline-comparison is only valid for channel')
    if args.case=='ceramic':
        if args.comparison is not None:parser.error('Ceramic preserves the previously approved direct-path selection')
        records=data.ceramic()
    elif args.case=='electropolish':records=data.electro(args.comparison)
    elif args.case=='melting':records=data.melting(args.comparison)
    elif args.case=='spinodal':records=data.spinodal(args.comparison)
    else:records=data.channel(args.comparison,args.baseline_comparison)
    output=args.output.resolve();report=render(records,output)
    command=[str(Path(sys.executable).absolute()),'-m','scripts.render_pino_three_method','--case',args.case,'--output',str(output)]
    if args.comparison is not None:command+=['--comparison',str(args.comparison.resolve())]
    if args.baseline_comparison is not None:command+=['--baseline-comparison',str(args.baseline_comparison.resolve())]
    report['reproduction']=dict(command=command,cwd=str(Path(__file__).resolve().parents[1]),
        environment=dict(PYTHONPATH='/tmp/pino-pdf-audit',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'),
        loader=str(Path(data.__file__).resolve()),renderer=str(Path(__file__).resolve()))
    (output/'provenance.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status='passed',output=str(output),example=records['example'],
        channels=[f['key'] for f in records['fields']],monitor_cases=len(records['monitor_ids']),
        final_cases=len(records['test_ids']),
        figures={k:{name:v[name] for name in ('pdf','png','bytes')} for k,v in report['figures'].items()}),indent=2))


if __name__=='__main__':main()
