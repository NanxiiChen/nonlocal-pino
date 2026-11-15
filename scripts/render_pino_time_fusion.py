"""Render audited complete64case time-fusion histories and final distributions."""
import argparse
import json
from pathlib import Path
import sys

from scripts.pino_comparison_data import require_report
from scripts.pino_comparison_plot import render
from scripts.time_fusion_wall_clock import attach


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--x-axis',choices=('step','cold-wall-time'),required=True)
    parser.add_argument('--clock-evidence',type=Path)
    args=parser.parse_args()
    source,report=require_report(args.comparison)
    if report['data']['comparison_kind']!='time_fusion':
        raise ValueError('A completed time-fusion audit is required')
    data=report['data']
    if args.x_axis=='cold-wall-time':
        if args.clock_evidence is None:parser.error('--clock-evidence is required for cold wall time')
        data=attach(data,args.clock_evidence)
    elif args.clock_evidence is not None:
        parser.error('--clock-evidence only applies to cold wall time')
    output=args.output.resolve();result=render(data,output)
    result['reproduction']=dict(command=[sys.executable,'-m','scripts.render_pino_time_fusion',
        '--comparison',str(source),'--output',str(output),'--x-axis',args.x_axis],cwd=str(Path(__file__).resolve().parents[1]),
        environment=dict(PYTHONPATH='/tmp/pino-pdf-audit',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'),
        audit=str(source),renderer=str(Path(__file__).resolve()))
    if args.clock_evidence is not None:
        result['reproduction']['command']+=['--clock-evidence',str(args.clock_evidence.resolve())]
    (output/'provenance.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('TIME_FUSION_FIGURES '+str(output),flush=True)


if __name__=='__main__':main()
