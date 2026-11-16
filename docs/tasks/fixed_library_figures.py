"""Final fixed-budget error versus library size; mean and one sampleSD only."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();journal=json.loads((a.bundle/'journal.json').read_text())
    if journal['status'] not in ('audited','complete'):
        raise ValueError('All eight completed audited runs are required')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import fitz
    from docs.tasks.ceramic_fem_comparison_figures import style, FONT
    a.output.mkdir(parents=True,exist_ok=False);values={}
    with matplotlib.rc_context(style()):
        fig,axes=plt.subplots(1,2,figsize=(170/25.4,62/25.4),layout='constrained')
        for ax,case,label,color in zip(axes,('electropolish','ceramic'),
                ('(a)','(b)'),('#E64B35','#4DBBD5'),strict=True):
            records=[json.loads((Path(run)/'fixed_library_audit.json').read_text()) for run in journal['runs'][case]]
            if any(r['status']!='passed' or r['steps']!=journal['budget'][case]
                   or r['sample_count']!=64 for r in records):
                raise ValueError('Passed final-budget64case audits are required')
            sizes=np.array([r['library_size'] for r in records]);errors=np.array([r['sample_errors'] for r in records])
            if list(sizes)!=[64,128,256,512] or errors.shape!=(4,64):raise ValueError('Incomplete fixed-size64case comparison')
            means=errors.mean(axis=1);sd=errors.std(axis=1,ddof=1)
            for r,mean,spread in zip(records,means,sd,strict=True):
                np.testing.assert_allclose([r['mean'],r['sample_std']],[mean,spread],rtol=1e-14)
            ax.plot(sizes,means,'o-',color=color,lw=1.25,ms=4,mec='white',mew=.7)
            ax.fill_between(sizes,means-sd,means+sd,color=color,alpha=.18,linewidth=0)
            ax.set_xscale('log',base=2);ax.set_xticks(sizes,[str(x) for x in sizes])
            ax.set_xlabel('Fixed initial-condition count' if case=='electropolish' else 'Fixed geometry count')
            ax.set_ylabel('Rel. L² error');ax.set_title(label,loc='left')
            values[case]=dict(sizes=sizes.tolist(),means=means.tolist(),sample_std=sd.tolist(),
                runs=journal['runs'][case],sample_errors=errors.tolist(),
                y_limits=list(ax.get_ylim()),automatic_y_limits=ax.get_autoscaley_on())
        base=a.output/'electropolish_ceramic_fixed_library_error'
        fig.savefig(base.with_suffix('.pdf'),metadata={'CreationDate':None,'ModDate':None})
        fig.savefig(base.with_suffix('.png'),dpi=300);plt.close(fig)
    with fitz.open(base.with_suffix('.pdf')) as doc:
        page=doc[0];spans=[s for b in page.get_text('dict')['blocks'] if b['type']==0 for line in b['lines'] for s in line['spans']]
        if any(not 4.99<=s['size']<=7.01 for s in spans):raise ValueError('Final font sizes must remain5–7pt')
        if any(not page.rect.contains(fitz.Rect(s['bbox'])) for s in spans):raise ValueError('Clipped labels')
        for font in page.get_fonts(full=True):
            if font[2] not in ('TrueType','Type0') or not doc.extract_font(font[0])[3]:raise ValueError('Unembedded font')
    result=dict(status='passed',journal=str(a.bundle/'journal.json'),data=values,font=str(FONT),
        pdf=str(base.with_suffix('.pdf')),png=str(base.with_suffix('.png')),
        uncertainty='One sample standard deviation over64test cases; not confidence interval or seed variability',
        metric='Equal-point full-space-time relativeL2 per channel, then arithmetic channel/sample mean',
        weights='Final fixed-budget model for every library size',visual_review='Pending human/agent inspection')
    (a.output/'provenance.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(result['png'],flush=True)


if __name__=='__main__':main()
