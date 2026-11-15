"""Shared editable scientific plots for completed, independently audited PINOs."""
from pathlib import Path
import json
import platform

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from scipy.stats import gaussian_kde

from docs.tasks.ceramic_fem_comparison_figures import style,save,log_axis,FONT,PDFFONTS
from scripts.local_trend import smooth_interval
from scripts.trajectory_trend import trajectory_interval

METHODS=('nonlocal','local_ad','fem')
LABELS=('Nonlocal','Local AD','FEM PINO')
COLORS=('#E64B35','#4DBBD5','#00A087')


def presentation(data):
    if 'comparison_kind' not in data:
        return METHODS,LABELS,COLORS,'three_method'
    if data['comparison_kind']=='spinodal_nonlocal_fem':
        from scripts.pino_comparison_data import SPINODAL_OMISSION
        if (data['example']!='confined_ch2d' or 'omitted_method' not in data
                or data['omitted_method']!=SPINODAL_OMISSION):
            raise ValueError('Only explicitly declared Spinodal AD omission is supported')
        return ('nonlocal','fem'),('Nonlocal','FEM PINO'),(COLORS[0],COLORS[2]),'nonlocal_fem'
    if data['comparison_kind']!='time_fusion':
        raise ValueError('Unknown comparison kind')
    return ('decoder_film','joint_query'),('Decoder FiLM','Joint (x, t)'),COLORS[:2],'time_fusion'


def validate(data):
    methods,_,_,suffix=presentation(data)
    if data['status']!='passed' or [a['method'] for a in data['arms']]!=list(methods):
        raise ValueError('All declared audited final PINO methods are required')
    if suffix=='time_fusion' and (len(data['test_ids'])!=64 or data['monitor_ids']!=data['test_ids']):
        raise ValueError('Time-fusion controls require the complete common64test cohort')
    if suffix=='nonlocal_fem' and (len(data['test_ids'])!=8 or data['monitor_ids']!=data['test_ids']
            or data['budget']!=50000 or data['steps']!=list(range(0,50001,500))):
        raise ValueError('Spinodal requires both final50k models and all8test cases at101records')
    fields=data['fields'];steps=np.asarray(data['steps'])
    if not fields or len(fields)>4 or len({f['key'] for f in fields})!=len(fields):raise ValueError('Invalid channels')
    if steps.ndim!=1 or steps[0]!=0 or steps[-1]!=data['budget'] or np.any(np.diff(steps)<=0):
        raise ValueError('Complete ordered actual monitoring steps required')
    for arm in data['arms']:
        for key in ('run','checkpoint','configuration','monitor'):
            if not Path(arm[key]).is_absolute() or not Path(arm[key]).exists():raise ValueError('Missing direct provenance: '+key)
        curves=np.asarray(arm['monitor_errors'],float);final=np.asarray(arm['final_errors'],float)
        if (curves.shape!=(len(steps),len(data['monitor_ids']),len(fields))
                or final.shape!=(len(data['test_ids']),len(fields))):raise ValueError('Metric/cohort shape mismatch')
        if not np.isfinite(curves).all() or not np.isfinite(final).all() or min(curves.min(),final.min())<=0:
            raise ValueError('Log figures require positive finite actual errors; no silent floors/dropping')
    kind=data['smoothing']['kind']
    if kind not in ('whole_trajectory','circular_residual'):raise ValueError('Declare the interval estimand')
    if kind=='whole_trajectory' and len(data['monitor_ids'])<2:raise ValueError('Insufficient independent monitored cases')
    if kind=='circular_residual' and len(data['monitor_ids'])!=1:
        raise ValueError('Use available casewise trajectories rather than pretending their mean is a single case')
    if len(data['test_ids'])<2:raise ValueError('A single reference case cannot define a violin distribution')


def panels(data,distribution):
    count=len(data['fields']);size=(85 if count==1 else 170,60)
    fig,axes=plt.subplots(1,count,figsize=np.asarray(size)/25.4,squeeze=False);axes=axes.ravel()
    fig.subplots_adjust(left=.18 if count==1 else .09,right=.98,bottom=.22,
        top=.84 if distribution else .75,wspace=.40 if count>=3 else .27)
    for ax,field in zip(axes,data['fields'],strict=True):
        log_axis(ax);ax.spines[['top','right']].set_visible(False);ax.set_title(field['label'],pad=5)
    axes[0].set_ylabel('Rel. L² error' if distribution else data['convergence_ylabel'],labelpad=5)
    return fig,axes,size


def render(data,output):
    validate(data);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    methods,labels,colors,suffix=presentation(data)
    wall_time='convergence_axis' in data
    if wall_time and (suffix!='time_fusion' or data['convergence_axis']['kind']!='cold_wall_time'):
        raise ValueError('Only explicitly qualified temporal-control wall clocks are supported')
    if wall_time:
        for arm in data['arms']:
            elapsed=np.asarray(arm['elapsed_seconds'],float)
            if (elapsed.shape!=(len(data['steps']),) or not np.isfinite(elapsed).all()
                    or elapsed[0]<=0 or np.any(np.diff(elapsed)<=0)
                    or elapsed[-1]!=arm['timing']['cold_process_seconds']):
                raise ValueError('Require actual ordered cold observations and complete-process endpoint')
    tick_labels=(tuple(label.replace(' ','\n') for label in labels)
                 if len(data['fields'])>=3 else labels)
    options=dict(data['smoothing']);kind=options.pop('kind');steps=np.asarray(data['steps'])
    with matplotlib.rc_context(style()):
        fig,axes,size=panels(data,False);trends=[]
        for channel,(ax,field) in enumerate(zip(axes,data['fields'],strict=True)):
            for arm,label,color in zip(data['arms'],labels,colors,strict=True):
                values=np.asarray(arm['monitor_errors'])[:,:,channel];raw=values.mean(axis=1)
                coordinates=np.asarray(arm['elapsed_seconds']) if wall_time else steps
                if kind=='whole_trajectory':fitted=trajectory_interval(coordinates,values,**options)
                else:
                    fitted=smooth_interval(steps,raw,**options)
                    fitted.update(raw_steps=steps,raw_mean=raw,raw_case_errors=values)
                fitted['raw_steps']=steps
                fitted['raw_abscissa']=coordinates
                line,=ax.plot(coordinates,raw,color=color,ls='-',lw=.55,alpha=.35,zorder=2)
                np.testing.assert_array_equal(line.get_xdata(),coordinates)
                np.testing.assert_array_equal(line.get_ydata(),raw)
                ax.fill_between(fitted['x'],fitted['lower'],fitted['upper'],color=color,alpha=.18,lw=0,zorder=1)
                ax.plot(fitted['x'],fitted['smooth'],color=color,ls='-',lw=1.05,label=label,zorder=3)
                path=output/f"{arm['method']}_{field['key']}_trend.npz"
                np.savez_compressed(path,**fitted)
                trends.append(dict(method=arm['method'],field=field['key'],data=str(path),
                    steps=steps.tolist(),abscissa=coordinates.tolist(),raw_mean=raw.tolist(),monitor_ids=data['monitor_ids']))
            if wall_time:
                ax.set_xlim(left=0)
                ax.xaxis.set_major_locator(MaxNLocator(nbins=5,min_n_ticks=3))
                ax.set_xlabel(data['convergence_axis']['label'],labelpad=4)
            else:
                ticks=np.linspace(0,data['budget'],5,dtype=int)
                ax.set_xticks(ticks,['0' if x==0 else f'{x/1000:g}k' for x in ticks]);ax.set_xlabel('Step',labelpad=4)
        fig.legend(*axes[0].get_legend_handles_labels(),ncol=len(methods),loc='upper center',
            bbox_to_anchor=(.55,.95),handlelength=2.5,columnspacing=1.3)
        convergence=save(fig,output/(data['example']+'_'+suffix+'_convergence'),size)
        convergence['trends']=trends
        fig,axes,size=panels(data,True);densities=[]
        for channel,(ax,field) in enumerate(zip(axes,data['fields'],strict=True)):
            for position,(arm,color) in enumerate(zip(data['arms'],colors,strict=True)):
                values=np.asarray(arm['final_errors'])[:,channel];logs=np.log10(values)
                if np.ptp(logs)==0:raise ValueError('Identical samples do not define a KDE; no fabricated jitter')
                kde=gaussian_kde(logs,bw_method='scott');bandwidth=float(np.sqrt(kde.covariance[0,0]))
                support=np.linspace(logs.min()-3*bandwidth,logs.max()+3*bandwidth,256)
                density=kde(support);width=.32*density/density.max()
                ax.fill_betweenx(10**support,position-width,position+width,color=color,
                    alpha=.85,edgecolor='#333333',linewidth=.6)
                quartiles=np.quantile(values,[.25,.5,.75])
                for q in quartiles:
                    w=.32*float(kde(np.log10(q))[0])/density.max()
                    ax.plot([position-w,position+w],[q,q],color='#333333',lw=.6,ls=(0,(2,1.5)))
                densities.append(dict(method=arm['method'],field=field['key'],count=len(values),
                    bandwidth_log10=bandwidth,scott_factor=float(kde.factor),quartiles=quartiles.tolist(),
                    support_log10=support.tolist(),density=density.tolist(),raw_values=values.tolist()))
            ax.set_xticks(range(len(methods)),tick_labels);ax.margins(x=.12)
        distribution=save(fig,output/(data['example']+'_'+suffix+'_distribution'),size)
        distribution['densities']=densities
    record=dict(schema='pino_'+suffix+'_figure_v1',status='passed',data=data,
        figures=dict(convergence=convergence,distribution=distribution),
        fonts=dict(regular=str(FONT),math='dejavusans;fallback=None',pdffonts=str(PDFFONTS)),
        software=dict(python=platform.python_version(),numpy=np.__version__,matplotlib=matplotlib.__version__),
        plotting_sources=[str(Path(__file__).resolve()),str(Path('scripts/trajectory_trend.py').resolve()),
            str(Path('scripts/local_trend.py').resolve()),str(Path('docs/tasks/ceramic_fem_comparison_figures.py').resolve())],
        training_performed=False,inference_performed=False,reference_solves=False,
        density='log10 Gaussian KDE;Scott bandwidth;3bandwidth tails;equal maximum width;quartiles;no additional samples')
    (output/'provenance.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    return record
