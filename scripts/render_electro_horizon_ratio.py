"""Audit the fixed-particle sweep and export a delta/herror curve with sampleSD."""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from docs.tasks.electro_horizon_ratio_study import fixed_particle_config
from scripts.render_electro_horizon_study import REPO, collect, read
from scripts.render_pino_derivative_comparisons import publication_style


def render(root):
    manifest, records = collect(root, fixed_particle_config)
    baseline = Path(manifest['reused_run'])
    fields = ('times', 'draw_ids', 'entry_draw_ids', 'core_ids', 'core_mask',
              'ic_ids', 'dirichlet_ids', 'side_ids')
    with np.load(baseline/'sampling.npz') as data:
        expected = {k:data[k] for k in fields}
    accepted = np.load(baseline/'online_accepted.npy')
    for arm in manifest['arms']:
        run = Path(arm['run'])
        config = read(run/'config.json')
        if (config['sample']['Nx'], config['sample']['Ny']) != (201, 101):
            raise ValueError('All arms must retain the declared201x101grid')
        with np.load(run/'sampling.npz') as data:
            for k in fields:
                np.testing.assert_array_equal(data[k], expected[k], err_msg=arm['name']+':'+k)
        np.testing.assert_array_equal(np.load(run/'online_accepted.npy'), accepted)
        np.testing.assert_allclose(arm['audit']['delta_over_h'], arm['ratio'], rtol=0, atol=1e-14)
    rows = sorted(records.values(), key=lambda r:r['ratio'])
    x = np.array([r['ratio'] for r in rows])
    mean = np.array([r['statistics']['channel_mean']['mean'] for r in rows])
    sd = np.array([r['statistics']['channel_mean']['std'] for r in rows])
    style = read(REPO/'artifacts/pino_derivative_comparisons.json')['plot']['distribution']['publication']
    rc, font = publication_style(style)
    out = root/'figures'; out.mkdir(exist_ok=True)
    pdf, png = (out/f'electropolish_fixed20301_horizon_ratio.{ext}' for ext in ('pdf','png'))
    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=(85/25.4, 65/25.4))
        fig.subplots_adjust(left=.18, right=.97, bottom=.19, top=.88)
        ax.plot(x, mean, '-o', color='#E64B35', lw=.8, markersize=2.5)
        ax.fill_between(x, mean-sd, mean+sd, color='#E64B35', alpha=.22, linewidth=0)
        ax.set_xticks(x); ax.set_xticklabels([f'{v:g}' for v in x])
        ax.set_xlabel(r'$\delta/h$'); ax.set_ylabel(style['label'])
        ax.set_title('20,301 particles')
        ax.spines[['top','right']].set_visible(False); ax.tick_params(length=2,pad=2)
        limits = list(ax.get_ylim())
        fig.savefig(pdf,dpi=300); fig.savefig(png,dpi=300); plt.close(fig)
    sys.path.insert(0,str(Path(manifest['pdf_audit_packages']).resolve(strict=True)))
    import fitz
    with fitz.open(pdf) as doc:
        page=doc[0]
        np.testing.assert_allclose([page.rect.width*25.4/72,page.rect.height*25.4/72],[85,65],atol=1e-4)
        spans=[s for b in page.get_text('dict')['blocks'] if 'lines' in b for l in b['lines'] for s in l['spans']]
        if not spans or any(not 4.999<=s['size']<=7.001 or not page.rect.contains(fitz.Rect(s['bbox'])) for s in spans):
            raise ValueError('PDF text must remain editable, in-page and5–7pt')
        fonts=page.get_fonts(full=True)
        if any(f[2]=='Type3' or not doc.extract_font(f[0])[3] for f in fonts):
            raise ValueError('PDF needs embedded TrueType fonts')
    receipt=dict(status='passed',manifest=str(root/'manifest.json'),records=records,
        cold_model_optimizer_conditions='exact match to reused baseline',
        fixed_grid_sampling='all10000time/core/IC/BC/draw identities and condition acceptances match',
        halo_shapes='intentionally differ; no claim of identicalFP32fusion/actual encoder-image rounding',
        norm_replay='384sample scores independently recomputed',
        metric=manifest['metric'],spread='sampleSD,ddof=1; not seed spread or confidence interval',
        ratios=x.tolist(),mean=mean.tolist(),std=sd.tolist(),ylim_auto=limits,
        figure_mm=[85,65],font=font,pdf_fonts=fonts,outputs=[str(pdf),str(png)])
    (root/'report.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    print('AUDIT_AND_PLOT_COMPLETE',pdf,png,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    render(parser.parse_args().root)
