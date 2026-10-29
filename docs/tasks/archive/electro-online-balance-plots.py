"""Scientific plots of recorded losses and frozen-checkpoint diagnostics."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out = Path('docs/tasks/archive')
r = json.loads((out/'electro-online-balance-diagnosis.json').read_text())
run = Path(r['run'])
log = [json.loads(l) for l in (run/'loss.jsonl').read_text().splitlines()]
steps = np.array([v['step'] for v in log])
losses = np.array([v['parts'] for v in log]); weights = np.array([v['weights'] for v in log])
names = r['labels']; colors = plt.cm.tab10(np.arange(5))
final = [g for g in r['gradients'] if g['checkpoint'] == 'final']
assert len(final) == 4 and len(r['fields']) == 4

fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), layout='constrained')
for i, name in enumerate(names):
    axes[0, 0].semilogy(steps, (losses*weights)[:, i], color=colors[i], label=name)
    axes[0, 1].semilogy(steps, weights[:, i], color=colors[i], label=name)
axes[0, 0].set(title='Recorded weighted losses', xlabel='Training update', ylabel='Weighted loss')
axes[0, 1].set(title='Recorded adaptive weights', xlabel='Training update', ylabel='Weight')
axes[0, 1].axhline(1000, color='gray', linestyle=':', label='Weight cap')
for ax in axes[0, :2]:
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=.2)

cos = np.mean([g['cosine'] for g in final], axis=0)
im = axes[0, 2].imshow(cos, vmin=-1, vmax=1, cmap='coolwarm')
for i in range(5):
    for j in range(5):
        axes[0, 2].text(j, i, f'{cos[i,j]:.2f}', ha='center', va='center', fontsize=9)
axes[0, 2].set(xticks=range(5), xticklabels=names, yticks=range(5), yticklabels=names,
               title='Final-model gradient cosine\nMean of 4 fixed online batches')
fig.colorbar(im, ax=axes[0, 2], shrink=.75)

for offset, scheme, label in [(-.19, 'saved_final_balance', 'Saved final weights'),
                              (.19, 'fresh_balance', 'Fresh balance (diagnostic)')]:
    fractions = np.array([g['scenarios'][scheme]['gradient_norm_fraction'] for g in final])
    axes[1, 0].bar(np.arange(5)+offset, 100*fractions.mean(0), .38, label=label)
axes[1, 0].set(xticks=range(5), xticklabels=names, ylabel='Share of sum of L2 norms (%)',
               title='Final-model weighted gradient norms')
axes[1, 0].legend(fontsize=8); axes[1, 0].grid(axis='y', alpha=.2)

f = next(v for v in r['fields'] if v['index'] == 16)
t = np.array([v['time'] for v in f['by_time']])
axes[1, 1].plot(t, [v['roughness_reference'] for v in f['by_time']], label='Reference')
axes[1, 1].plot(t, [v['roughness_prediction'] for v in f['by_time']], label='Prediction')
axes[1, 1].set(title='Unseen16: interface roughness', xlabel='Normalized time', ylabel='Std. of interface height')
axes[1, 1].legend(); axes[1, 1].grid(alpha=.2)
for key, label in [('c_l2', 'Total c'), ('q_l2', 'Liquid driving q'), ('Dq_l2', 'Transport Dq')]:
    axes[1, 2].semilogy(t, [100*v[key] for v in f['by_time']], label=label)
axes[1, 2].set(title='Unseen16: concentration / transport error', xlabel='Normalized time', ylabel='Relative L2 (%)')
axes[1, 2].legend(); axes[1, 2].grid(alpha=.2)
fig.suptitle('Online B8 / 5000 updates — diagnosis, no retraining')
fig.savefig(out/'electro-online-balance-summary.png', dpi=180); plt.close(fig)

with np.load(out/'electro-online-balance-fields.npz') as f:
    a = {k:f[k] for k in f.files}
ny, nx = a['X'].shape
ids = [int(np.argmin(abs(a['time']-v))) for v in (0, .05, .25, 1.)]
fig, axes = plt.subplots(3, 4, figsize=(12, 5.5), layout='constrained')
lo = min(a['q'][ids].min(), a['reference_q'][ids].min())
hi = max(a['q'][ids].max(), a['reference_q'][ids].max())
error = abs(a['q']-a['reference_q']); emax = error[ids].max()
for j, i in enumerate(ids):
    for row, values in enumerate((a['reference_q'][i], a['q'][i], error[i])):
        im = axes[row, j].imshow(values.reshape(ny, nx), origin='lower', extent=(0, 1, 0, .5),
            vmin=lo if row < 2 else 0, vmax=hi if row < 2 else emax,
            cmap='viridis' if row < 2 else 'magma')
        axes[row, j].set_xticks([]); axes[row, j].set_yticks([])
        if row == 0: axes[row, j].set_title(f't* = {a["time"][i]:.2f}')
        if j == 0: axes[row, j].set_ylabel(('Reference q', 'Prediction q', 'Absolute error')[row])
        if row == 1: imq = im
fig.colorbar(imq, ax=axes[:2, :], shrink=.65, label='q = c - (cSe-cLe) h(phi) - cLe')
fig.colorbar(im, ax=axes[2, :], shrink=.65, label='Absolute q error')
fig.suptitle('Unseen interface16 — concentration driving field')
fig.savefig(out/'electro-online-balance-q-fields.png', dpi=180); plt.close(fig)
print('plots complete')

early = json.loads((out/'electro-online-balance-early.json').read_text())
with np.load(run/'prediction.npz') as f:
    use = f['condition_indices'] >= 16
    pp, rr, times = f['prediction'][use, ..., 0], f['target'][use, ..., 0], f['time']
    gx, gy = f['X'][0], f['Y'][:, 0]


def heights(phase):
    crossing = (phase[:, :, :-1] > .5) != (phase[:, :, 1:] > .5)
    assert np.all(crossing.sum(2) == 1)
    k = crossing.argmax(2)
    a, b, d = np.indices(k.shape)
    low, high = phase[a, b, k, d], phase[a, b, k+1, d]
    return gy[k]+(.5-low)/(high-low)*(gy[k+1]-gy[k])


hp, hr = heights(pp), heights(rr)
fig, axes = plt.subplots(2, 4, figsize=(13, 5), layout='constrained')
for j, desired in enumerate((0., .05, .25, 1.)):
    k = int(np.argmin(abs(times-desired)))
    for i, index in enumerate((16,17,18,19)):
        axes[0, j].plot(gx, hr[i, k], color=colors[i], label=f'Interface{index}')
        axes[1, j].plot(gx, hp[i, k], color=colors[i])
    for ax in axes[:, j]:
        ax.set_ylim(.15, .48); ax.set_xlim(0,1); ax.grid(alpha=.2)
    axes[0, j].set_title(f't* = {times[k]:.2f}')
    axes[1, j].set_xlabel('x*')
axes[0, 0].set_ylabel('Reference interface height')
axes[1, 0].set_ylabel('Predicted interface height')
axes[0, 0].legend(fontsize=7)
fig.suptitle('Four different inputs — predicted interfaces become nearly identical')
fig.savefig(out/'electro-online-balance-condition-collapse.png', dpi=180); plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), layout='constrained')
sp = early['spread']; tt = [s['time'] for s in sp]
for field, name in [('phi', 'Phase'), ('c', 'Concentration')]:
    axes[0].semilogy(tt, [100*s[field+'_predicted_spread_over_reference'] for s in sp], label=name)
axes[0].axhline(100, color='gray', linestyle=':')
axes[0].set(xlabel='Normalized time', ylabel='Predicted / reference input spread (%)',
            title='Response to different initial interfaces')
values = early['fine']['by_time']
for name in ('AC', 'CH'):
    axes[1].loglog([s['time'] for s in values[1:]], [s[name] for s in values[1:]], label=name)
axes[1].axvline(.05, color='gray', linestyle=':', label='First 5% of time')
axes[1].set(xlabel='Normalized time', ylabel='Full free-grid mean absolute residual',
            title='Unseen16: near-initial-time residual peak')
for ax in axes: ax.legend(fontsize=8); ax.grid(alpha=.2)
fig.savefig(out/'electro-online-balance-early.png', dpi=180); plt.close(fig)
print('early/condition plots complete')

g = json.loads((out/'electro-online-balance-early-gradient.json').read_text())
cos = np.array(g['cosine'])
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), layout='constrained')
im = axes[0].imshow(cos, vmin=-1, vmax=1, cmap='coolwarm')
for i in range(5):
    for j in range(5):
        axes[0].text(j, i, f'{cos[i,j]:.2f}', ha='center', va='center', fontsize=9)
axes[0].set(xticks=range(5), xticklabels=names, yticks=range(5), yticklabels=names,
            title='Early-time parameter-gradient cosine')
fig.colorbar(im, ax=axes[0], shrink=.8)
for offset, key, label in [(-.19, 'saved_final', 'Saved final weights'), (.19, 'fresh', 'Fresh balance')]:
    axes[1].bar(np.arange(5)+offset, 100*np.array(g['scenarios'][key]['gradient_norm_fraction']), .38, label=label)
axes[1].set(xticks=range(5), xticklabels=names, ylabel='Share of sum of L2 norms (%)',
            title='Early-time weighted gradients')
axes[1].legend(fontsize=8); axes[1].grid(axis='y', alpha=.2)
fig.suptitle('Frozen final model / batch0 / t* = .001, .003, .005, .02')
fig.savefig(out/'electro-online-balance-early-gradient.png', dpi=180); plt.close(fig)
print('early gradient plot complete')
