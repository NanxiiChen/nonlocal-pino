"""Independent CPU checkpoint/transport audit and paired scientific plots."""
import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ml_collections import ConfigDict
import numpy as np
from scipy import ndimage, sparse

from examples.corrosion2d import residual_experiment as r
from examples.corrosion2d.residuals import _raw_fields_and_derivatives
from examples.corrosion2d.training_precision import cast_floating, audit_floating

OUT = Path(__file__).with_suffix('')
LOGS = Path('/root/autodl-tmp/tf-logs/corrosion2d')
RUNS = {
    'float32': LOGS/'20261025-transport-kks_raw_edges-chl1-s_bc-icmatched_graph-bcmatched_graph-phase-seed42',
    'float64': LOGS/'20261026-precision-float64-phase-seed42',
}


def rel(a, b):
    return float(np.linalg.norm(a-b)/np.linalg.norm(b))


def main():
    if jax.default_backend() != 'cpu':
        raise RuntimeError('Frozen audit must use CPU')
    jax.config.update('jax_default_matmul_precision', 'highest')
    first = ConfigDict(json.loads((RUNS['float32']/'config.json').read_text()))
    reference = r.load_reference(first)
    points = np.stack([reference['X'].ravel(), reference['Y'].ravel()], -1)
    zero, one, _ = r.boundary_values(first, points)
    free = ~(zero|one)
    row, col = (reference[k] for k in ('graph_row', 'graph_col'))
    weight = reference['graph_weight'].astype(np.float64)
    D = sparse.diags(np.bincount(row, weights=weight, minlength=len(points)))-sparse.csr_matrix((weight, (row,col)), shape=(len(points), len(points)))
    target_p, target_c = (reference[k].reshape(len(reference['time']), -1) for k in ('phi', 'c'))
    chn, _, _, _, dc, cle = r.nondim(first)
    target_q = target_c-dc*(3*target_p**2-2*target_p**3)-cle
    target_dq = (D@target_q.T).T
    results = {'scope': 'One paired seed42, final5000 checkpoints; independent CPU field/AD and SciPy graph evaluation',
               'reference': first.experiment.reference_path, 'runs': {}}
    histories, fields, initials = {}, {}, {}
    for name, run in RUNS.items():
        dtype = np.dtype(name)
        jax.config.update('jax_enable_x64', name == 'float64')
        config = ConfigDict(json.loads((run/'config.json').read_text()))
        assert r.reference_spec(config) == r.reference_spec(first)
        with jax.enable_x64(False):
            kn, _ = jax.random.split(jax.random.PRNGKey(42))
            template32 = r.build_model(config, kn)
        template = cast_floating(template32, dtype)
        net = eqx.tree_deserialise_leaves(run/'ckpt_005000.eqx', template)
        initial = eqx.tree_deserialise_leaves(run/'model_initial.eqx', template)
        initials[name] = [np.asarray(a, dtype=np.float64) for a in jax.tree.leaves(eqx.filter(initial, eqx.is_array))]
        audit = audit_floating(net, dtype)
        assert (run/'ckpt_005000.eqx').read_bytes() == (run/'model_final.eqx').read_bytes()
        x = jnp.asarray(points.astype(np.float32), dtype=dtype)
        times = jnp.asarray(reference['time'].astype(np.float32), dtype=dtype)

        @eqx.filter_jit
        def evaluate(model):
            raw, _, physical, deriv = _raw_fields_and_derivatives(model, x, times)
            return raw, physical, deriv

        raw, physical, deriv = (np.asarray(a, dtype=np.float64) for a in evaluate(net))
        p, c, ct = (v.T for v in (physical[...,0], physical[...,1], deriv[...,1]))
        # Raw values are (Nt,Nx,2); physical/JVP outputs are (Nx,Nt,2).
        q = .5*cle*(np.tanh(raw[...,1])-1)
        assert p.shape == c.shape == q.shape == target_p.shape
        dq = (D@q.T).T
        ch = ct+chn*dq
        fields[name] = (p, c, q, dq)
        for array in (p, c, q, dq, ch):
            assert np.isfinite(array).all()
        with np.load(run/'prediction.npz') as saved:
            for pred, key in ((p,'phi'),(c,'c')):
                np.testing.assert_allclose(pred.reshape(saved[key].shape), saved[key],
                                           rtol=0, atol=3e-6 if name=='float32' else 2e-11)
        summary = json.loads((run/'summary.json').read_text())
        metrics = {'parameter_dtype': audit,
                   'terminal_phi_relative_l2': rel(p[-1], target_p[-1]),
                   'terminal_c_relative_l2': rel(c[-1], target_c[-1]),
                   'terminal_q_relative_l2_free': rel(q[-1,free], target_q[-1,free]),
                   'terminal_Dq_relative_l2_free': rel(dq[-1,free], target_dq[-1,free]),
                   'terminal_CH_rms_free': float(np.sqrt(np.mean(ch[-1,free]**2))),
                   'pooled_phi_relative_l2': rel(p,target_p),
                   'pooled_c_relative_l2': rel(c,target_c),
                   'boundary_q_rmse_over_cle': {'pit':float(np.sqrt(np.mean((q[:,zero]+cle)**2))/cle),
                                                   'solid':float(np.sqrt(np.mean(q[:,one]**2))/cle)},
                   'liquid_components_by_time':[int(ndimage.label(v.reshape(reference['X'].shape)<.5)[1]) for v in p],
                   'per_time_phi_relative_l2':[rel(a,b) for a,b in zip(p,target_p,strict=True)],
                   'per_time_c_relative_l2':[rel(a,b) for a,b in zip(c,target_c,strict=True)],
                   'run':str(run), 'wall_seconds':summary['wall_seconds']}
        for key in ('phi','c'):
            np.testing.assert_allclose(metrics[f'terminal_{key}_relative_l2'],summary['final_metrics'][f'terminal_rel_l2_{key}'],atol=3e-6 if name=='float32' else 2e-11,rtol=0)
        # Separate different optimization trajectories from inference roundoff.
        other_dtype = np.dtype('float64' if name=='float32' else 'float32')
        with jax.enable_x64(other_dtype == np.dtype('float64')):
            frozen = cast_floating(net, other_dtype)
            x_other, t_other = cast_floating((x, times[-1]), other_dtype)

            @eqx.filter_jit
            def frozen_inference(model, coordinates, time):
                return jax.vmap(model, in_axes=(0, None))(coordinates, time)

            other_fields = np.asarray(frozen_inference(frozen, x_other, t_other), dtype=np.float64)
        metrics['frozen_parameters_other_inference_precision'] = {
            'dtype':str(other_dtype),
            'terminal_phi_relative_l2':rel(other_fields[:,0],target_p[-1]),
            'terminal_c_relative_l2':rel(other_fields[:,1],target_c[-1]),
            'max_abs_phi_change':float(np.max(np.abs(other_fields[:,0]-p[-1]))),
            'max_abs_c_change':float(np.max(np.abs(other_fields[:,1]-c[-1])))}
        history = [json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
        assert [v['step'] for v in history] == list(range(0,5001,200))
        histories[name] = history
        late = [v for v in history if 4000 <= v['step'] <= 5000]
        metrics['late_window_4000_5000'] = {
            key:{'median':float(np.median([v[f'terminal_rel_l2_{key}'] for v in late])),
                 'min':float(min(v[f'terminal_rel_l2_{key}'] for v in late)),
                 'max':float(max(v[f'terminal_rel_l2_{key}'] for v in late))}
            for key in ('phi','c','q','Dq_free')}
        results['runs'][name] = metrics
    for a,b in zip(initials['float32'], initials['float64'],strict=True):
        np.testing.assert_array_equal(a,b)
    results['paired_initial_parameters_exact'] = True
    results['raw_histories'] = histories
    results['status'] = 'passed'
    OUT.with_suffix('.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')

    fig, axes = plt.subplots(2,2,figsize=(11,7.2),layout='constrained')
    for ax, key, title in zip(axes.ravel(),('phi','c','q','Dq_free'),
                              (r'$\phi$',r'$c$',r'$q$ (all nodes)',r'$Dq$ (free nodes)'),strict=True):
        for name, history in histories.items():
            ax.semilogy([v['step'] for v in history],[v[f'terminal_rel_l2_{key}'] for v in history],label=name.upper(),marker='.',ms=4)
        ax.axvspan(4000,5000,color='grey',alpha=.1)
        ax.set(xlabel='Training updates',ylabel='Relative L2 at t*=1 (50 s)',title=title)
        ax.grid(alpha=.25);ax.legend()
    fig.suptitle('Corrosion: identical FP32 initial parameters/data, FP32 vs FP64 arithmetic (seed 42)')
    fig.savefig(OUT.with_name(OUT.name+'-convergence.png'),dpi=180)
    fig.savefig(OUT.with_name(OUT.name+'-convergence.svg'))
    plt.close(fig)
    xx, yy = reference['X'],reference['Y']
    fig, axes = plt.subplots(3,3,figsize=(11,6.6),layout='constrained')
    sources = [('Reference',(target_p,target_c,target_q)),
               ('FP32',fields['float32'][:3]),('FP64',fields['float64'][:3])]
    for i,(label, arrays) in enumerate(sources):
        for j,(key, array) in enumerate(zip(('phi','c','q'),arrays,strict=True)):
            ax=axes[i,j]
            im=ax.pcolormesh(xx,yy,array[-1].reshape(xx.shape),shading='auto',
                             vmin=-cle if key=='q' else 0,vmax=0 if key=='q' else 1,cmap='viridis')
            ax.contour(xx,yy,arrays[0][-1].reshape(xx.shape),levels=[.5],colors='white',linewidths=.6)
            ax.set_aspect('equal');ax.set_title(f'{label}: {key}, t*=1')
            if i==2: ax.set_xlabel('x*')
            if j==0: ax.set_ylabel('y*')
            fig.colorbar(im,ax=ax,shrink=.8)
    fig.savefig(OUT.with_name(OUT.name+'-fields.png'),dpi=180)
    plt.close(fig)
    print(json.dumps({k:{m:v for m,v in a.items() if m.startswith('terminal_') or m=='late_window_4000_5000'} for k,a in results['runs'].items()},indent=2),flush=True)


if __name__ == '__main__':
    main()
