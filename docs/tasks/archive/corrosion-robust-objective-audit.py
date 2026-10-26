"""Full-grid CPU audit of predeclared corrosion objective controls.

Adapted from the archived seed-mechanism probe; all metrics remain physical.
"""
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
from scipy import ndimage
from scipy import sparse

from examples.corrosion2d import residual_experiment as r
from examples.corrosion2d.residuals import _raw_fields_and_derivatives, graph_action
from nonlocal_pinn.operators import build_D

OUT = Path(__file__).with_suffix('')
LOGS = Path('/root/autodl-tmp/tf-logs')
RUNS = {'corrosion2d': {
    **{f'bounded{seed}': LOGS/f'corrosion2d/20261025-transport-kks_raw_edges-chl1-s_bc-icmatched_graph-bcmatched_graph-bounded-seed{seed}' for seed in (42, 7)},
    'bounded2026': LOGS/'corrosion2d/20261025-independent-seed2026-default-strategy',
    **{f'original{seed}': LOGS/f'corrosion2d/20261025-transport-kks_raw_edges-chl1-s_bc-icmatched_graph-bcmatched_graph-phase-seed{seed}' for seed in (42, 7)},
    'original2026': LOGS/'corrosion2d/20261026-objective-phase-seed2026',
    **{f'frozen{seed}': LOGS/f'corrosion2d/20261026-objective-bounded_frozen-seed{seed}' for seed in (42, 7, 2026)}
}}


def rel(a, b):
    return float(np.linalg.norm(a-b)/np.linalg.norm(b))


def rms(a):
    return float(np.sqrt(np.mean(np.square(a))))


def square_fraction(error, mask):
    denominator = np.sum(error**2)
    if denominator <= 0:
        raise ValueError('Nonzero error required')
    return float(np.sum(error[mask]**2)/denominator)


def main():
    if jax.default_backend() != 'cpu':
        raise ValueError('CPU-only frozen audit')
    jax.config.update('jax_default_matmul_precision', 'highest')
    output = {'scope': 'Matched objective intervention; common full graph/times and final5000 checkpoints, no seed selection', 'cases': {}}
    for case, runs in RUNS.items():
        first = ConfigDict(json.loads((next(iter(runs.values()))/'config.json').read_text()))
        reference = r.load_reference(first)
        points = np.stack([reference['X'].ravel(), reference['Y'].ravel()], -1)
        n = len(points)
        x = jnp.asarray(points, dtype=jnp.float32)
        times = jnp.asarray(reference['time'], dtype=jnp.float32)
        zero, one, _ = r.boundary_values(first, points)
        free = ~(zero|one)
        op = build_D(x, first.nl.delta, method='sparse', kernel=first.nl.kernel)
        np.testing.assert_array_equal(np.asarray(op._row_idx), reference['graph_row'])
        np.testing.assert_array_equal(-np.asarray(op._vals), reference['graph_weight'])
        sub = op._row_idx, op._col_idx, op._vals, op._diag
        pools = op.build_subgraph_pool(first.sample.K_core, first.sample.K_pool)
        counts = np.zeros(n, dtype=int)
        train_weights = np.zeros(n, dtype=float)
        for ids, _, mask, _ in pools:
            selected = np.asarray(ids)[np.asarray(mask)]
            ids_free = selected[free[selected]]
            counts[ids_free] += 1
            train_weights[ids_free] += 1/(len(pools)*len(ids_free))
        np.testing.assert_allclose(train_weights.sum(), 1, atol=1e-12)
        never = free & (counts == 0)
        row, col, weight = (reference[k] for k in ('graph_row', 'graph_col', 'graph_weight'))
        D = sparse.diags(np.bincount(row, weights=weight, minlength=n))-sparse.csr_matrix((weight,(row,col)),shape=(n,n))
        template = r.build_model(first, jax.random.PRNGKey(0))
        chn, n1, n2, n3, dc, cle = r.nondim(first)
        target_p, target_c = (reference[k].reshape(len(times), n) for k in ('phi', 'c'))
        target_q = target_c-dc*(3*target_p**2-2*target_p**3)-cle
        target_dq = (D@target_q.T).T

        @eqx.filter_jit
        def evaluate(model):
            raw, raw_t, fields, deriv = _raw_fields_and_derivatives(model, x, times)
            p, c = fields[..., 0], fields[..., 1]
            h = -2*p**3+3*p**2
            q_physical = c-dc*h-cle
            diffusion = n3*graph_action(p, sub)
            reaction = n2*(4*p**3-6*p**2+2*p)
            drive = -n1*q_physical*dc*(-6*p**2+6*p)
            ac = deriv[..., 0]+diffusion+reaction+drive
            return raw, fields, deriv, ac, diffusion, reaction, drive

        def summarize(model, config, label):
            raw, fields, deriv, ac, diffusion, reaction, drive = (np.asarray(v) for v in evaluate(model))
            p, c, pt, ct = (v.astype(float).T for v in (fields[..., 0],fields[..., 1],deriv[..., 0],deriv[..., 1]))
            raw = raw.astype(float)
            q = .5*cle*(np.tanh(raw[..., 1])-1)
            dq = (D@q.T).T
            ch = ct+chn*dq
            # AC uses exactly the FP32 physical-difference formula of training;
            # independent CH uses stable raw-q and SciPy graph accumulation.
            ac = ac.astype(float).T
            w = (1+config.experiment.bounded_eta)/(4*p*(1-p)+config.experiment.bounded_eta)
            objective_ac = ac*w if config.experiment.formulation in ('bounded', 'bounded_frozen') else ac
            valid = np.broadcast_to(free, p.shape).copy()
            valid[0] = False  # t=0 belongs to IC, not the sampled PDE horizon
            weighted_ac = float(np.sum(np.mean(np.abs(objective_ac[1:]),axis=0)*train_weights)*config.loss.ac_scale)
            weighted_ch = float(np.sum(np.mean(np.abs(ch[1:]),axis=0)*train_weights)*config.loss.ch_scale)
            full_ac = float(np.mean(np.abs(objective_ac[valid]))*config.loss.ac_scale)
            full_ch = float(np.mean(np.abs(ch[valid]))*config.loss.ch_scale)
            error = p[-1]-target_p[-1]
            dq_error = dq[-1]-target_dq[-1]
            bulk = ((target_p < .05)|(target_p > .95)) & valid
            phi_jac = 2*p*(1-p)
            jac_cl = .5*cle*(1-np.tanh(raw[..., 1])**2)
            a = {
                'initial_phi_relative_l2': rel(p[0], target_p[0]),
                'initial_c_relative_l2': rel(c[0], target_c[0]),
                'initial_q_rmse_over_cle_free': rms((q[0]-target_q[0])[free])/cle,
                'initial_Dq_relative_l2_free': rel(dq[0, free], target_dq[0, free]),
                'terminal_phi_relative_l2': rel(p[-1], target_p[-1]),
                'terminal_c_relative_l2': rel(c[-1], target_c[-1]),
                'zero_bc_q_rmse_over_cle': rms(q[:, zero]/cle+1),
                'one_bc_q_rmse_over_cle': rms(q[:, one]/cle),
                'graph_bc_max_abs_physical_error': max(float(np.max(np.abs(np.stack([p[:,zero],c[:,zero]],-1)))),float(np.max(np.abs(np.stack([p[:,one],c[:,one]],-1)-1)))),
                'terminal_q_relative_l2_free': rel(q[-1, free], target_q[-1, free]),
                'terminal_Dq_relative_l2_free': rel(dq[-1, free], target_dq[-1, free]),
                'full_positive_time_scaled_AC_MAE': full_ac,
                'pool_distribution_scaled_AC_MAE': weighted_ac,
                'full_positive_time_CH_MAE': full_ch,
                'pool_distribution_CH_MAE': weighted_ch,
                'terminal_CH_rms_free': rms(ch[-1, free]),
                'never_core_phi_squared_error_fraction': square_fraction(error, never),
                'never_core_Dq_squared_error_fraction': square_fraction(dq_error, never),
                'terminal_phase_output_jacobian_median': float(np.median(phi_jac[-1,free])),
                'terminal_cL_output_jacobian_median': float(np.median(jac_cl[-1,free])),
                'terminal_cL_derivative_zero_fraction_fp32': float(np.mean(np.asarray(jnp.tanh(jnp.asarray(raw[-1,:,1],dtype=jnp.float32)))**2 == 1)),
                'phase_bulk_absolute_residual_fraction': float(np.sum(np.abs(objective_ac[bulk]))/np.sum(np.abs(objective_ac[valid]))),
                'AC_term_rms_positive_times_free': {'dt':rms(pt[valid]),'diffusion':rms(diffusion.T[valid]),'reaction':rms(reaction.T[valid]),'drive':rms(drive.T[valid])},
                'liquid_components_by_time': [int(ndimage.label(v.reshape(reference['X'].shape)<.5)[1]) for v in p] if case=='corrosion2d' else None,
                'per_time_phi_relative_l2': [rel(v,t) for v,t in zip(p,target_p)],
            }
            history=[json.loads(v) for v in (runs[label]/'loss_history.jsonl').read_text().splitlines()]
            a['last_logged_loss_weights'] = history[-1]
            return a

        case_data = {'coverage': {'IC_nodes': n if first.experiment.initial_sampling=='matched_graph' else first.sample.N_ic_global+first.sample.N_ic_local,
                    'zero_graph_BC_nodes':int(zero.sum()),'one_graph_BC_nodes':int(one.sum()),
                    'PDE_free_nodes':int(free.sum()),'never_PDE_core_nodes':int(never.sum()),
                    'core_inclusions_per_100_batches_free':{str(int(k)):int(v) for k,v in zip(*np.unique(counts[free],return_counts=True))},
                    'pool_used_repeatedly':5000//len(pools), 'PDE_times_represented_by_common_positive_snapshots':len(times)-1}, 'runs':{}}
        for label, run in runs.items():
            config = ConfigDict(json.loads((run/'config.json').read_text()))
            config_ref = r.reference_spec(config)
            assert config_ref == r.reference_spec(first)
            model=eqx.tree_deserialise_leaves(run/'ckpt_005000.eqx',template)
            data = summarize(model,config,label)
            expected=json.loads((run/'summary.json').read_text())['final_metrics']['terminal_rel_l2_phi']
            np.testing.assert_allclose(data['terminal_phi_relative_l2'],expected,atol=3e-6,rtol=0)
            expected_c=json.loads((run/'summary.json').read_text())['final_metrics']['terminal_rel_l2_c']
            np.testing.assert_allclose(data['terminal_c_relative_l2'],expected_c,atol=3e-6,rtol=0)
            assert (run/'model_final.eqx').read_bytes() == (run/'ckpt_005000.eqx').read_bytes()
            hist=[json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
            late=[v for v in hist if 4000 <= v['step'] <= 5000]
            assert len(late) == 6
            data['late_window_4000_5000'] = {
                field: {'median':float(np.median([v['terminal_rel_l2_'+field] for v in late])),
                        'min':float(min(v['terminal_rel_l2_'+field] for v in late)),
                        'max':float(max(v['terminal_rel_l2_'+field] for v in late))}
                for field in ('phi', 'c', 'q', 'Dq_free')}
            data['path']=str(run)
            case_data['runs'][label] = data
            print(case,label,json.dumps({k:data[k] for k in ['initial_phi_relative_l2','graph_bc_max_abs_physical_error','terminal_phi_relative_l2','full_positive_time_scaled_AC_MAE','pool_distribution_scaled_AC_MAE','full_positive_time_CH_MAE','never_core_phi_squared_error_fraction']}),flush=True)
        output['cases'][case]=case_data
    # Direct byte/structured comparisons; no file hashes.
    for seed in (42, 7, 2026):
        names = [f'bounded{seed}',f'original{seed}',f'frozen{seed}']
        paths = [RUNS['corrosion2d'][name] for name in names]
        assert len({(p/'model_initial.eqx').read_bytes() for p in paths}) == 1
        configs = [json.loads((p/'config.json').read_text()) for p in paths]
        for c, path in zip(configs, paths):
            # Historical runs predate CH norm continuation and always used L1.
            # Check their actual source instead of supplying a runtime default.
            if 'ch_rms_after' in c['experiment']:
                assert c['experiment']['ch_rms_after'] == 0
                del c['experiment']['ch_rms_after']
            else:
                old = (path/'source/examples/corrosion2d/residual_experiment.py').read_text()
                assert 'ch_rms_after' not in old
                assert 'make_transport_loss_vector(config,' in old
                assert c['experiment']['ch_norm'] == 'l1'
            if 'lr_tail_start' in c['experiment']:
                assert c['experiment']['lr_tail_start'] == 0
                assert c['experiment']['lr_tail_end'] == 1e-5
                del c['experiment']['lr_tail_start']; del c['experiment']['lr_tail_end']
            else:
                old = (path/'source/examples/corrosion2d/residual_experiment.py').read_text()
                assert 'make_tail_optimizer' not in old
                assert 'optimizer = make_optimizer(config)' in old
                assert 'lr_tail_end' not in c['experiment']
            del c['experiment']['formulation']; del c['log']['run_name']
        assert configs[0] == configs[1] == configs[2]
    output['matched_control_validation'] = 'Same-seed initial bytes/configs equal except formulation/run name and explicitly source-verified absent historical disabled CH/LR continuation fields'
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for label,data in output['cases']['corrosion2d']['runs'].items():
        axes[0].semilogy(np.arange(21)/20,data['per_time_phi_relative_l2'],label=label)
        axes[1].scatter(data['initial_phi_relative_l2'],data['terminal_phi_relative_l2'],label=label)
        axes[2].scatter(data['graph_bc_max_abs_physical_error'],data['terminal_phi_relative_l2'],label=label)
    axes[0].set(xlabel='Nondimensional time',ylabel='Phase relative L2',title='Final checkpoints: early fit vs late evolution')
    axes[1].set(xlabel='Initial phase relative L2',ylabel='Terminal phase relative L2',xscale='log',yscale='log',title='Accurate initial field does not ensure evolution')
    axes[2].set(xlabel='Max physical nodal BC error',ylabel='Terminal phase relative L2',xscale='log',yscale='log',title='Failed model can have smaller BC errors')
    for ax in axes:ax.grid(which='both',alpha=.2);ax.legend(fontsize=8)
    fig.savefig(str(OUT)+'.png',dpi=180,bbox_inches='tight')
    plt.close(fig)
    OUT.with_suffix('.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
