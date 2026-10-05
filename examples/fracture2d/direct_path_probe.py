"""Tight common-floor direct benchmark and convergence-gated path diagnostics."""
import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import ConfigDict

from examples.fracture2d import nonlocal_baseline as direct
from examples.fracture2d.coupling_diagnostics import atomic_json, metrics
from examples.fracture2d.diagnostic_continuation import _strict_parent
from examples.fracture2d.fixed_load_sweeps import _experiment_configs
from examples.fracture2d.fp64 import FP64A2Backend
from examples.fracture2d.legacy_coupled_loading import load_diagnostic
from examples.fracture2d.legacy_coupling_refine import readiness
from examples.fracture2d.losses import FractureResidual
from examples.fracture2d.stagger import _warning_advancement_gates
from examples.fracture2d.subproblem_matrix import damage_solver
from examples.fracture2d.train import _build_grid


def target_grid(start, end, step):
    if not np.all(np.isfinite([start, end, step])) or not 0 <= start < end or step <= 0:
        raise ValueError('invalid sequential load grid')
    count = int(round((end-start)/step))
    if count < 1 or abs(count*step-(end-start)) > 32*np.finfo(float).eps*max(abs(end),step):
        raise ValueError('step size must partition the declared load interval')
    return np.linspace(start, end, count+1, dtype=np.float64)[1:]


def contract_limits(tolerance):
    if not np.isfinite(tolerance) or not 0 < tolerance < 1:
        raise ValueError('invalid tight diagnostic tolerance')
    return ConfigDict(dict(at2_relative_inf=tolerance, damage_change_inf=tolerance,
        q_change_relative_inf=tolerance, H_change_relative_inf=tolerance,
        reaction_change_relative=tolerance))


def run(args):
    if args.baseline_sweeps < 3 or args.path_sweeps < 3:
        raise ValueError('budgets must allow three confirmations')
    limits = contract_limits(args.tolerance)
    if args.output.exists() or not args.output.parent.is_dir():
        raise ValueError('output requires a new root under an existing parent')
    if not any(device.platform == 'gpu' for device in jax.devices()):
        raise RuntimeError('direct path diagnostic requires GPU')
    source = ConfigDict(json.loads((args.source_root/'config.json').read_text()))
    pc, _ = _experiment_configs(source, args.source_root, args.output, 4,
                                tensorboard_root=args.output.parent/'events')
    model = _build_grid(pc)
    strict, sd = _strict_parent(pc, model)
    c, parent, pd = load_diagnostic(args.parent_state, args.parent_sha256, 60,
        model, source, strict, sd, require_ready=True)
    if hashlib.sha256(args.seed_fields.read_bytes()).hexdigest() != args.seed_sha256:
        raise ValueError('baseline numerical seed SHA differs')
    with np.load(args.seed_fields, allow_pickle=False) as seed:
        q0, d0 = (seed[name].copy() for name in ('q','d'))
    with np.load(c.data.reference, allow_pickle=False) as data:
        load0, load1, end = (float(data['loads'][i]) for i in (60,61,62))
        first_load = float(data['loads'][1])
    grids = [target_grid(load1,end,step) for step in args.steps]
    backend = FP64A2Backend(model.static,model.constraints,jax.devices()[0])
    residual = FractureResidual(model,backend)
    residual.validate_fields(q0,d0,np.asarray(parent[6]),np.asarray(parent[6]),np.asarray(load1))
    if (min(d0) < -c.stagger.damage_range_atol or max(d0)>1
            or max(np.maximum(np.asarray(parent[5])-d0,0))>c.stagger.healing_atol):
        raise ValueError('baseline numerical seed violates damage bounds/floor')
    solve_d = damage_solver(residual.operator,np.asarray(model.geometry.material_reference_volumes),
        float(model.Gc/model.l0),float(model.Gc*model.l0),tol=1e-11,atol=1e-13,maxiter=10000)
    points = jnp.asarray(model.geometry.coordinates)
    first_q = residual.lift(jnp.stack((jnp.zeros_like(points[:,1]),first_load*points[:,1]),axis=1).reshape(-1),first_load)
    first_force = backend._residual_from_factor(backend._edge_factor(jnp.zeros(model.n_material,jnp.float64)),first_q)[backend.free_dofs]
    first_scale = float(jnp.sqrt(jnp.mean(first_force**2)))
    args.output.mkdir()
    atomic_json(args.output/'policy.json',dict(schema='fracture2d-tight-direct-path-v1',
        parent_sha256=pd, strict_ancestor_sha256=sd, seed_path=str(args.seed_fields),
        seed_sha256=args.seed_sha256, baseline_load=load1, end_load=end,
        baseline_sweeps=args.baseline_sweeps, path_sweeps=args.path_sweeps,
        steps=args.steps, tolerance=args.tolerance, confirmations=3, mixing=.5,
        model_digest=model.model_digest, original_config=json.loads(c.to_json_best_effort()),
        diagnostic_only=True, reference_validation=False, strict_accepted=False))

    def attempt(folder, load, floor_d, floor_H, initial_q, initial_d, budget):
        folder.mkdir(parents=True)
        q, d = jnp.asarray(initial_q),jnp.asarray(initial_d)
        immutable_H = np.asarray(floor_H).copy()
        previous, consecutive, rows = None,0,[]
        eligible = None
        status = 'budget_exhausted'
        for sweep in range(1,budget+1):
            phase = 'mechanics_before'
            try:
                before = backend.solve_strict(np.asarray(d),load,1e-9,1e-10,10000,initial_guess=np.asarray(q))
                H_used = residual.evaluate(before.displacement,d,None,jnp.asarray(floor_H)).trial_history
                phase = 'damage'
                solved = solve_d(H_used,d)
                trial = d+.5*(solved-d)
                phase = 'mechanics_after'
                after = backend.solve_strict(np.asarray(trial),load,1e-9,1e-10,10000,
                    initial_guess=np.asarray(before.displacement))
                q = after.displacement
                fields = residual.evaluate(q,trial,None,jnp.asarray(floor_H))
            except RuntimeError as error:
                status = 'numerical_failure'
                atomic_json(folder/'failure.json',dict(sweep=sweep,phase=phase,reason=str(error)))
                break
            q_np,d_np,H_np = map(np.asarray,(q,trial,fields.trial_history))
            residual.validate_fields(q_np,d_np,H_np,immutable_H,np.asarray(load))
            ratio = np.linalg.norm(q_np[:2*model.n_material].reshape(-1,2),axis=1)/load
            row: dict = dict(sweep=sweep,load=load,reaction=float(fields.reaction),
                damage_inf=float(jnp.max(jnp.abs(fields.damage))),
                mechanics_inf=float(jnp.max(jnp.abs(fields.mechanics))),
                mechanics_loss=float(jnp.mean((fields.mechanics/(first_scale*load/first_load))**2)),
                damage_min=float(min(d_np)),damage_max=float(max(d_np)),
                damage_healing_max=float(max(np.maximum(np.asarray(floor_d)-d_np,0))),
                history_decrease_max=float(max(np.maximum(immutable_H-H_np,0))),
                drift_count=int(np.count_nonzero(ratio>direct.DRIFT_ENVELOPE)),
                max_displacement_ratio=float(max(ratio)),
                accepted=False,diagnostic_only=True)
            # Comparator entries are needed only by the shared gate factory;
            # they are removed on its warning path and never published as errors.
            gates = _warning_advancement_gates(c,{**row,'reaction_relative':0.,'d_rmse':0.})
            row['gates'] = {name:bool(value) for name,value in gates.items()}
            gc0,gc1 = model.Gc/model.l0,model.Gc*model.l0
            full: dict = dict(q=q_np,d=d_np,H=H_np,H_used=np.asarray(H_used),
                d_solved=np.asarray(solved),d_previous=np.asarray(d),
                at2_local=gc0*d_np,
                at2_nonlocal=np.asarray(gc1*residual.operator.deterministic_matvec(trial)),
                at2_history=2*H_np*(d_np-1),
                at2_fixed_relaxed=np.asarray(gc0*trial+gc1*residual.operator.deterministic_matvec(trial)+2*H_used*(trial-1)),
                at2_fixed_solved=np.asarray(gc0*solved+gc1*residual.operator.deterministic_matvec(solved)+2*H_used*(solved-1)),
                coordinates=np.asarray(model.geometry.coordinates[:model.n_material]))
            diag = metrics(row,full,previous)
            checks = readiness(diag,row['gates'],limits)
            consecutive = consecutive+1 if all(checks.values()) else 0
            row.update(coupling=diag,checks=checks,consecutive=consecutive,
                tight_coupled_ready=consecutive>=3,strict_accepted=False)
            np.savez_compressed(folder/f'fields_{sweep:04d}.npz',**full)
            rows.append(row)
            atomic_json(folder/'sweeps.json',rows)
            print(json.dumps(dict(event='direct_path_sweep',case=str(folder.relative_to(args.output)),
                load=load,sweep=sweep,reaction=row['reaction'],at2_relative=diag['at2_relative_inf'],
                drift=row['drift_count'],consecutive=consecutive)),flush=True)
            previous = full,row
            d = trial
            if not all(gates.values()):
                status = 'physical_hard_stop'
                break
            if consecutive>=3:
                status = 'tight_coupled_ready'
                eligible = (q,d,fields.trial_history)
                break
        if not np.array_equal(immutable_H,np.asarray(floor_H)):
            raise RuntimeError('committed floor mutated within attempt')
        result = dict(status=status,completed_sweeps=len(rows),load=load,
            final=None if not rows else rows[-1],diagnostic_only=True,strict_accepted=False)
        atomic_json(folder/'result.json',result)
        return result, eligible

    base_result,baseline = attempt(args.output/'baseline_0061',load1,parent[5],parent[6],q0,d0,args.baseline_sweeps)
    summary: dict = dict(baseline=base_result,paths=[],diagnostic_only=True,strict_accepted=False)
    atomic_json(args.output/'result.json',summary)
    if baseline is None:
        return summary
    for step,targets in zip(args.steps,grids):
        name = f'path_du_{step:.8f}'
        committed = baseline
        committed_load = load1
        path_result: dict = dict(step=step,attempts=[],committed_loads=[load1])
        for ordinal,target in enumerate(targets,1):
            q_guess = residual.lift(committed[0]*(target/committed_load),target)
            result,candidate = attempt(args.output/name/f'step_{ordinal:04d}',float(target),
                committed[1],committed[2],q_guess,committed[1],args.path_sweeps)
            path_result['attempts'].append(result)
            if candidate is None:
                break
            committed,committed_load = candidate,float(target)
            path_result['committed_loads'].append(committed_load)
        path_result['last_committed_load'] = committed_load
        path_result['reached_endpoint'] = committed_load==end
        summary['paths'].append(path_result)
        atomic_json(args.output/'result.json',summary)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source-root','parent-state','seed-fields','output'):
        parser.add_argument(f'--{name}',type=Path,required=True)
    parser.add_argument('--parent-sha256',required=True)
    parser.add_argument('--seed-sha256',required=True)
    parser.add_argument('--baseline-sweeps',type=int,required=True)
    parser.add_argument('--path-sweeps',type=int,required=True)
    parser.add_argument('--tolerance',type=float,required=True)
    parser.add_argument('--steps',type=float,nargs='+',required=True)
    run(parser.parse_args())


if __name__=='__main__':
    main()
