"""Explicit, independently checked import of a physics-only auxiliary-head fit."""
from pathlib import Path
import json
import numpy as np


FILENAMES={'model':'stress_initializer.eqx','report':'stress_initializer.json',
           'quadratic':'stress_initializer_quadratic.npz'}


def initializer_paths(spec):
    if (set(spec)!={'kind','model','report','quadratic'}
            or spec.kind!='physics_auxiliary_block'):
        raise ValueError('Stress initializer requires kind=physics_auxiliary_block and explicit model/report/quadratic')
    paths={key:Path(spec[key]) for key in FILENAMES}
    if any(not path.is_absolute() for path in paths.values()):
        raise ValueError('Stress initializer paths must be absolute')
    for path in paths.values():
        if not path.is_file():raise FileNotFoundError(path)
    return paths


def import_initializer(parent,template,c,paths):
    """Reconstruct rounded auxiliary rows; no other learned array may change."""
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    from examples.channel_nsch2d.network import extend_stress_outputs
    if set(paths)!=set(FILENAMES):raise ValueError('All three initializer payloads required')
    parent_run=Path(c.train.refinement.parent_run)
    saved=json.loads((parent_run/'config.json').read_text())
    report=json.loads(Path(paths['report']).read_text())
    count=round(c.physical.terminal_time/c.time_discretization.dt)
    if (report['status']!='physics_stress_head_equilibration_passed'
            or Path(report['parent_run'])!=parent_run or report['parent_config']!=saved
            or report['stress_formulation']!=c.stress_formulation.to_dict()
            or report['time_nodes']!=count or report['input_nodes']!=(c.sample.Nx+1)*(c.sample.Ny+1)
            or report['supervised_evolved_field_samples']!=0):
        raise ValueError('Stress initializer parent, numerical recipe or physics-only scope disagrees')
    scalars=np.asarray([report[k] for k in ('objective_before','objective_after_quadratic',
        'objective_after_actual_FP32','old_GPU_physical_replay_gap','projected_GPU_physical_replay_gap','pressure_feature_gap')])
    if (not np.isfinite(scalars).all() or np.any(scalars<0)
            or scalars[2]>=scalars[0] or np.max(scalars[3:])>2e-5
            or abs(scalars[1]-scalars[2])>1e-6+.001*abs(scalars[1])):
        raise ValueError('Stress initializer physical replay/reduction report failed')
    with np.load(paths['quadratic'],allow_pickle=False) as f:
        coef=f['coefficients'];nodes=f['time'];G=f['gram'];g=f['linear'];constant=float(f['constant'])
    width=parent.output.weight.shape[1]+1
    if (coef.shape!=(2,width) or G.shape!=(2*width,2*width) or g.shape!=(2*width,)
            or any(not np.isfinite(a).all() for a in (coef,nodes,G,g)) or not np.isfinite(constant)):
        raise ValueError('Stress initializer quadratic shapes/finite values disagree')
    expected_nodes=np.arange(1,count+1,dtype=c.train.compute_dtype)*np.asarray(c.time_discretization.dt,dtype=c.train.compute_dtype)
    np.testing.assert_array_equal(nodes,expected_nodes)
    vector=coef.ravel();value=constant+2*g@vector+vector@G@vector
    if abs(constant-scalars[0])>1e-12 or abs(value-scalars[1])>1e-8+1e-6*abs(scalars[1]):
        raise ValueError('Stress initializer stored quadratic/result disagrees')
    expected=extend_stress_outputs(parent,template);dtype=expected.output.weight.dtype
    d1=jnp.asarray(coef[0],dtype=dtype);d2=jnp.asarray(coef[1],dtype=dtype)
    weights=expected.output.weight.at[3].add(d1[:-1]).at[5].add(-d1[:-1]).at[4].set(d2[:-1])
    bias=expected.output.bias.at[3].add(d1[-1]).at[5].add(-d1[-1]).at[4].set(d2[-1])
    expected=eqx.tree_at(lambda m:(m.output.weight,m.output.bias),expected,(weights,bias))
    actual=eqx.tree_deserialise_leaves(paths['model'],template)
    for a,b in zip(jax.tree.leaves(expected),jax.tree.leaves(actual),strict=True):
        if eqx.is_array(a):
            if not np.isfinite(np.asarray(b)).all():raise ValueError('Nonfinite stress initializer model')
            np.testing.assert_array_equal(a,b)
    return actual
