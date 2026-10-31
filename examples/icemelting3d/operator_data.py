"""Explicit schemas, online volume construction and strict held-out loading."""
import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from nonlocal_pinn.nn.conditional import ConditionalFieldOperator
from examples.icemelting3d.ellipsoids import image_points, sample_parameters, signed_distance, validate_parameters


def validate_config(c):
    a, s, o, i = c.arch, c.sample, c.operator, c.initial
    if a.type != 'conditional_field' or tuple(a.extent) != (1., 1., 1.):
        raise ValueError('Melting operator requires conditional_field on the unit cube')
    if len(a.image_shape) != 4 or a.image_shape[0] != 1 or len(a.patch_shape) != 3:
        raise ValueError('Melting initial volume must have shape (1,Nz,Ny,Nx), with3Dpatches')
    if min(a.patch_shape) < 1 or any(n < 2 or n % p for n,p in zip(a.image_shape[1:], a.patch_shape)):
        raise ValueError('Positive3Dpatch dimensions must divide initial volume dimensions')
    if not (s.Nx == s.Ny == s.Nz and s.Nx >= 4):
        raise ValueError('Melting reference requires a uniform cubic sampling grid')
    h = 1/(s.Nx-1)
    if c.nl.method != 'sparse' or c.nl.kernel != 'cubic' or not h < c.nl.delta < np.sqrt(2)*h:
        raise ValueError('Operator pilot preserves the sparse cubic six-axial-neighbor graph')
    numeric = [i.center_min,i.center_max,i.axis_min,i.axis_max,c.physical.M,c.physical.lam,
               c.physical.eps_star,c.nondim.L_c,c.nondim.T_c,c.train.lr_init,o.reference_dt,o.reference_tolerance]
    if not np.isfinite(numeric).all() or not (0 < i.axis_min <= i.axis_max < .5
            and 0 < i.center_min <= i.center_max < 1
            and i.center_min-i.axis_max > 0 and i.center_max+i.axis_max < 1):
        raise ValueError('Finite ellipsoid ranges must guarantee strict unit-cube containment')
    if i.kind != 'ellipsoid_sdf' or i.sdf_iterations < 32 or i.reference_sdf_iterations < 48:
        raise ValueError('Euclidean ellipsoid SDF requires at least32/48iterations for FP32/64')
    if min(numeric[4:]) <= 0 or c.phase.formulation not in ('ic_latent','ic_phase'):
        raise ValueError('Positive physical/reference settings and explicit hard-IC formulation required')
    if o.action not in ('prepare','train','memory') or min(o.condition_batch,o.monitor_conditions,o.test_conditions,o.probe_steps) < 1:
        raise ValueError('Explicit action prepare/train/memory and positive condition/probe counts required')
    if o.snapshots < 2 or o.evaluation_axis_points < 2 or o.evaluation_axis_points > s.Nx or o.evaluation_chunk < 1:
        raise ValueError('Invalid reference snapshot/evaluation sampling dimensions')
    interval = 1/(o.snapshots-1)
    if not np.isclose(round(interval/o.reference_dt)*o.reference_dt, interval, rtol=0, atol=1e-12):
        raise ValueError('Reference dt must divide each snapshot interval')
    if not 0 < s.K_core < s.Nx**3 or s.K_pool < 1 or s.Nt < 1 or not 0 < s.t_min < 1:
        raise ValueError('Invalid fixed-shape core/time/pool settings')
    if c.loss.use_causal or c.loss.iface_gamma_init != 0 or c.loss.iface_gamma_decay_to != 0:
        raise ValueError('Pilot explicitly uses one unweighted noncausal physical PDE objective')
    if c.train.optimizer not in ('adam','soap') or c.train.iterations < 1:
        raise ValueError('Operator optimizer must be adam/soap with positive update budget')
    if 'spatial' in s:
        from examples.icemelting3d.spatial_sampling import validate_spatial
        validate_spatial(c)


def physics(c):
    n1 = c.physical.M*c.nondim.T_c/c.nondim.L_c**2
    width = np.sqrt(2)*c.physical.eps_star
    return n1, n1/c.physical.eps_star**2, c.physical.lam*c.nondim.T_c/c.nondim.L_c/width, width


def identity(c):
    return json.loads(json.dumps(dict(physical=c.physical.to_dict(),nondim=c.nondim.to_dict(),
        nl=c.nl.to_dict(),grid=[c.sample.Nx,c.sample.Ny,c.sample.Nz],initial=c.initial.to_dict(),
        image_shape=list(c.arch.image_shape),monitor_conditions=c.operator.monitor_conditions,
        test_conditions=c.operator.test_conditions,dataset_seed=c.operator.dataset_seed,
        snapshots=c.operator.snapshots,evaluation_axis_points=c.operator.evaluation_axis_points)))


def build_network(c, key):
    a = c.arch
    net = ConditionalFieldOperator(image_shape=tuple(a.image_shape),patch_shape=tuple(a.patch_shape),
        extent=tuple(a.extent),width=a.width,encoder_depth=a.encoder_depth,heads=a.heads,
        mlp_ratio=a.mlp_ratio,decoder_depth=a.decoder_depth,spatial_features=a.spatial_features,
        spatial_scale=a.spatial_scale,time_features=a.time_features,time_scale=a.time_scale,
        local_dim=0,out_dim=1,conditioning=a.conditioning,norm_eps=a.norm_eps,key=key)
    return eqx.tree_at(lambda n:(n.output.weight,n.output.bias),net,
        (jnp.zeros_like(net.output.weight),jnp.zeros_like(net.output.bias)))


def make_sampler(c):
    input_points = jnp.asarray(image_points(tuple(c.arch.image_shape)),dtype=jnp.float32)
    width = physics(c)[3]
    @eqx.filter_jit
    def sample(key, points):
        parameters = sample_parameters(key,c.operator.condition_batch,c.initial)
        distance = jax.vmap(lambda p:signed_distance(input_points,p,c.initial.sdf_iterations))(parameters)
        volumes = jnp.tanh(distance/width).reshape((c.operator.condition_batch,)+tuple(c.arch.image_shape))
        initial = jax.vmap(lambda p:signed_distance(points,p,c.initial.sdf_iterations))(parameters)/width
        return volumes,initial,parameters
    return sample


def load_dataset(c):
    folder = Path(c.operator.dataset_path)
    manifest = json.loads((folder/'manifest.json').read_text())
    if manifest['identity'] != identity(c):
        raise ValueError('Held-out ellipsoid reference/config physical-grid-IC mismatch')
    refs = []
    for index,parameters in enumerate(manifest['parameters']):
        validate_parameters(parameters)
        with np.load(folder/f'condition_{index:03d}.npz') as stored:
            ref = {k:stored[k] for k in stored.files if k != 'metadata_json'}
            info = json.loads(str(stored['metadata_json']))
        if info['maximum_time_refinement_error'] >= c.operator.reference_tolerance:
            raise ValueError(f'Unconverged reference input{index}')
        if (ref['phi'].shape != (c.operator.snapshots,len(ref['points']))
                or ref['points'].ndim != 2 or ref['points'].shape[1] != 3
                or not np.isfinite(ref['points']).all()
                or np.any(ref['points'] < 0) or np.any(ref['points'] > 1)
                or not np.array_equal(ref['time'],np.linspace(0,1,c.operator.snapshots))
                or not np.array_equal(ref['parameters'],np.asarray(parameters))
                or not np.isfinite(ref['phi']).all() or ref['phi'].min() < -1.00001 or ref['phi'].max() > 1.00001):
            raise ValueError(f'Invalid held-out reference shapes/ranges/times input{index}')
        refs.append(ref)
    if len(refs) != c.operator.monitor_conditions+c.operator.test_conditions:
        raise ValueError('Held-out condition count mismatch')
    return manifest,refs
