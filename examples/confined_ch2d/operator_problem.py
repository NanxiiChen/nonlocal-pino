"""Explicit geometry prior, fixed ambient initial field and boundary measures."""
import json
from pathlib import Path
import numpy as np
from ml_collections import ConfigDict
from examples.confined_ch2d.operator_conditioning import initial_rhs_enabled


def fixed_initial(points, c, *, xp):
    waves = xp.asarray(c.initial.waves, dtype=points.dtype)
    z = (points-xp.asarray(c.geometry.center, dtype=points.dtype))/xp.asarray(c.geometry.axes, dtype=points.dtype)
    angles = xp.pi*xp.sum(z[..., None, :]*waves[:, :2], axis=-1)+waves[:, 3]
    return c.initial.amplitude*xp.sum(waves[:, 2]*xp.cos(angles), axis=-1)


def fixed_initial_laplacian(points, c):
    waves = np.asarray(c.initial.waves)
    k = np.pi*waves[:, :2]/np.asarray(c.geometry.axes)
    angles = np.sum((points-np.asarray(c.geometry.center))[..., None, :]*k, axis=-1)+waves[:, 3]
    return -c.initial.amplitude*np.sum(waves[:, 2]*np.sum(k*k, axis=-1)*np.cos(angles), axis=-1)


def validate(c):
    if 'online' in c.operator:
        from examples.confined_ch2d.operator_online import validate_online
        validate_online(c)
    if 'initial_rhs_condition' in c.operator and not isinstance(c.operator.initial_rhs_condition,bool):
        raise ValueError('initial_rhs_condition must be an explicit boolean')
    rhs_condition=initial_rhs_enabled(c)
    if rhs_condition and (c.operator.action!='train' or 'augmentation' in c.operator):
        raise ValueError('Initial-RHS conditioning currently requires unaugmented train mode')
    potential = 'formulation' in c.operator
    if potential:
        if (c.operator.formulation != 'conservative_potential' or c.operator.metric != 'identity'
                or c.operator.action != 'train' or 'continuation' in c.train
                or 'augmentation' in c.operator or 'initial_rhs_condition' in c.operator):
            raise ValueError('Conservative potential requires cold unaugmented identity-metric training')
    if c.operator.action not in ('prepare', 'train') or c.operator.metric not in ('biharmonic_resolvent','linear_stable_resolvent','identity'):
        raise ValueError('Explicit prepare/train action and supported residual metric required')
    if c.operator.metric == 'identity' and not potential:
        raise ValueError('Identity metric is explicit only for the conservative-potential experiment')
    if 'spatial' in c.sample:
        if not potential:
            raise ValueError('Compact spatial sampling requires the conservative-potential coordinate')
        from examples.confined_ch2d.operator_core_halo import validate_spatial
        validate_spatial(c)
    if c.operator.metric=='linear_stable_resolvent':
        if c.operator.action!='train':raise ValueError('Prepare the unchanged biharmonic bank before selecting the derived training metric')
        workers=c.operator.metric_workers
        if isinstance(workers,bool) or not isinstance(workers,int) or workers<1:
            raise ValueError('Explicit positive integer metric_workers required')
    if c.train.compute_dtype not in ('float32','float64'):
        raise ValueError('Geometry operator train.compute_dtype must be float32 or float64')
    if c.nl.method != 'nonlocal' or c.nl.kernel != 'quadratic_visible':
        raise ValueError('Geometry pilot requires the visible regional nonlocal operator')
    if c.initial.kind != 'fixed_ambient':
        raise ValueError('Geometry pilot requires the same fixed ambient initial function')
    center,axes=np.asarray(c.geometry.center),np.asarray(c.geometry.axes)
    if center.shape!=(2,) or axes.shape!=(2,) or not np.isfinite(center).all() or not np.isfinite(axes).all() or np.any(axes<=0):
        raise ValueError('Require finite two-dimensional ellipse center and positive axes')
    counts=(c.operator.train_conditions,c.operator.test_conditions,c.operator.condition_batch,
        c.operator.maximum_sampling_attempts,c.geometry.boundary_outer,c.geometry.boundary_per_hole,
        c.sample.Nt,c.train.iterations,c.log.eval_every,c.log.checkpoint_every,c.reference.snapshots)
    if any(isinstance(n,bool) or not isinstance(n,int) or n<1 for n in counts) or c.reference.snapshots<2:
        raise ValueError('Geometry/reference/training counts must be positive integers; snapshots>=2')
    scales=(c.initial.amplitude,c.physical.terminal_time,c.physical.mobility,c.physical.epsilon,
        c.physical.beta,c.nl.delta,c.geometry.mesh_size,c.reference.dt,c.reference.refined_dt,
        c.reference.phase_tolerance,c.reference.chemical_tolerance,c.operator.clearance,
        c.train.lr_init,c.train.lr_min,c.train.lr_decay_rate,c.train.grad_clip)
    if not np.isfinite(scales).all() or min(scales)<=0:
        raise ValueError('Physical/reference/learning scales must be positive finite')
    if c.reference.refined_dt>=c.reference.dt or c.train.lr_min>c.train.lr_init or c.train.lr_decay_rate>1:
        raise ValueError('Require a smaller refined dt and a nonincreasing learning-rate schedule')
    if 'initial_residual_fraction' in c.loss:
        fraction=c.loss.initial_residual_fraction
        if isinstance(fraction,bool) or not np.isfinite(fraction) or not 0<=fraction<1:
            raise ValueError('initial_residual_fraction must be finite in [0,1), retaining positive-time PDE information')
    if 'phase' in c:
        if c.phase.lifting!='logarithmic_additive' or isinstance(c.phase.initial_time_scale,bool) or not np.isfinite(c.phase.initial_time_scale) or c.phase.initial_time_scale<=0:
            raise ValueError('Operator optional phase requires logarithmic_additive and positive finite initial_time_scale')
    if 'model' in c.arch:
        expected=(5,4) if rhs_condition else (5,3)
        if c.arch.model!='heterogeneous' or tuple(c.arch.input_dims)!=expected:
            raise ValueError(f'Heterogeneous CH requires explicit boundary/domain input_dims={expected}')
    elif rhs_condition:
        raise ValueError('Initial-RHS conditioning requires the heterogeneous function encoder')
    elif c.arch.input_dim!=5:
        raise ValueError('Boundary condition features require input_dim=5')
    waves = np.asarray(c.initial.waves)
    if waves.ndim != 2 or waves.shape[1] != 4 or not np.isfinite(waves).all():
        raise ValueError('Initial waves must be finite (kx,ky,amplitude,phase) rows')
    if len(waves)==0 or np.all(waves[:,2]==0) or np.any(np.sum(waves[:,:2]**2,axis=1)==0) or c.initial.amplitude*np.sum(abs(waves[:, 2])) >= 1:
        raise ValueError('Require a nonzero initial field strictly inside the pure-phase limits')
    if 'temporal' in c.sample:
        s=c.sample.temporal
        if s.kind=='fixed_nodes':
            nodes=np.asarray(s.nodes)
            if nodes.shape!=(c.sample.Nt,) or not np.isfinite(nodes).all() or np.any(np.diff(nodes)<=0) or nodes[0]<=0 or nodes[-1]!=c.physical.terminal_time:
                raise ValueError('Fixed global nodes must be positive ordered(Nt,) and end at T')
        elif s.kind=='log_uniform':
            if not isinstance(s.curriculum,bool):raise ValueError('Explicit boolean temporal curriculum required')
            if not np.isfinite([s.time_scale,s.initial_horizon]).all() or s.time_scale<=0 or not 0<s.initial_horizon<=c.physical.terminal_time:
                raise ValueError('Temporal scale positive and initial_horizon in (0,T] required')
            if isinstance(s.warmup_iterations,bool) or not isinstance(s.warmup_iterations,int) or not 2<=s.warmup_iterations<=c.train.iterations:
                raise ValueError('Temporal warmup_iterations must be integer in [2,iterations]')
        else:raise ValueError('Unknown temporal sampling rule')
    o = c.operator
    for name in ('reference_workers','local_reference_workers'):
        if name in o:
            if isinstance(o[name],bool) or not isinstance(o[name],int) or o[name]<1:
                raise ValueError('Reference preparation requires a positive integer worker count')
    low, high = np.asarray(o.center_min), np.asarray(o.center_max)
    r0, r1 = np.asarray(o.radius_min), np.asarray(o.radius_max)
    if low.shape != (2, 2) or high.shape != (2, 2) or r0.shape != (2,) or r1.shape != (2,):
        raise ValueError('Exactly two hole centers and radii required')
    if not all(np.isfinite(v).all() for v in (low, high, r0, r1)) or np.any(low >= high) or np.any(r0 <= 0) or np.any(r0 >= r1):
        raise ValueError('Geometry prior bounds must be finite and strictly ordered')
    if 'augmentation' in o:
        if o.augmentation!='axis_reflections' or not np.allclose(low+high,2*center,rtol=0,atol=1e-14):
            raise ValueError('Axis-reflection augmentation requires an invariant center prior and supported isometry')
    if o.clearance <= 0 or o.maximum_sampling_attempts < o.train_conditions+o.test_conditions:
        raise ValueError('Positive wall clearance and sufficient sampling budget required')
    if min(o.train_conditions, o.test_conditions, o.condition_batch, c.sample.Nt, c.train.iterations) < 1:
        raise ValueError('Training/test/batch/time/iteration counts must be positive')
    if c.physical.terminal_time <= 0 or min(c.physical.mobility,c.physical.epsilon,c.physical.beta,c.nl.delta,c.geometry.mesh_size) <= 0:
        raise ValueError('Physical/time/operator scales must be positive')
    for dt in (c.reference.dt,c.reference.refined_dt):
        steps = round(c.physical.terminal_time/dt)
        if abs(steps*dt-c.physical.terminal_time)>1e-12 or steps%(c.reference.snapshots-1):
            raise ValueError('Reference time steps must align the full interval and snapshots')
    for index in o.local_validation_indices:
        if not o.train_conditions <= index < o.train_conditions+o.test_conditions:
            raise ValueError('Local validation must use declared held-out indices')
    if len(set(o.local_validation_indices))!=len(o.local_validation_indices):
        raise ValueError('Local validation indices must be unique; empty explicitly defers FEM')
    if 'precompute' in o:
        p=o.precompute
        if 'reuse_mode' in p and p.reuse_mode not in ('copy','hardlink'):
            raise ValueError('precompute.reuse_mode must be copy or hardlink')
        total=o.train_conditions+o.test_conditions
        if isinstance(p.workers,bool) or not isinstance(p.workers,int) or p.workers<1:
            raise ValueError('Precomputation requires a positive integer worker count')
        for ids,limit,count in ((p.reserved_test_draws,total,o.test_conditions),
                               (p.monitor_train_indices,o.train_conditions,len(p.monitor_train_indices))):
            if not count or len(ids)!=count or len(set(ids))!=len(ids) or any(isinstance(i,bool) or not isinstance(i,int) or not 0<=i<limit for i in ids):
                raise ValueError('Explicit unique in-range reserved/monitor geometry indices required')
        if not Path(p.source_path).is_absolute() or Path(p.source_path)==Path(o.dataset_path):
            raise ValueError('Use a separate absolute source bank path')


def valid_holes(holes, c):
    h = np.asarray(holes)
    if h.shape != (2,3) or not np.isfinite(h).all() or np.any(h[:,2] <= 0):
        return False
    center, axes = np.asarray(c.geometry.center), np.asarray(c.geometry.axes)
    # Sufficient Euclidean-ball containment bound inside the scaled ellipse.
    enclosed = np.linalg.norm((h[:,:2]-center)/axes,axis=1)+(h[:,2]+c.operator.clearance)/min(axes)
    gap = np.linalg.norm(h[0,:2]-h[1,:2])-h[0,2]-h[1,2]
    return bool(np.all(enclosed < 1) and gap > c.operator.clearance)


def sample_geometries(c):
    rng = np.random.default_rng(c.operator.dataset_seed)
    result=[]
    for _ in range(c.operator.maximum_sampling_attempts):
        centers=rng.uniform(c.operator.center_min,c.operator.center_max)
        radii=rng.uniform(c.operator.radius_min,c.operator.radius_max)
        holes=np.column_stack((centers,radii))
        if valid_holes(holes,c):
            result.append(holes)
        if len(result)==c.operator.train_conditions+c.operator.test_conditions:
            if 'precompute' in c.operator:
                held=list(c.operator.precompute.reserved_test_draws)
                order=[i for i in range(len(result)) if i not in held]+held
                result=[result[i] for i in order]
            return np.stack(result)
    raise RuntimeError('Insufficient valid geometry draws; no repeated/padded replacement')


def condition(holes,c):
    angles=2*np.pi*np.arange(c.geometry.boundary_outer)/c.geometry.boundary_outer
    cs,sn=np.cos(angles),np.sin(angles)
    axes=np.asarray(c.geometry.axes)
    points=np.asarray(c.geometry.center)+np.stack((axes[0]*cs,axes[1]*sn),axis=1)
    normal=np.stack((cs/axes[0],sn/axes[1]),axis=1)
    normal/=np.linalg.norm(normal,axis=1)[:,None]
    weights=2*np.pi/len(angles)*np.sqrt((axes[0]*sn)**2+(axes[1]*cs)**2)
    rows=[np.column_stack((points,normal,np.zeros(len(points))))];measure=[weights]
    angles=2*np.pi*np.arange(c.geometry.boundary_per_hole)/c.geometry.boundary_per_hole
    unit=np.stack((np.cos(angles),np.sin(angles)),axis=1)
    for cx,cy,r in holes:
        rows.append(np.column_stack((np.array([cx,cy])+r*unit,-unit,np.ones(len(unit)))))
        measure.append(np.full(len(unit),2*np.pi*r/len(unit)))
    return np.concatenate(rows),np.concatenate(measure)


def case_config(c,holes,index):
    result=ConfigDict(c.to_dict())
    result.geometry.holes=tuple(tuple(float(v) for v in row) for row in holes)
    result.reference.root=str(Path(c.operator.dataset_path)/f'case_{index:03d}')
    result.reference.path=str(Path(result.reference.root)/'reference.npz')
    return result


def specification(c):
    result=json.loads(json.dumps(dict(geometry=c.geometry.to_dict(),physical=c.physical.to_dict(),
        initial=c.initial.to_dict(),nl=c.nl.to_dict(),reference=c.reference.to_dict(),
        boundary_counts=(c.geometry.boundary_outer,c.geometry.boundary_per_hole),
        prior={key:c.operator[key] for key in ('dataset_seed','train_conditions','test_conditions',
            'center_min','center_max','radius_min','radius_max','clearance')})))
    if 'precompute' in c.operator:
        result['prior']['reserved_test_draws']=list(c.operator.precompute.reserved_test_draws)
    return result


def evaluation_indices(manifest):
    """Legacy full references or explicit monitor-only mixed bank."""
    if 'format' not in manifest:
        return manifest['train_indices'],manifest['train_indices']+manifest['test_indices']
    if manifest['format']!='operator_only_v1':
        raise ValueError('Unknown geometry bank format')
    monitor=manifest['monitor_train_indices']
    if not monitor or len(set(monitor))!=len(monitor) or any(i not in manifest['train_indices'] for i in monitor):
        raise ValueError('Training monitors must be unique members of the training partition')
    indices=monitor+manifest['test_indices']
    for record in manifest['conditions']:
        if record['has_reference']!=(record['index'] in indices):
            raise ValueError('Reference availability must match explicit evaluation partitions')
    return monitor,indices
