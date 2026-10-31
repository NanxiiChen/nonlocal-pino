"""Shared physical inputs and validation; imports no JAX or FEM implementation."""
import json
from pathlib import Path
import numpy as np


def validate_config(c):
    if 'loss_balance' in c.train:
        b=c.train.loss_balance
        if b.kind not in ('fixed','gradient_norm','stagger'):raise ValueError('Unknown loss_balance.kind')
        if c.arch.model!='global_mixed' or c.arch.field_layout!='shared' or c.formulation.chemical!='learned' or c.nl.method!='nonlocal':
            raise ValueError('Two-loss balancing requires shared global mixed nonlocal fields')
        if 'windows' in c.train or 'causal' in c.loss:
            raise ValueError('Balancing pilot rejects windows/causal weighting')
        if 'phase_metric' in c.loss:
            if c.loss.phase_metric.kind!='neumann_inverse':raise ValueError('Balancing metric must be instantaneous Neumann inverse')
        elif 'evolution_scale' not in c.loss:raise ValueError('Raw balancing requires an explicit evolution_scale')
        if b.kind=='gradient_norm':
            if b.refinement!='freeze':raise ValueError('Gradient-norm refinement must freeze positive joint weights')
            if isinstance(b.update_every,bool) or not isinstance(b.update_every,int) or b.update_every<1:
                raise ValueError('Positive integer balancing update_every required')
            if not np.isfinite([b.alpha,b.norm_floor,b.norm_ceiling,b.weight_floor,b.weight_cap]).all() or not 0<b.alpha<=1 or not 0<b.norm_floor<b.norm_ceiling or not 0<b.weight_floor<1<b.weight_cap:
                raise ValueError('Explicit finite positive balancing alpha, norm bounds and weight bounds required')
        else:
            if b.refinement!='joint':raise ValueError('Fixed/stagger refinement must use the joint objective')
        if b.kind=='stagger':
            for key in ('stagger_steps','switch_every'):
                value=b[key]
                if isinstance(value,bool) or not isinstance(value,int) or value<1:raise ValueError('Positive integer '+key+' required')
            if b.stagger_steps>=c.train.iterations or b.stagger_steps%(2*b.switch_every):
                raise ValueError('Stagger needs complete alternating cycles followed by joint training')
    if 'optimizer' in c.train.refinement:
        s=c.train.refinement
        if s.optimizer!='newton_cg' or not np.isfinite(s.step_tolerance) or s.step_tolerance<=0 or not 0<s.c1<s.c2<1:
            raise ValueError('Newton-CG requires positive step_tolerance and 0<c1<c2<1')
    eliminated=False
    if 'formulation' in c:
        if c.formulation.chemical not in ('learned','eliminated'):
            raise ValueError('formulation.chemical must be learned/eliminated')
        eliminated=c.formulation.chemical=='eliminated'
        if eliminated and c.nl.method!='nonlocal':raise ValueError('Chemical elimination currently requires explicit nonlocal spatial differentiation')
    if 'model' in c.arch:
        if c.arch.model not in ('global_basis','global_mixed'):raise ValueError('Unknown global model')
        if c.arch.model=='global_basis' and not eliminated:raise ValueError('Global-basis model requires exact native chemical elimination')
        if c.arch.model=='global_mixed':
            if eliminated or c.nl.method!='nonlocal':raise ValueError('Global mixed requires learned nonlocal chemical field')
            if c.arch.field_layout not in ('shared','separate'):raise ValueError('Global mixed field_layout must be shared/separate')
            if not np.isfinite(c.arch.chemical_scale) or c.arch.chemical_scale<=0:raise ValueError('Positive finite chemical output scale required')
        if isinstance(c.arch.rank,bool) or not isinstance(c.arch.rank,int) or c.arch.rank<1:raise ValueError('Positive integer global basis rank required')
        if not np.isfinite(c.arch.output_scale) or c.arch.output_scale<=0:raise ValueError('Positive global output initialization scale required')
        if c.phase.lifting!='logarithmic_additive':raise ValueError('Global basis requires explicit logarithmic_additive coordinate')
        if 'phase_metric' in c.loss and c.loss.phase_metric.kind=='integral_resolvent':
            if c.sample.Nt<3*c.arch.rank or c.phase.initial_time_scale!=c.loss.phase_metric.time_scale:
                raise ValueError('Integral global basis needs at least3*rank quadrature nodes and matching log coordinates')
        if 'energy_weight' in c.loss and c.sample.Nt<6*c.arch.rank:raise ValueError('Energy integration needs at least6*rank quadrature nodes')
    mapped='phase' in c and 'output_map' in c.phase
    if mapped:
        q=c.phase.output_map
        if q.kind!='tanh' or not np.isfinite(q.amplitude) or q.amplitude<=0:raise ValueError('Tanh map requires positive finite amplitude')
        if not eliminated or c.nl.method!='nonlocal' or c.arch.model!='global_basis' or 'windows' in c.train:
            raise ValueError('Tanh map requires one global-basis eliminated nonlocal model')
        if 'phase_metric' in c.loss and c.loss.phase_metric.kind not in ('neumann_inverse','biharmonic_resolvent'):
            raise ValueError('Tanh requires instantaneous residuals; polynomial chemical integration is not unaliased')
    if 'residual_transform' in c.loss:
        kind=c.loss.residual_transform.kind
        if kind not in ('none','inverse_tanh_jacobian'):raise ValueError('Unknown phase residual transform')
        if kind=='inverse_tanh_jacobian' and not mapped:raise ValueError('Inverse tanh Jacobian requires the actual projected tanh state map')
    if 'energy_weight' in c.loss:
        if not np.isfinite(c.loss.energy_weight) or c.loss.energy_weight<0:raise ValueError('Nonnegative finite energy weight required')
        if c.loss.phase_metric.kind!='integral_resolvent':raise ValueError('Energy balance pilot requires global integral residual')
    if 'evolution_scale' in c.loss:
        if eliminated or c.nl.method!='nonlocal' or 'phase_metric' in c.loss or not np.isfinite(c.loss.evolution_scale) or c.loss.evolution_scale<=0:
            raise ValueError('Raw mixed evolution_scale must be finite positive without a spatial metric')
    if 'chemical_scale' in c.loss:
        scaled_raw='evolution_scale' in c.loss
        mixed_metric='phase_metric' in c.loss and c.loss.phase_metric.kind in ('neumann_inverse','mixed_block')
        if eliminated or not (scaled_raw or mixed_metric) or not np.isfinite(c.loss.chemical_scale) or c.loss.chemical_scale<=0:
            raise ValueError('Mixed chemical_scale requires a supported residual norm and finite positive scale')
    if 'causal' in c.loss:
        q=c.loss.causal
        if not eliminated or c.loss.phase_metric.kind!='neumann_inverse' or c.sample.temporal.kind!='fixed_nodes' or not c.sample.include_initial_time:
            raise ValueError('Causal control requires fixed ordered full-horizon nodes, initial evolution and eliminated Neumann inverse')
        if not np.isfinite([q.strength,q.minimum]).all() or q.strength<=0 or not 0<q.minimum<1 or q.refinement!='uniform':
            raise ValueError('Positive causal strength, minimum in(0,1), uniform refinement required')
    if 'windows' in c.train:
        w=c.train.windows
        if not eliminated or c.nl.method!='nonlocal' or w.spacing!='logarithmic' or w.metric!='biharmonic_resolvent':
            raise ValueError('Windows require eliminated nonlocal CH and explicit logarithmic/biharmonic choices')
        if 'temporal' in c.sample or 'phase_metric' in c.loss:raise ValueError('Window-local time/metric replaces global temporal/phase_metric settings')
        if not w.reset_optimizer or not w.warm_start:raise ValueError('Window pilot requires explicit warm_start and reset_optimizer')
        for name in ('count','first_soap','soap_each','first_lbfgs','lbfgs_each','first_maxfun','maxfun_each'):
            value=w[name]
            if isinstance(value,bool) or not isinstance(value,int) or value<1:raise ValueError('Positive integer windows.'+name+' required')
        if not np.isfinite(w.time_scale) or w.time_scale<=0:raise ValueError('Positive finite windows.time_scale required')
        if w.first_soap+(w.count-1)*w.soap_each!=c.train.iterations or w.first_lbfgs+(w.count-1)*w.lbfgs_each!=c.train.refinement.iterations:
            raise ValueError('Window SOAP/L-BFGS budgets must sum to declared total iterations')
        if w.first_maxfun+(w.count-1)*w.maxfun_each!=c.train.refinement.max_evaluations:
            raise ValueError('Window function-evaluation caps must sum to declared total')
    if 'phase_metric' in c.loss:
        m=c.loss.phase_metric
        if c.nl.method!='nonlocal':raise ValueError('Phase metric requires explicit nonlocal spatial differentiation')
        if m.kind=='biharmonic_resolvent':
            if not np.isfinite(m.time_scale) or m.time_scale<=0:raise ValueError('Biharmonic metric requires finite positive time_scale')
        elif m.kind=='space_time_resolvent':
            if not eliminated or c.sample.include_initial_time:raise ValueError('Space-time metric requires eliminated chemical law and positive sample times')
            if 'temporal' not in c.sample or c.sample.temporal.curriculum:raise ValueError('Space-time metric requires explicit full-horizon temporal sampling')
            if 'time_nodes' in c.train.refinement and c.train.refinement.time_nodes[-1]>=c.physical.terminal_time:
                raise ValueError('Space-time refinement nodes must be strictly below T; terminal residual is appended explicitly')
        elif m.kind=='integral_resolvent':
            if not eliminated or c.sample.include_initial_time:raise ValueError('Integral residual requires eliminated chemical law and explicitly provided global nodes')
            if c.sample.temporal.kind!='fixed_nodes':raise ValueError('Integral residual requires fixed global temporal quadrature')
            if not np.isfinite(m.time_scale) or m.time_scale<=0:raise ValueError('Positive integral time_scale required')
            if tuple(c.sample.temporal.nodes)!=tuple(c.train.refinement.time_nodes):raise ValueError('SOAP/refinement integral nodes must be identical')
            if 'normalization' in m and m.normalization!='time_average':raise ValueError('Only explicit time_average integral normalization supported')
        elif m.kind=='mixed_block':
            if eliminated or not c.sample.include_initial_time or c.sample.temporal.kind!='fixed_nodes':raise ValueError('Mixed block requires both learned fields, fixed nodes and initial residual')
            if not np.isfinite(m.time_scale) or m.time_scale<=0:raise ValueError('Positive mixed integration time scale required')
            if tuple(c.sample.temporal.nodes)!=tuple(c.train.refinement.time_nodes):raise ValueError('Mixed SOAP/refinement nodes must agree')
            if c.arch.model!='global_mixed' or c.sample.Nt<3*c.arch.rank or c.phase.initial_time_scale!=m.time_scale:raise ValueError('Unaliased mixed block needs global mixed basis, >=3*rank nodes, matching log coordinates')
        elif m.kind=='neumann_inverse':
            pass # Applies to the evolution row of both mixed and eliminated CH.
        else:raise ValueError('Unknown phase residual metric')
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
    if 'time_nodes' in c.train.refinement:
        nodes=np.asarray(c.train.refinement.time_nodes)
        if nodes.shape!=(c.train.refinement.time_points,) or not np.isfinite(nodes).all() or np.any(np.diff(nodes)<=0) or nodes[0]<=0 or nodes[-1]>c.physical.terminal_time:
            raise ValueError('Refinement nodes must be finite ordered(time_points,) within (0,T]')
    if 'phase' in c:
        if c.phase.lifting not in ('additive','exponential_blend','logarithmic_additive','logarithmic_blend','logarithmic_increment'):
            raise ValueError('Unknown phase lifting')
        if not np.isfinite(c.phase.initial_time_scale) or c.phase.initial_time_scale<=0:
            raise ValueError('phase.initial_time_scale must be finite positive')
    if 'initial_feature' in c.arch and not isinstance(c.arch.initial_feature,bool):raise ValueError('arch.initial_feature must be boolean')
    center,axes=np.asarray(c.geometry.center),np.asarray(c.geometry.axes)
    holes=np.asarray(c.geometry.holes)
    if center.shape!=(2,) or axes.shape!=(2,) or not np.isfinite([center,axes]).all() or np.any(axes<=0):
        raise ValueError('Require finite 2D center and positive ellipse axes')
    if holes.ndim!=2 or holes.shape[1]!=3 or len(holes)<1 or not np.isfinite(holes).all() or np.any(holes[:,2]<=0):
        raise ValueError('Require finite nonempty holes(cx,cy,r>0)')
    for i,(x,y,r) in enumerate(holes):
        # Conservative sufficient containment bound, validated before meshing.
        if np.linalg.norm((np.array([x,y])-center)/axes)+r/min(axes)>=1:
            raise ValueError('Hole must lie strictly inside ellipse')
        for xx,yy,rr in holes[:i]:
            if np.hypot(x-xx,y-yy)<=r+rr:raise ValueError('Holes must not overlap')
    positive=[c.geometry.mesh_size,c.physical.beta,c.physical.epsilon,c.physical.mobility,
        c.physical.terminal_time,c.initial.wall_width,c.initial.amplitude,c.nl.delta,
        c.train.lr_init,c.loss.phase_weight,c.loss.boundary_phi_weight,
        c.loss.boundary_mu_weight,c.reference.dt,c.reference.refined_dt,
        c.reference.phase_tolerance,c.reference.chemical_tolerance,*c.reference.mesh_sizes]
    if not eliminated:positive.append(c.loss.chemical_weight)
    if not np.isfinite(positive).all() or min(positive)<=0:raise ValueError('All scales/weights must be finite positive')
    if 'kind' in c.initial:
        if c.initial.kind!='multimode':raise ValueError('Explicit initial.kind must be multimode; legacy schema omits kind')
        waves=np.asarray(c.initial.waves)
        if waves.ndim!=2 or waves.shape[1]!=4 or len(waves)<2 or not np.isfinite(waves).all():
            raise ValueError('Initial waves must be finite(K,4): kx,ky,amplitude,phase with K>=2')
        if np.any(np.sum(waves[:,:2]**2,axis=1)==0) or not np.array_equal(waves[:,:2],np.round(waves[:,:2])):
            raise ValueError('Initial wave vectors must be nonzero integer pairs')
        if not np.isfinite(c.initial.scale) or c.initial.scale<=0 or c.initial.amplitude>=1:
            raise ValueError('Multimode IC requires positive scale and amplitude<1')
    if c.nl.method not in ('nonlocal','local_ad') or c.nl.kernel!='quadratic_visible':
        raise ValueError('Require nonlocal/local_ad and explicit quadratic_visible kernel')
    if c.train.compute_dtype!='float64':raise ValueError('Pilot requires explicit float64')
    if c.reference.refined_dt>=c.reference.dt or not c.reference.mesh_sizes[1]<c.reference.mesh_sizes[0]:
        raise ValueError('Reference requires smaller dt and mesh size')
    for name,v in [('Nt',c.sample.Nt),('iterations',c.train.iterations),('snapshots',c.reference.snapshots),
        ('eval_every',c.log.eval_every),('checkpoint_every',c.log.checkpoint_every),
        ('hidden_dim',c.arch.hidden_dim),('num_layers',c.arch.num_layers),
        ('boundary_outer',c.geometry.boundary_outer),('boundary_per_hole',c.geometry.boundary_per_hole)]:
        if isinstance(v,bool) or not isinstance(v,int) or v<1:raise ValueError(name+' must be a positive integer')
    if c.reference.snapshots<2:raise ValueError('Need at least2 snapshots')
    for dt in (c.reference.dt,c.reference.refined_dt):
        steps=round(c.physical.terminal_time/dt)
        if abs(steps*dt-c.physical.terminal_time)>1e-12 or steps%(c.reference.snapshots-1):
            raise ValueError('dt must divide duration and align snapshots')


def initial_field(points,c,*,xp):
    """Legacy perturbations vanish to order4 at each analytic wall.

    Hence both normal phi and normal local chemical potential vanish at t0.
    The geometry operator's explicit fixed_ambient variant instead restricts
    the identical spatial function to each domain, without a wall envelope.
    No future boundary or trajectory is encoded in either initial function.
    """
    if 'kind' in c.initial and c.initial.kind=='fixed_ambient':
        from examples.confined_ch2d.operator_problem import fixed_initial
        return fixed_initial(points,c,xp=xp)
    z=(points-xp.asarray(c.geometry.center))/xp.asarray(c.geometry.axes)
    wall=.5*min(c.geometry.axes)*(1-xp.sum(z*z,axis=-1))
    width=c.initial.wall_width
    envelope=wall**4/(wall**4+width**4)
    for x,y,r in c.geometry.holes:
        wall=(xp.sum((points-xp.asarray((x,y)))**2,axis=-1)-r*r)/(2*r)
        envelope=envelope*wall**4/(wall**4+width**4)
    if 'kind' in c.initial:
        waves=xp.asarray(c.initial.waves)
        angles=xp.pi*xp.sum(z[...,None,:]*waves[:,:2],axis=-1)+waves[:,3]
        pattern=xp.tanh(xp.sum(waves[:,2]*xp.cos(angles),axis=-1)/c.initial.scale)
        return c.initial.amplitude*envelope*pattern
    pattern=xp.cos(2*xp.pi*z[...,0])+.6*xp.sin(1.5*xp.pi*z[...,1])
    return c.initial.amplitude*envelope*pattern


def identity(c):
    return dict(model='confined_quartic_ch_visible_regional_v1',geometry=c.geometry.to_dict(),
        physical=c.physical.to_dict(),initial=c.initial.to_dict(),delta=c.nl.delta,kernel=c.nl.kernel,
        boundary='neutral_wall_and_impermeable_all_components')


def boundary_samples(c):
    g=c.geometry;n=g.boundary_outer
    t=(np.arange(n)+.5)*2*np.pi/n
    axes=np.asarray(g.axes);center=np.asarray(g.center)
    p=center+np.stack((np.cos(t),np.sin(t)),axis=-1)*axes
    normal=(p-center)/(axes*axes);normal/=np.linalg.norm(normal,axis=1)[:,None]
    weight=2*np.pi/n*np.sqrt((axes[0]*np.sin(t))**2+(axes[1]*np.cos(t))**2)
    points,normals,weights,labels=[p],[normal],[weight],[np.zeros(n,dtype=int)]
    for i,(x,y,r) in enumerate(g.holes,1):
        n=g.boundary_per_hole;t=(np.arange(n)+.5)*2*np.pi/n
        e=np.stack((np.cos(t),np.sin(t)),axis=-1)
        points.append(np.array([x,y])+r*e);normals.append(-e)
        weights.append(np.full(n,2*np.pi*r/n));labels.append(np.full(n,i))
    return tuple(np.concatenate(a) for a in (points,normals,weights,labels))


def load_reference(c):
    validate_config(c)
    path=Path(c.reference.path)
    with np.load(path,allow_pickle=False) as f:d={k:f[k].copy() for k in f.files}
    if json.loads(str(d['identity']))!=json.loads(json.dumps(identity(c))):raise ValueError('Reference identity mismatch: '+str(path))
    n=len(d['points']);nt=c.reference.snapshots
    if d['points'].shape!=(n,2) or d['volumes'].shape!=(n,) or not np.isfinite(d['points']).all() or not np.isfinite(d['volumes']).all() or np.any(d['volumes']<=0):raise ValueError('Invalid reference quadrature')
    from examples.confined_ch2d.geometry import quadrature
    reconstructed,volumes=quadrature(d['mesh_points'],d['mesh_cells'])
    np.testing.assert_array_equal(d['points'],reconstructed)
    np.testing.assert_array_equal(d['volumes'],volumes)
    if np.unique(d['points'],axis=0).shape[0]!=n:raise ValueError('Duplicate quadrature points')
    level=np.sum(((d['points']-np.asarray(c.geometry.center))/np.asarray(c.geometry.axes))**2,axis=1)
    if np.any(level>=1):raise ValueError('Quadrature outside ellipse')
    for cx,cy,r in c.geometry.holes:
        if np.any(np.sum((d['points']-np.array([cx,cy]))**2,axis=1)<=r*r):raise ValueError('Quadrature inside impermeable hole')
    np.testing.assert_allclose(d['time'],np.linspace(0,c.physical.terminal_time,nt),atol=1e-14,rtol=0)
    for k in ('phi_local','mu_local','phi_nonlocal','mu_nonlocal'):
        if d[k].shape!=(nt,n) or not np.isfinite(d[k]).all():raise ValueError('Invalid reference field '+k)
    checks=json.loads(path.with_suffix('.json').read_text())
    if checks['status']!='passed':raise ValueError('Reference refinement not validated')
    return d


def relative(a,b,volumes):
    norm=lambda v:np.sqrt(np.sum(np.asarray(v)**2*volumes,axis=-1).sum())
    denominator=norm(b)
    if denominator<=0:raise ValueError('Relative norm requires a nonzero target')
    return float(norm(np.asarray(a)-b)/denominator)


def initial_laplacian(points,c):
    """Independent NumPy product/chain-rule Laplacian of the analytic IC."""
    if 'kind' in c.initial and c.initial.kind=='fixed_ambient':
        from examples.confined_ch2d.operator_problem import fixed_initial_laplacian
        return fixed_initial_laplacian(points,c)
    x=np.asarray(points);axes=np.asarray(c.geometry.axes);z=(x-np.asarray(c.geometry.center))/axes
    scale=min(axes);w4=c.initial.wall_width**4
    factors=[(.5*scale*(1-np.sum(z*z,axis=-1)),-scale*z/axes,np.full(x.shape[:-1],-scale*np.sum(1/axes**2)))]
    for cx,cy,r in c.geometry.holes:
        d=x-np.array([cx,cy]);factors.append(((np.sum(d*d,axis=-1)-r*r)/(2*r),d/r,np.full(x.shape[:-1],2/r)))
    E=np.ones(x.shape[:-1]);G=np.zeros_like(x);L=np.zeros_like(E)
    for s,g,l in factors:
        denominator=s**4+w4
        H=s**4/denominator
        first=4*w4*s**3/denominator**2
        second=12*w4*s**2/denominator**2-32*w4*s**6/denominator**3
        grad=first[...,None]*g;lap=second*np.sum(g*g,axis=-1)+first*l
        Lnew=L*H+2*np.sum(G*grad,axis=-1)+E*lap
        Gnew=G*H[...,None]+E[...,None]*grad
        E=E*H;G=Gnew;L=Lnew
    if 'kind' in c.initial:
        waves=np.asarray(c.initial.waves);k=np.pi*waves[:,:2]/axes
        angles=np.sum((x-np.asarray(c.geometry.center))[...,None,:]*k,axis=-1)+waves[:,3]
        P=np.sum(waves[:,2]*np.cos(angles),axis=-1)
        GP=np.sum((-waves[:,2]*np.sin(angles))[...,None]*k,axis=-2)
        LP=-np.sum(waves[:,2]*np.cos(angles)*np.sum(k*k,axis=-1),axis=-1)
        pattern=np.tanh(P/c.initial.scale)
        first=(1-pattern**2)/c.initial.scale
        second=-2*pattern*(1-pattern**2)/c.initial.scale**2
        return c.initial.amplitude*(L*pattern+2*first*np.sum(G*GP,axis=-1)+E*(second*np.sum(GP*GP,axis=-1)+first*LP))
    a=2*np.pi*z[...,0];b=1.5*np.pi*z[...,1]
    pattern=np.cos(a)+.6*np.sin(b)
    grad_pattern=np.stack((-2*np.pi/axes[0]*np.sin(a),.6*1.5*np.pi/axes[1]*np.cos(b)),axis=-1)
    lap_pattern=-(2*np.pi/axes[0])**2*np.cos(a)-.6*(1.5*np.pi/axes[1])**2*np.sin(b)
    return c.initial.amplitude*(L*pattern+2*np.sum(G*grad_pattern,axis=-1)+E*lap_pattern)
