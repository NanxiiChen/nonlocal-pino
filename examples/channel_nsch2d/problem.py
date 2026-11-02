"""Known geometry/initial/driving data; no numerical solver or JAX imports."""
import numpy as np


def smoothstep(z,*,xp):
    z=xp.clip(z,0.,1.)
    return z*z*z*(10.+z*(-15.+6.*z))


def walls(x,amplitudes,c,*,xp):
    s=(x-c.geometry.center)/c.geometry.half_length
    b=xp.maximum(1.-s*s,0.)**4
    bottom=amplitudes[1]*b
    top=c.geometry.height-amplitudes[0]*b
    return bottom,top


def wall_slopes(x,amplitudes,c,*,xp):
    s=(x-c.geometry.center)/c.geometry.half_length
    derivative=-8.*s*xp.maximum(1.-s*s,0.)**3/c.geometry.half_length
    return amplitudes[1]*derivative,-amplitudes[0]*derivative


def initial_field(points,c,*,xp):
    delta=points-xp.asarray(c.initial.center)
    radius2=xp.sum(delta*delta,axis=-1)
    z=.5+(radius2-c.initial.radius**2)/(4*c.initial.radius*c.initial.transition)
    return 2*smoothstep(z,xp=xp)-1.


def flow_rate(t,c,*,xp):
    return c.physical.mean_velocity*c.geometry.height*smoothstep(t/c.physical.ramp_time,xp=xp)


def material(c):
    A=3*c.physical.surface_tension/(2*np.sqrt(2)*c.physical.epsilon)
    return float(A),float(A*c.physical.epsilon**2)


def identity(c):
    return dict(formulation='matched-nsch-open-traction-v1',geometry=c.geometry.to_dict(),
                physical=c.physical.to_dict(),initial=c.initial.to_dict())


def validate_config(c):
    g,p=c.geometry,c.physical
    integral=c.nl.method=='integral_moment'
    if integral:
        from examples.channel_nsch2d.integral_operators import validate_integral
        validate_integral(c)
        if ('derivatives' not in c or 'chemical_constraint'in c
                or 'time_derivative'not in c or any(k in c for k in ('balance','causal'))):
            raise ValueError('Integral pilot requires explicit timeJVP/PDE+BC derivatives and fixed joint losses')
    positive=(g.length,g.height,g.half_length,p.density,p.viscosity,p.mean_velocity,
        p.surface_tension,p.epsilon,p.mobility,p.terminal_time,p.ramp_time,
        c.initial.radius,c.initial.transition)
    if not all(np.isfinite(v) and v>0 for v in positive):
        raise ValueError('Geometry/material/time/initial scales must be positive finite')
    if not (0<=g.amplitude_min<g.amplitude_max<g.height/2):
        raise ValueError('Independent inward amplitudes require 0<=min<max<H/2')
    if not (g.half_length<g.center<g.length-g.half_length):
        raise ValueError('Compact wall indentation must preserve straight inlet/outlet')
    if not (0<=g.upper<=g.amplitude_max and 0<=g.lower<=g.amplitude_max):
        raise ValueError('Requested reference geometry outside amplitude bounds')
    integers=(c.sample.Nx,c.sample.Ny,c.sample.Nt,c.sample.boundary_points,
        c.train.iterations,c.operator.train_conditions,c.operator.condition_batch,
        c.log.eval_every,c.log.checkpoint_every,c.reference.snapshots)
    if any(isinstance(v,bool) or not isinstance(v,int) or v<1 for v in integers):
        raise ValueError('Sampling, training, condition and logging counts must be positive integers')
    if min(c.sample.Nx,c.sample.Ny)<4 or c.sample.Nt<1:
        raise ValueError('Require Nx,Ny>=4 and Nt>=1')
    if c.train.iterations<2 or c.reference.snapshots<2:
        raise ValueError('Require at least two updates and two reference snapshots')
    if 'online' in c.operator:
        from examples.channel_nsch2d.online import validate_online
        validate_online(c)
    if c.train.compute_dtype not in ('float32','float64'):
        raise ValueError('Explicit compute_dtype must be float32 or float64')
    if 'microbatch' in c.train:
        m=c.train.microbatch
        if set(m)!={'kind','size'} or m.kind!='time':
            raise ValueError('Microbatch requires explicit time kind and size')
        if (isinstance(m.size,bool) or not isinstance(m.size,int) or m.size<1
                or m.size>c.sample.Nt or c.sample.Nt%m.size):
            raise ValueError('Time microbatch size must be a positive integer divisor of Nt')
        term_balance=('balance' in c and 'grouping' in c.balance and c.balance.grouping=='terms')
        if ('stagger' in c or ('balance' in c and not term_balance)
                or any(k in c.train for k in ('continuation','refinement'))):
            raise ValueError('Time microbatch requires cold uniform joint losses or explicit termwise balance')
    if 'causal' in c:
        q=c.causal
        adaptive_causal='adaptation' in q
        fields=({'kind','adaptation','weight_gradient','boundary'} if adaptive_causal else
                {'kind','epsilon_values','steps_per_epsilon','weight_gradient','boundary'})
        if (set(q)!=fields or q.kind!='coupled_prefix_mse' or q.weight_gradient!='stop'
                or q.boundary!='same_time_weights'):
            raise ValueError('Causal requires explicit coupled_prefix_mse, stopped weights and same_time_weights BCs')
        if adaptive_causal:
            a=q.adaptation
            names=('eps_init','step_size','max_eps','max_last_weight','min_mean_weight')
            if (set(a)!={'kind','geometry_reduction',*names} or a.kind!='weight_feedback'
                    or a.geometry_reduction!='minimum'):
                raise ValueError('Causal adaptation requires explicit weight_feedback and minimum geometry reduction')
            if any(isinstance(a[k],(bool,np.bool_)) or not isinstance(a[k],(int,float))
                   or not np.isfinite(a[k]) for k in names):
                raise ValueError('Causal adaptation numbers must be finite numeric scalars')
            if (not 0<a.eps_init<=a.max_eps or a.step_size<=1
                    or not 1/c.sample.Nt<a.min_mean_weight<a.max_last_weight<1):
                raise ValueError('Causal adaptation requires 0<eps_init<=max_eps, step_size>1 and 1/Nt<min_mean_weight<max_last_weight<1')
        else:
            eps=np.asarray(q.epsilon_values)
            if (eps.ndim!=1 or not len(eps) or eps.dtype.kind not in 'fiu'
                    or any(isinstance(v,(bool,np.bool_)) for v in q.epsilon_values)
                    or not np.isfinite(eps).all() or np.any(eps<=0) or np.any(np.diff(eps)<=0)):
                raise ValueError('Causal epsilon_values must be positive finite and strictly increasing')
            if (isinstance(q.steps_per_epsilon,bool) or not isinstance(q.steps_per_epsilon,int)
                    or q.steps_per_epsilon<1 or q.steps_per_epsilon*len(eps)!=c.train.iterations):
                raise ValueError('Causal epsilon stages must exactly cover the finite training budget')
        if ('microbatch' not in c.train or 'derivatives' not in c or c.sample.Nt<2
                or any(k in c for k in ('balance','stagger','stress_formulation','droplet_metric','phase_metric','residual_transform'))
                or any(k in c.train for k in ('continuation','refinement'))):
            raise ValueError('Causal requires cold GMLS time microbatches and joint original residuals')
    scales=(c.loss.momentum_scale,c.loss.phase_scale,c.loss.chemical_scale,
        c.loss.continuity_scale,c.loss.boundary_mu_scale,c.loss.boundary_phi_scale,
        c.loss.traction_scale,c.loss.pressure_scale)
    if not all(np.isfinite(v) and v>0 for v in scales):
        raise ValueError('All PDE/boundary/output scales must be positive finite')
    if 'chemical_residual_scale'in c.loss:
        s=c.loss.chemical_residual_scale
        if (not np.isscalar(s)or np.asarray(s).dtype.kind not in 'fiu'
                or not np.isfinite(s)or s<=0):
            raise ValueError('chemical_residual_scale must be an explicit positive finite real scalar')
        if 'chemical_constraint'in c:
            raise ValueError('chemical_residual_scale is unused when the chemical row is eliminated')
    if 'lifting' in c:
        if c.lifting.kind!='normalized' or not all(np.isfinite(v) and v>0 for v in
                (c.lifting.time_scale,c.lifting.inlet_length_scale)):
            raise ValueError('Normalized lifting requires explicit positive time/inlet scales')
    if 'velocity' in c:
        if c.velocity.kind!='streamfunction':
            raise ValueError('The explicit velocity extension only supports streamfunction')
        if not np.isfinite(c.loss.stream_scale) or c.loss.stream_scale<=0:
            raise ValueError('Streamfunction output scale must be positive finite')
    if 'stream_lifting' in c:
        if (set(c.stream_lifting)!={'kind'} or c.stream_lifting.kind not in ('normalized_inlet','normalized_height')
                or 'velocity' not in c or 'lifting' not in c or c.lifting.kind!='normalized'):
            raise ValueError('Stream lifting requires explicit normalized_inlet/normalized_height, streamfunction and normalized lifting')
    if 'balance' in c:
        b=c.balance
        if b.kind!='gradnorm' or b.norm!='l1':raise ValueError('Explicit balance requires gradnorm/L1')
        if 'grouping' in b and b.grouping not in ('physics','terms'):
            raise ValueError('Balance grouping must explicitly select physics or terms')
        if not 0<b.epsilon<1 or not np.isfinite(b.cap) or b.cap<1:
            raise ValueError('Balance requires finite epsilon in(0,1) and cap>=1')
        if not 0<b.new_weight_fraction<=1:raise ValueError('Balance EMA fraction must be in(0,1]')
        if isinstance(b.update_every,bool) or not isinstance(b.update_every,int) or b.update_every<1:
            raise ValueError('Balance update_every must be a positive integer')
    if 'boundary_sampling' in c:
        b=c.boundary_sampling
        if (set(b)!={'kind','points_per_side','encoder'} or b.kind!='gauss_legendre'
                or b.encoder!='unchanged_midpoints' or 'derivatives' not in c):
            raise ValueError('Boundary sampling requires explicit gauss_legendre/unchanged_midpoints and GMLS derivatives')
        if isinstance(b.points_per_side,bool) or not isinstance(b.points_per_side,int) or b.points_per_side<2:
            raise ValueError('Boundary points_per_side must be an integer>=2')
    if 'phase_metric' in c:
        m=c.phase_metric
        if m.kind!='balanced_regions' or tuple(m.fields)!=('phase','chemical') or m.stop_weights is not True:
            raise ValueError('Explicit phase metric requires balanced_regions, phase/chemical and stopped weights')
        if not 0<m.region_fraction<1 or not 0<m.minimum_fraction<1:
            raise ValueError('Region budget and minimum fraction must both lie strictly in(0,1)')
    if 'droplet_metric' in c:
        m=c.droplet_metric
        if (set(m)!={'kind','contrast','temperature','fields','normalization','source'}
                or m.kind!='frozen_soft_phase' or m.source!='parent_model'
                or m.normalization!='interior_mean'
                or tuple(m.fields) not in (('phase','chemical'),('phase','chemical','momentum_x','momentum_y'))):
            raise ValueError('Droplet metric requires explicit frozen_soft_phase/parent_model/interior_mean and declared PDE fields')
        if not np.isfinite(m.contrast) or m.contrast<0 or not np.isfinite(m.temperature) or m.temperature<=0:
            raise ValueError('Droplet contrast must be nonnegative finite and temperature positive finite')
        if ('refinement' not in c.train or 'velocity' not in c
                or 'time_discretization' not in c or 'phase_transport' not in c
                or any(k in c for k in ('balance','stagger','phase_metric','residual_transform','phase_coordinate','stress_formulation','pressure_coordinate'))):
            raise ValueError('Frozen droplet metric requires original-variable single-case BDF/SBP full-time refinement')
    if 'phase_coordinate' in c:
        s=c.phase_coordinate
        if s.kind!='scaled_tanh' or not np.isfinite(s.amplitude) or s.amplitude<=1:
            raise ValueError('Scaled tanh requires explicit finite amplitude>1 for the saturated IC')
    if 'chemical_constraint'in c:
        s=c.chemical_constraint
        if (set(s)!={'kind','boundary','query'}or s.kind!='interior_gmls_projection'
                or s.boundary!='decoder_trace'or s.query!='mapped_bilinear'):
            raise ValueError('Chemical constraint requires interior_gmls_projection/decoder_trace/mapped_bilinear')
        if ('velocity'not in c or 'derivatives'not in c or 'time_derivative'not in c
                or 'phase_transport'not in c or c.phase_transport.kind!='gmls_conservative'
                or any(k in c for k in ('phase_coordinate','residual_transform','stress_formulation','pressure_coordinate','balance','stagger','droplet_metric','phase_metric'))
                or any(k in c.train for k in ('continuation','refinement'))):
            raise ValueError('Interior chemical projection requires original-variable GMLS/JVP joint losses and retained traces')
    if 'initialization'in c.train:
        from examples.channel_nsch2d.initialization import model_initialization_spec
        model_initialization_spec(c)
    if 'pressure_coordinate' in c:
        s=c.pressure_coordinate
        if set(s)!={'kind'} or s.kind!='chemical_shift' or 'velocity' not in c:
            raise ValueError('Pressure coordinate requires explicit chemical_shift and streamfunction representation')
    if 'stress_formulation' in c:
        s=c.stress_formulation
        if (set(s)!={'kind','initialization','constitutive_scale'} or s.kind!='cauchy_mixed'
                or s.initialization!='preserve_original' or 'velocity' not in c
                or not np.isfinite(s.constitutive_scale) or s.constitutive_scale<=0):
            raise ValueError('Stress formulation requires cauchy_mixed, preserve_original and positive constitutive_scale')
        if any(k in c for k in ('time_embedding','pressure_coordinate','phase_coordinate','residual_transform','phase_metric','balance','stagger')):
            raise ValueError('First stress controls require original phase/time coordinates and uniform joint losses')
    if 'time_embedding' in c:
        s=c.time_embedding
        if set(s) not in ({'kind','scale'},{'kind','scale','initialization'}) or s.kind!='linear_log' or not np.isfinite(s.scale) or s.scale<=0:
            raise ValueError('Time embedding requires linear_log and an explicit positive finite scale')
        if 'initialization' in s and s.initialization!='preserve_linear':
            raise ValueError('Time embedding initialization must explicitly preserve_linear')
        if c.arch.time_features<4 or c.arch.time_features%2 or c.arch.time_max_frequency is None:
            raise ValueError('Dual time embedding requires an even number>=4 and explicit maximum frequency')
    if 'residual_transform' in c:
        s=c.residual_transform
        if 'phase_coordinate' not in c or s.kind!='phase_inverse_map_jacobian':
            raise ValueError('Inverse-map residual requires an explicit scaled-tanh phase coordinate')
        if not np.isfinite(s.minimum_jacobian) or not 0<s.minimum_jacobian<c.phase_coordinate.amplitude:
            raise ValueError('Minimum Jacobian must lie strictly between zero and map amplitude')
        if any(key in c for key in ('balance','stagger','phase_metric')):
            raise ValueError('First inverse-map controls require joint uniform losses without other metrics')
    if 'time_discretization' in c:
        s=c.time_discretization
        if s.kind!='bdf2' or not np.isfinite(s.dt) or s.dt<=0:
            raise ValueError('Discrete time requires explicit bdf2 and positive finite dt')
        steps=p.terminal_time/s.dt
        if abs(steps-round(steps))>1e-10 or round(steps)<2 or round(steps)%c.sample.Nt:
            raise ValueError('Uniform discrete time must divide T and equal integer sampling strata')
        if c.sample.include_initial_time:
            raise ValueError('Discrete time uses positive intervals and the hard initial state, not a t0 PDE row')
        if any(key in c for key in ('residual_transform','phase_metric','stagger')) or ('balance' in c and 'derivatives' not in c):
            raise ValueError('First discrete-time controls require uniform original physical residuals')
    if 'time_derivative' in c:
        s=c.time_derivative
        if (set(s)!={'kind','sampling'} or s.kind!='jvp' or s.sampling!='continuous_stratified'
                or 'time_discretization' in c or 'derivatives' not in c
                or c.sample.include_initial_time):
            raise ValueError('Explicit time derivative requires jvp/continuous_stratified, GMLS space and no discrete-time/t0 row')
    if 'phase_transport' in c:
        if c.phase_transport.kind not in ('sbp4_mapped_curl','gmls_conservative','integral_conservative') or 'velocity' not in c:
            raise ValueError('Explicit SBP/GMLS phase transport requires the streamfunction representation')
        if c.phase_transport.kind in ('gmls_conservative','integral_conservative') and ('derivatives' not in c
                or set(c.phase_transport)!={'kind','quadrature'} or c.phase_transport.quadrature!='sbp4_mapped'):
            raise ValueError('GMLS conservative transport requires GMLS gradients and explicit sbp4_mapped loss quadrature')
        if min(c.sample.Nx,c.sample.Ny)<8:
            raise ValueError('Mapped SBP4-2 transport requires at least eight intervals in both axes')
        if any(key in c for key in ('phase_metric','stagger','residual_transform')) or ('balance' in c and 'derivatives' not in c):
            raise ValueError('First SBP transport controls require uniform original residual metrics')
    if 'derivatives' in c:
        s=c.derivatives
        expected=('integral_first','integral','integral_conservative')if integral else ('gmls_first','gmls','gmls_conservative')
        if (set(s)!={'kind','boundary','velocity'} or s.kind!=expected[0] or s.boundary!=expected[1]
                or s.velocity!='unchanged_streamfunction_ad' or 'velocity' not in c
                or not any(k in c for k in ('time_discretization','time_derivative')) or 'phase_transport' not in c
                or c.phase_transport.kind!=expected[2]):
            raise ValueError('GMLS-first residual control requires explicit boundary/unchanged_streamfunction_ad, BDF2 or explicit timeJVP, and GMLS flux')
        if any(k in c for k in ('stagger','stress_formulation','phase_metric','droplet_metric','residual_transform','phase_coordinate')):
            raise ValueError('GMLS residual control requires original physical outputs and joint losses')
    if 'refinement' in c.train:
        s=c.train.refinement
        required={'parent_run','optimizer','iterations','time_batch','history_size',
                  'max_evaluations','max_line_search','gradient_tolerance','function_tolerance'}
        if 'stress_initializer' in s:
            from examples.channel_nsch2d.stress_initialization import initializer_paths
            if 'stress_formulation' not in c:raise ValueError('Auxiliary initializer requires mixed stress outputs')
            initializer_paths(s.stress_initializer)
            required.add('stress_initializer')
        if set(s)!=required or s.optimizer!='lbfgs' or 'time_discretization' not in c:
            raise ValueError('Refinement requires explicit L-BFGS settings and BDF2 time discretization')
        if ('continuation' in c.train or 'online' in c.operator
                or c.operator.train_conditions!=1 or c.operator.condition_batch!=1
                or any(k in c for k in ('balance','stagger','phase_metric','residual_transform','phase_coordinate','pressure_coordinate'))):
            raise ValueError('First full-time refinement requires one uniform finite geometry and its own parent')
        for key in ('iterations','time_batch','history_size','max_evaluations','max_line_search'):
            if isinstance(s[key],bool) or not isinstance(s[key],int) or s[key]<1:
                raise ValueError(f'Positive integer refinement.{key} required')
        for key in ('gradient_tolerance','function_tolerance'):
            if not np.isfinite(s[key]) or s[key]<=0:raise ValueError(f'Positive finite refinement.{key} required')
        if int(round(p.terminal_time/c.time_discretization.dt))%s.time_batch:
            raise ValueError('Refinement time_batch must divide the full BDF node count')
    if 'stagger' in c:
        s=c.stagger
        if ('velocity' not in c or 'balance' in c or s.kind!='phase_flow'
                or s.coupling!='lagged' or s.inactive_head!='mask_updates' or s.optimizer_state!='shared'):
            raise ValueError('Stagger requires streamfunction, lagged coupling, masked heads/shared optimizer and no GradNorm')
        for name in ('switch_every','alternating_steps'):
            if isinstance(s[name],bool) or not isinstance(s[name],int) or s[name]<1:
                raise ValueError(f'Stagger {name} must be a positive integer')
        if s.alternating_steps>=c.train.iterations or s.alternating_steps%(2*s.switch_every):
            raise ValueError('Stagger requires complete phase/flow cycles and a nonempty joint tail')
    r=np.sqrt(c.initial.radius**2+2*c.initial.radius*c.initial.transition)
    x,y=c.initial.center
    if not (r<x<g.center-g.half_length-r and r<y<g.height-r):
        raise ValueError('Whole compact initial droplet must lie in the common straight inlet region')
    for dt in (c.reference.dt,c.reference.refined_dt):
        if not np.isfinite(dt) or dt<=0:raise ValueError('Reference dt must be positive finite')
        steps=p.terminal_time/dt
        if abs(steps-round(steps))>1e-10 or round(steps)%(c.reference.snapshots-1):
            raise ValueError('Reference steps must align exactly with snapshots')


def quadrature(amplitudes,c,nx,ny):
    """Mapped tensor nodes and trapezoid area measures, including actual walls."""
    xx=np.linspace(0,c.geometry.length,nx+1);ee=np.linspace(0,1,ny+1)
    x,e=np.meshgrid(xx,ee,indexing='ij');b,t=walls(x,amplitudes,c,xp=np)
    points=np.stack((x,b+e*(t-b)),axis=-1).reshape(-1,2)
    wx=np.ones(nx+1);wx[[0,-1]]=.5
    wy=np.ones(ny+1);wy[[0,-1]]=.5
    weights=(wx[:,None]*wy[None,:]*(t-b)*c.geometry.length/nx/ny).ravel()
    return points,weights


def boundary(amplitudes,c):
    n=c.sample.boundary_points
    x=(np.arange(n)+.5)*c.geometry.length/n
    b,t=walls(x,amplitudes,c,xp=np);db,dt=wall_slopes(x,amplitudes,c,xp=np)
    nb=np.stack((db,-np.ones(n)),axis=-1);nt=np.stack((-dt,np.ones(n)),axis=-1)
    wb=np.linalg.norm(nb,axis=-1)*c.geometry.length/n
    wt=np.linalg.norm(nt,axis=-1)*c.geometry.length/n
    nb/=np.linalg.norm(nb,axis=-1)[:,None];nt/=np.linalg.norm(nt,axis=-1)[:,None]
    eta=(np.arange(n)+.5)/n;y=eta*c.geometry.height
    rows=[np.column_stack((x,b,nb,np.zeros(n))),np.column_stack((x,t,nt,np.zeros(n))),
        np.column_stack((np.zeros(n),y,-np.ones(n),np.zeros(n),np.ones(n))),
        np.column_stack((np.full(n,c.geometry.length),y,np.ones(n),np.zeros(n),np.full(n,2.)))]
    return np.concatenate(rows),np.concatenate((wb,wt,np.full(2*n,c.geometry.height/n)))


def boundary_loss_quadrature(amplitudes,c):
    """Open Gauss-Legendre side queries; physical ds, no corner-normal ambiguity.

    Walls are graph curves, ds=sqrt(1+y'(x)^2) dx. The straight inlet/outlet
    use ds=H d(eta). Encoder quadrature remains the original midpoint rule.
    """
    spec=c.boundary_sampling
    if spec.kind!='gauss_legendre':raise ValueError('Explicit Gauss-Legendre boundary rule required')
    roots,weight=np.polynomial.legendre.leggauss(spec.points_per_side)
    eta=(roots+1)/2;weight=weight/2;n=len(eta)
    x=eta*c.geometry.length;y=eta*c.geometry.height
    bottom,top=walls(x,amplitudes,c,xp=np)
    db,dt=wall_slopes(x,amplitudes,c,xp=np)
    nb=np.column_stack((db,-np.ones(n)));nt=np.column_stack((-dt,np.ones(n)))
    lb=np.linalg.norm(nb,axis=1);lt=np.linalg.norm(nt,axis=1)
    rows=[np.column_stack((x,bottom,nb/lb[:,None],np.zeros(n))),
          np.column_stack((x,top,nt/lt[:,None],np.zeros(n))),
          np.column_stack((np.zeros(n),y,-np.ones(n),np.zeros(n),np.ones(n))),
          np.column_stack((np.full(n,c.geometry.length),y,np.ones(n),np.zeros(n),np.full(n,2.)))]
    measures=np.concatenate((lb*weight*c.geometry.length,lt*weight*c.geometry.length,
                             np.tile(weight*c.geometry.height,2)))
    return np.concatenate(rows),measures


def boundary_payload_shapes(c):
    """Strict optional schema: dense BC queries and unchanged encoder stream."""
    if 'boundary_sampling' not in c:return {}
    b=4*c.sample.boundary_points
    return dict(encoder_boundary=(b,5),encoder_boundary_weights=(b,))
