"""Two-stream geometry conditioning and shared spatial/time differentiation."""
import equinox as eqx
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict
from nonlocal_pinn.nn.point_conditional import HeterogeneousPointOperator,_time_frequencies
from examples.channel_nsch2d.problem import walls,initial_field,flow_rate
from examples.channel_nsch2d.phase_coordinate import mapped_phase,map_jacobian


def _build_network(c,key):
    net=HeterogeneousPointOperator(input_dims=(5,3),extent=(c.geometry.height,)*2,
        **c.arch.to_dict(),key=key)
    # The common point module has a scalar-output convenience decoder. Retain
    # its spatial encoder, replace only the output layer functionally, and use
    # the explicit multichannel decoder below. Existing examples are untouched.
    channels=(6 if 'stress_formulation' in c else 4) if 'velocity' in c else 5
    output=eqx.nn.Linear(c.arch.width,channels,key=jax.random.fold_in(key,741))
    net=eqx.tree_at(lambda n:n.output,net,output)
    return jax.tree.map(lambda a:a.astype(c.train.compute_dtype)
                       if eqx.is_inexact_array(a) else a,net)


def extend_time_basis(parent,template):
    """Preserve every old linear Fourier column; initialize new log columns0."""
    old=parent.time_features
    if template.time_features!=2*old or template.width!=parent.width:
        raise ValueError('Time expansion requires twice the Fourier width and unchanged model width')
    previous=[(jax.tree_util.keystr(p),a) for p,a in jax.tree_util.tree_flatten_with_path(parent)[0] if eqx.is_array(a)]
    current=[(jax.tree_util.keystr(p),a) for p,a in jax.tree_util.tree_flatten_with_path(template)[0] if eqx.is_array(a)]
    if [p for p,_ in previous]!=[p for p,_ in current]:raise ValueError('Time expansion parameter paths differ')
    replacement=[];expanded=0
    for (path,a),(_,b) in zip(previous,current,strict=True):
        if path=='.time.layers[0].weight':
            if a.shape!=(b.shape[0],1+2*old) or b.shape[1]!=1+4*old or a.dtype!=b.dtype:
                raise ValueError('Unexpected time modulation matrix shape/dtype')
            value=jnp.zeros_like(b).at[:,0].set(a[:,0])
            value=value.at[:,1:1+old].set(a[:,1:1+old])
            value=value.at[:,1+2*old:1+3*old].set(a[:,1+old:1+2*old])
            replacement.append(value);expanded+=1
        else:
            if a.shape!=b.shape or a.dtype!=b.dtype:raise ValueError(f'Unexpected expansion change: {path}')
            replacement.append(a)
    if expanded!=1:raise ValueError('Expected exactly one time input matrix')
    positions=tuple(i for i,a in enumerate(jax.tree.leaves(template)) if eqx.is_array(a))
    def select_arrays(n):
        leaves=jax.tree.leaves(n)
        return tuple(leaves[i] for i in positions)
    return eqx.tree_at(select_arrays,template,tuple(replacement))


def build_network(c,key):
    net=_build_network(c,key)
    if 'time_embedding' in c and 'initialization' in c.time_embedding:
        if c.time_embedding.initialization!='preserve_linear':raise ValueError('Unsupported explicit time initialization')
        baseline=ConfigDict(c.to_dict());del baseline.time_embedding
        baseline.arch.time_features=c.arch.time_features//2
        net=extend_time_basis(_build_network(baseline,key),net)
    if 'stress_formulation' in c:
        baseline=ConfigDict(c.to_dict());del baseline.stress_formulation
        net=extend_stress_outputs(_build_network(baseline,key),net)
    return net


def extend_stress_outputs(parent,template):
    """Preserve phi/mu/psi/p; initialize isotropic sigma=-p*I, shear0."""
    if (parent.output.weight.shape!=(4,template.width)
            or template.output.weight.shape!=(6,parent.width)
            or parent.width!=template.width):
        raise ValueError('Stress transfer requires matching-width4->6output heads')
    old=[(jax.tree_util.keystr(p),a) for p,a in jax.tree_util.tree_flatten_with_path(parent)[0] if eqx.is_array(a)]
    new=[(jax.tree_util.keystr(p),a) for p,a in jax.tree_util.tree_flatten_with_path(template)[0] if eqx.is_array(a)]
    if [p for p,_ in old]!=[p for p,_ in new]:raise ValueError('Stress transfer model paths differ')
    for (path,a),(_,b) in zip(old,new,strict=True):
        if a.dtype!=b.dtype or (path not in ('.output.weight','.output.bias') and a.shape!=b.shape):
            raise ValueError(f'Unsupported stress transfer parameter change:{path}')
    weight=jnp.zeros_like(template.output.weight).at[:3].set(parent.output.weight[:3])
    weight=weight.at[3].set(-parent.output.weight[3]).at[5].set(-parent.output.weight[3])
    bias=jnp.zeros_like(template.output.bias).at[:3].set(parent.output.bias[:3])
    bias=bias.at[3].set(-parent.output.bias[3]).at[5].set(-parent.output.bias[3])
    output=eqx.tree_at(lambda layer:(layer.weight,layer.bias),template.output,(weight,bias))
    return eqx.tree_at(lambda n:n.output,parent,output)


def encode(net,data):
    boundary=data['encoder_boundary'] if 'encoder_boundary' in data else data['boundary']
    weights=data['encoder_boundary_weights'] if 'encoder_boundary' in data else data['boundary_weights']
    return net.encode((boundary,data['domain']),(weights,data['domain_weights']))


def spatial_jets(net,encoded,points):
    prepare=lambda x:net.prepare_encoded(encoded,x)
    ex=jnp.broadcast_to(jnp.array([1.,0.],dtype=points.dtype),points.shape)
    ey=jnp.broadcast_to(jnp.array([0.,1.],dtype=points.dtype),points.shape)
    h,hx=jax.jvp(prepare,(points,),(ex,));_,hy=jax.jvp(prepare,(points,),(ey,))
    return h,hx,hy


def temporal_features(net,times,c):
    """Legacy linear basis, or explicit equal-width linear/log time features."""
    t=times/c.physical.terminal_time
    if 'time_embedding' in c:
        spec=c.time_embedding
        logarithmic=jnp.log1p(times/spec.scale)/jnp.log1p(c.physical.terminal_time/spec.scale)
        frequency=_time_frequencies(net.time_features//2,net.time_max_frequency,t.dtype)
        angle=jnp.concatenate((jnp.pi*t[:,None]*frequency,
            jnp.pi*logarithmic[:,None]*frequency),axis=-1)
    else:
        frequency=_time_frequencies(net.time_features,net.time_max_frequency,t.dtype)
        angle=jnp.pi*t[:,None]*frequency
    return jnp.concatenate((t[:,None],jnp.sin(angle),jnp.cos(angle)),axis=-1)


def decode(net,h,times,c):
    temporal=temporal_features(net,times,c)
    modulation=jax.vmap(net.time)(temporal).reshape(len(times),len(net.layers),2,net.width)
    hidden=jnp.broadcast_to(h,(len(times),*h.shape))
    for i,layer in enumerate(net.layers):
        hidden=jax.nn.gelu(jax.vmap(jax.vmap(layer))(hidden)*(1+modulation[:,i,0,None,:])
                            +modulation[:,i,1,None,:])
    return net.output_scale*jax.vmap(jax.vmap(net.output))(hidden)


def lifting_gates(points,times,c):
    # The optional record is a versioned extension: its absence preserves the
    # first saved pilot's linear t/T,x/L arithmetic for checkpoint replay.
    if 'lifting' in c:
        if c.lifting.kind!='normalized':raise ValueError('Unsupported explicit lifting kind')
        xgate=-jnp.expm1(-points[:,0]/c.lifting.inlet_length_scale)
        time=-jnp.expm1(-times/c.lifting.time_scale)
    else:
        xgate=points[:,0]/c.geometry.length
        time=times/c.physical.terminal_time
    return xgate,time


def phase_jacobian(net,encoded,points,times,c):
    """Jacobian of the output coordinate, evaluated from its unsaturated latent."""
    raw=decode(net,net.prepare_encoded(encoded,points),times,c)[...,0]
    xgate,time=lifting_gates(points,times,c)
    _,latent=mapped_phase(initial_field(points,c,xp=jnp)[None,:],
        time[:,None]*xgate[None,:]*raw,c.phase_coordinate)
    return map_jacobian(latent,c.phase_coordinate.amplitude)


def lift(raw,points,times,amplitudes,c):
    """Generic channel Dirichlet/initial lifting; no evolved interface prior."""
    bottom,top=walls(points[:,0],amplitudes,c,xp=jnp)
    height=top-bottom;eta=(points[:,1]-bottom)/height
    xgate,time=lifting_gates(points,times,c)
    wall=4*eta*(1-eta)
    gate=time[:,None]*xgate[None,:]
    phi=initial_field(points,c,xp=jnp)[None,:]+gate*raw[...,0]
    if 'phase_coordinate' in c:
        phi,_=mapped_phase(initial_field(points,c,xp=jnp)[None,:],
            gate*raw[...,0],c.phase_coordinate)
    mu=c.loss.chemical_scale*xgate[None,:]*raw[...,1]
    if 'velocity' in c:
        if c.velocity.kind!='streamfunction':raise ValueError('Unsupported velocity representation')
        # Stream values0,Q and their zero normal derivative at both walls give
        # exact flux and no slip. x^2 preserves the full inlet velocity profile.
        shape=3*eta**2-2*eta**3
        if 'stream_lifting' in c:
            if c.stream_lifting.kind=='normalized_inlet':inlet=xgate**2
            elif c.stream_lifting.kind=='normalized_height':
                inlet=(-jnp.expm1(-points[:,0]/c.geometry.height))**2
            else:raise ValueError('Unsupported explicit stream lifting')
        else:inlet=(points[:,0]/c.geometry.length)**2
        correction=16*(eta*(1-eta))**2*inlet
        stream=flow_rate(times,c,xp=jnp)[:,None]*shape[None,:]
        stream+=c.loss.stream_scale*time[:,None]*correction[None,:]*raw[...,2]
        pressure=c.loss.pressure_scale*raw[...,3]
        if 'stress_formulation' in c:pressure=-.5*c.loss.pressure_scale*(raw[...,3]+raw[...,5])
        if 'pressure_coordinate' in c:
            if c.pressure_coordinate.kind!='chemical_shift':raise ValueError('Unsupported explicit pressure coordinate')
            pressure=pressure-phi*mu
        return jnp.stack((phi,mu,stream,pressure),axis=-1)
    inlet=6*flow_rate(times,c,xp=jnp)[:,None]*eta[None,:]*(1-eta[None,:])/height[None,:]
    # This lifting satisfies wall/inlet/initial velocities, not NS. Transverse
    # velocity, pressure, and all geometry-dependent corrections are learned.
    ux=inlet+c.physical.mean_velocity*gate*wall[None,:]*raw[...,2]
    uy=c.physical.mean_velocity*gate*wall[None,:]*raw[...,3]
    pressure=c.loss.pressure_scale*raw[...,4]
    return jnp.stack((phi,mu,ux,uy,pressure),axis=-1)


def potentials(net,encoded,points,times,amplitudes,c):
    if 'chemical_constraint'in c:
        raise ValueError('Projected mu requires projected_fields factories; point-decoded mu is only a boundary trace')
    h=net.prepare_encoded(encoded,points)
    return lift(decode(net,h,times,c),points,times,amplitudes,c)


def stresses(net,encoded,points,times,c):
    if 'stress_formulation' not in c or net.output.out_features!=6:
        raise ValueError('Stress queries require the explicit six-output stress formulation')
    raw=decode(net,net.prepare_encoded(encoded,points),times,c)
    return c.loss.pressure_scale*raw[...,3:6]


def stress_jets(net,encoded,points,times,c):
    fn=lambda x:stresses(net,encoded,x,times,c)
    ex=jnp.broadcast_to(jnp.array([1.,0.],dtype=points.dtype),points.shape)
    ey=jnp.broadcast_to(jnp.array([0.,1.],dtype=points.dtype),points.shape)
    value,dx=jax.jvp(fn,(points,),(ex,));_,dy=jax.jvp(fn,(points,),(ey,))
    return value,dx,dy


def fields(net,encoded,points,times,amplitudes,c):
    if 'chemical_constraint'in c:
        raise ValueError('Projected mu requires projected_fields factories, not raw coordinate fields')
    if 'velocity' not in c:return potentials(net,encoded,points,times,amplitudes,c)
    fn=lambda x:potentials(net,encoded,x,times,amplitudes,c)
    ex=jnp.broadcast_to(jnp.array([1.,0.],dtype=points.dtype),points.shape)
    ey=jnp.broadcast_to(jnp.array([0.,1.],dtype=points.dtype),points.shape)
    value,dx=jax.jvp(fn,(points,),(ex,));_,dy=jax.jvp(fn,(points,),(ey,))
    return jnp.stack((value[...,0],value[...,1],dy[...,2],-dx[...,2],value[...,3]),axis=-1)


def time_jets(net,encoded,points,times,amplitudes,c):
    """Physical fields and time JVP, with one shared spatial query set.

    Time points do not attend to one another, so the all-ones tangent gives
    each query's own derivative, including the physical streamfunction curl.
    No residual spatial AD derivatives are requested here.
    """
    return jax.jvp(lambda t:fields(net,encoded,points,t,amplitudes,c),
        (times,),(jnp.ones_like(times),))


def spatial_field_jets(net,encoded,points,times,amplitudes,c):
    """Current-time spatial derivatives for an explicitly discrete-time loss."""
    fn=lambda x:fields(net,encoded,x,times,amplitudes,c)
    ex=jnp.broadcast_to(jnp.array([1.,0.],dtype=points.dtype),points.shape)
    ey=jnp.broadcast_to(jnp.array([0.,1.],dtype=points.dtype),points.shape)
    value,dx=jax.jvp(fn,(points,),(ex,));_,dy=jax.jvp(fn,(points,),(ey,))
    return value,dx,dy


def field_jets(net,encoded,points,times,amplitudes,c):
    if 'velocity' in c:
        fn=lambda x,t:fields(net,encoded,x,t,amplitudes,c)
        ex=jnp.broadcast_to(jnp.array([1.,0.],dtype=points.dtype),points.shape)
        ey=jnp.broadcast_to(jnp.array([0.,1.],dtype=points.dtype),points.shape)
        value,dt=jax.jvp(lambda t:fn(points,t),(times,),(jnp.ones_like(times),))
        _,dx=jax.jvp(lambda x:fn(x,times),(points,),(ex,))
        _,dy=jax.jvp(lambda x:fn(x,times),(points,),(ey,))
        return value,dt,dx,dy
    h,hx,hy=spatial_jets(net,encoded,points)
    physical=lambda z,x,t:lift(decode(net,z,t,c),x,t,amplitudes,c)
    value,dt=jax.jvp(lambda t:physical(h,points,t),(times,),(jnp.ones_like(times),))
    ex=jnp.broadcast_to(jnp.array([1.,0.],dtype=points.dtype),points.shape)
    ey=jnp.broadcast_to(jnp.array([0.,1.],dtype=points.dtype),points.shape)
    _,dx=jax.jvp(lambda z,x:physical(z,x,times),(h,points),(hx,ex))
    _,dy=jax.jvp(lambda z,x:physical(z,x,times),(h,points),(hy,ey))
    return value,dt,dx,dy
