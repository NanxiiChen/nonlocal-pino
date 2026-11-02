"""Full momentum/continuity/CH/chemical residuals plus explicit wall/outlet BCs."""
import jax
import jax.numpy as jnp
from examples.channel_nsch2d.network import encode,field_jets,time_jets,phase_jacobian,spatial_field_jets,fields,potentials,stress_jets,stresses
from examples.channel_nsch2d.phase_coordinate import transformed_transport
from examples.channel_nsch2d.temporal import backward_rate,previous_times
from examples.channel_nsch2d.transport import conservative_advection
from examples.channel_nsch2d.problem import material


PARTS=('phase','chemical','momentum_x','momentum_y','continuity',
       'wall_phase','wall_mu','outlet_traction','outlet_phase','outlet_mu')


def part_names(c):
    names=tuple(p for p in PARTS if p!='continuity') if 'velocity' in c else PARTS
    if 'chemical_constraint'in c:names=tuple(p for p in names if p!='chemical')
    if 'stress_formulation' in c:names+=('stress_normal','stress_shear','boundary_stress_normal','boundary_stress_shear')
    return names


def residual_components(value,dt,dx,dy,lap,c,active=None,phase_advection=None,stress_divergence=None):
    p=c.physical;A,kappa=material(c)
    phi,mu,ux,uy=[value[...,i] for i in range(4)]
    adv=lambda i:ux*dx[...,i]+uy*dy[...,i]
    transport=adv(0) if phase_advection is None else phase_advection
    force_x=phi*dx[...,1];force_y=phi*dy[...,1]
    if active is not None:
        # Preserve every primal residual while lagging cross-physics factors.
        gate=lambda a,g:jax.lax.stop_gradient(a)+g*(a-jax.lax.stop_gradient(a))
        transport=gate(ux,active[1])*dx[...,0]+gate(uy,active[1])*dy[...,0]
        force_x=gate(phi,active[0])*gate(dx[...,1],active[0])
        force_y=gate(phi,active[0])*gate(dy[...,1],active[0])
    momentum_x=p.density*(dt[...,2]+adv(2))+dx[...,4]-p.viscosity*lap[...,2]+force_x
    momentum_y=p.density*(dt[...,3]+adv(3))+dy[...,4]-p.viscosity*lap[...,3]+force_y
    if stress_divergence is not None:
        momentum_x=p.density*(dt[...,2]+adv(2))-stress_divergence[...,0]+force_x
        momentum_y=p.density*(dt[...,3]+adv(3))-stress_divergence[...,1]+force_y
    return jnp.stack((dt[...,0]+transport-p.mobility*lap[...,1],
        mu-A*(phi**3-phi)+kappa*lap[...,0],
        momentum_x,momentum_y,
        dx[...,2]+dy[...,3]),axis=-1)


def gmls_first_jets(value,center,data,*,boundary):
    """Differentiate the unchanged physical field using precomputed weights."""
    index=data['boundary_neighbors'] if boundary else data['neighbors']
    gx=data['boundary_gradient_x'] if boundary else data['gradient_x']
    gy=data['boundary_gradient_y'] if boundary else data['gradient_y']
    if (value.ndim!=3 or center.ndim!=3 or value.shape[0]!=center.shape[0]
            or value.shape[-1]!=center.shape[-1] or index.ndim!=2
            or gx.shape!=index.shape or gy.shape!=index.shape or center.shape[1]!=index.shape[0]):
        raise ValueError('GMLS gradients require aligned(time,points,channels) values and target/weight shapes')
    adjacent=value[:,index,:]-center[:,:,None,:]
    return (jnp.sum(gx[None,:,:,None]*adjacent,axis=2),
        jnp.sum(gy[None,:,:,None]*adjacent,axis=2))


def gmls_phase_advection(value,data):
    """GMLS divergence of phi*v; conservative form, no SBP mass identity."""
    flux=value[...,:1]*value[...,2:4]
    dx,dy=gmls_first_jets(flux,flux,data,boundary=False)
    return dx[...,0]+dy[...,1]


def _make_condition_loss(c,*,per_time):
    p=c.physical;l=c.loss
    chemical_scale=l.chemical_residual_scale if 'chemical_residual_scale'in l else l.chemical_scale
    projected='chemical_constraint'in c
    if projected:
        from examples.channel_nsch2d.projected_fields import make_domain_time_fn,make_decoder_fn
        domain_time=make_domain_time_fn(c);boundary_value=make_decoder_fn(c)
    def one(net,data,times,active):
        encoded=encode(net,data)
        if projected:
            value,dt=domain_time(net,encoded,data,times)
            dx,dy=gmls_first_jets(value,value,data,boundary=False)
        elif 'time_discretization' in c:
            if 'derivatives' in c:
                value=fields(net,encoded,data['points'],times,data['amplitudes'],c)
                dx,dy=gmls_first_jets(value,value,data,boundary=False)
            else:
                value,dx,dy=spatial_field_jets(net,encoded,data['points'],times,data['amplitudes'],c)
            t1,t2=previous_times(times,c.time_discretization.dt)
            history=fields(net,encoded,data['points'],jnp.concatenate((t1,t2)),data['amplitudes'],c)
            dt=backward_rate(value,history[:len(times)],history[len(times):],times,c.time_discretization.dt)
        elif 'time_derivative' in c:
            value,dt=time_jets(net,encoded,data['points'],times,data['amplitudes'],c)
            dx,dy=gmls_first_jets(value,value,data,boundary=False)
        else:
            value,dt,dx,dy=field_jets(net,encoded,data['points'],times,data['amplitudes'],c)
        neighbor=value[:,data['neighbors'],:]-value[:,:,None,:]
        lap=jnp.sum(data['laplacian'][None,:,:,None]*neighbor,axis=2)
        scales=jnp.array([l.phase_scale,chemical_scale,l.momentum_scale,l.momentum_scale,l.continuity_scale])
        phase_advection=None
        if 'phase_transport' in c:
            if c.phase_transport.kind=='gmls_conservative':
                phase_advection=gmls_phase_advection(value,data)
            else:
                stream=potentials(net,encoded,data['points'],times,data['amplitudes'],c)[...,2]
                phase_advection,_,_=conservative_advection(value[...,0],stream,data,xp=jnp)
        stress_div=None
        if 'stress_formulation' in c:
            from examples.channel_nsch2d.stress import constitutive,divergence
            sigma,sx,sy=stress_jets(net,encoded,data['points'],times,c)
            stress_div=divergence(sx,sy,xp=jnp)
        residual=residual_components(value,dt,dx,dy,lap,c,active,phase_advection,stress_div)/scales
        if 'residual_transform' in c:
            jacobian=phase_jacobian(net,encoded,data['points'],times,c)
            phase=transformed_transport(residual[...,0],jacobian,
                c.phase_coordinate.amplitude,c.residual_transform.minimum_jacobian)
            residual=jnp.concatenate((phase[...,None],residual[...,1:]),axis=-1)
        if projected:residual=residual[...,jnp.asarray([0,2,3])]
        elif 'velocity' in c:residual=residual[...,:4]  # incompressibility is exact; no redundant loss
        weight=data['volumes']*data['interior']
        if 'phase_transport' in c:weight=data['transport_measure']*data['interior']
        square=residual**2
        if 'phase_metric' in c:
            from examples.channel_nsch2d.phase_metric import region_weights
            metric=region_weights(value[...,0],weight,c.phase_metric)
            square=jnp.concatenate((square[...,:2]*metric[...,None],square[...,2:]),axis=-1)
        if 'droplet_metric' in c:
            import equinox as eqx
            indices=jnp.rint(times/c.time_discretization.dt).astype(jnp.int32)-1
            indices=eqx.error_if(indices,jnp.any((indices<0)|(indices>=data['droplet_weights'].shape[0])),
                'Frozen droplet weights require actual positive BDF time nodes')
            metric=data['droplet_weights'][indices]
            selected=jnp.asarray([name in c.droplet_metric.fields for name in part_names(c)[:square.shape[-1]]])
            square=square*jnp.where(selected[None,None,:],metric[...,None],1.)
        axes=1 if per_time else (0,1)
        count=1 if per_time else len(times)
        parts=jnp.sum(square*weight[None,:,None],axis=axes)/(jnp.sum(weight)*count)
        b=data['boundary'];point=b[:,:2];normal=b[:,2:4];kind=b[:,4]
        if 'derivatives' in c:
            bv=(boundary_value(net,encoded,point,times,data['amplitudes'])if projected else
                fields(net,encoded,point,times,data['amplitudes'],c))
            bx,by=gmls_first_jets(value,bv,data,boundary=True)
        elif 'time_discretization' in c:
            bv,bx,by=spatial_field_jets(net,encoded,point,times,data['amplitudes'],c)
        else:
            bv,_,bx,by=field_jets(net,encoded,point,times,data['amplitudes'],c)
        normal_derivative=bx*normal[None,:,0,None]+by*normal[None,:,1,None]
        if per_time:
            average=lambda quantity,mask:jnp.sum(quantity*mask[None,:]*data['boundary_weights'][None,:],axis=1)/jnp.sum(mask*data['boundary_weights'])
        else:
            average=lambda quantity,mask:jnp.sum(quantity*mask[None,:]*data['boundary_weights'][None,:])/(len(times)*jnp.sum(mask*data['boundary_weights']))
        wall=kind==0;outlet=kind==2
        phase_wall=average((normal_derivative[...,0]/l.boundary_phi_scale)**2,wall)
        mu_wall=average((normal_derivative[...,1]/l.boundary_mu_scale)**2,wall)
        # Hydrodynamic symmetric Newtonian stress, exterior reference pressure0.
        traction_x=-bv[...,4]+2*p.viscosity*bx[...,2]
        traction_y=p.viscosity*(by[...,2]+bx[...,3])
        traction=average((traction_x**2+traction_y**2)/l.traction_scale**2,outlet)
        phase_out=average((normal_derivative[...,0]/l.boundary_phi_scale)**2,outlet)
        mu_out=average((normal_derivative[...,1]/l.boundary_mu_scale)**2,outlet)
        result=jnp.concatenate((parts,jnp.stack((phase_wall,mu_wall,traction,phase_out,mu_out),axis=-1)),axis=-1)
        if 'stress_formulation' in c:
            scale=c.stress_formulation.constitutive_scale
            interior_constitutive=constitutive(sigma,dx,dy,p.viscosity,xp=jnp)/scale
            bc_constitutive=constitutive(stresses(net,encoded,point,times,c),bx,by,p.viscosity,xp=jnp)/scale
            inside=jnp.sum(interior_constitutive**2*weight[None,:,None],axis=axes)/(count*jnp.sum(weight))
            boundary_const=jnp.sum(bc_constitutive**2*data['boundary_weights'][None,:,None],axis=axes)/(count*jnp.sum(data['boundary_weights']))
            result=jnp.concatenate((result,inside,boundary_const),axis=-1)
        return result
    return one


def make_time_parts_fn(c):
    """Unweighted (condition,time,part) MSE; do not mix geometry prefixes."""
    one=_make_condition_loss(c,per_time=True)
    return lambda net,batch,times:jax.vmap(lambda d:one(net,d,times,None))(batch)


def make_loss_fn(c):
    one=_make_condition_loss(c,per_time=False)

    def loss(net,batch,times,active=None):
        parts=jnp.mean(jax.vmap(lambda d:one(net,d,times,active))(batch),axis=0)
        if active is None:return jnp.sum(parts),parts
        if 'stagger' not in c or 'velocity' not in c:
            raise ValueError('Active block masks require an explicit streamfunction stagger config')
        # Stream parts: phase,chemical,mx,my,wall_phi,wall_mu,traction,out_phi,out_mu.
        phase=jnp.asarray([1,1,0,0,1,1,0,1,1],dtype=parts.dtype)
        return jnp.dot(parts,phase*active[0]+(1-phase)*active[1]),parts
    return loss
