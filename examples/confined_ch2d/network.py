"""Nonperiodic coordinate network with only initial-phase and weighted mass constraints."""
import equinox as eqx
import jax
import jax.numpy as jnp
from nonlocal_pinn.nn.arch import make_pinn


class PhaseNetwork(eqx.Module):
    net: object
    center: tuple=eqx.field(static=True)
    axes: tuple=eqx.field(static=True)
    holes: tuple=eqx.field(static=True)
    amplitude: float=eqx.field(static=True)
    wall_width: float=eqx.field(static=True)
    duration: float=eqx.field(static=True)
    modes: tuple=eqx.field(static=True)
    initial_kind: str=eqx.field(static=True)
    initial_waves: tuple=eqx.field(static=True)
    initial_scale: float=eqx.field(static=True)
    lifting: str=eqx.field(static=True)
    initial_time_scale: float=eqx.field(static=True)
    chemical_mode: str=eqx.field(static=True)
    beta: float=eqx.field(static=True)
    epsilon_squared: float=eqx.field(static=True)
    initial_feature: bool=eqx.field(static=True)

    def __init__(self,c,key):
        self.center=tuple(c.geometry.center);self.axes=tuple(c.geometry.axes)
        self.holes=tuple(tuple(v) for v in c.geometry.holes)
        self.amplitude=c.initial.amplitude;self.wall_width=c.initial.wall_width
        self.duration=c.physical.terminal_time;self.modes=tuple(range(1,c.arch.spatial_modes+1))
        if 'kind' in c.initial:
            self.initial_kind=c.initial.kind;self.initial_waves=tuple(tuple(v) for v in c.initial.waves);self.initial_scale=c.initial.scale
        else:
            self.initial_kind='legacy';self.initial_waves=();self.initial_scale=1.
        if 'phase' in c:
            self.lifting=c.phase.lifting;self.initial_time_scale=c.phase.initial_time_scale
        else:
            # Historical saved configs use precisely the original arithmetic.
            self.lifting='additive';self.initial_time_scale=self.duration
        self.chemical_mode=c.formulation.chemical if 'formulation' in c else 'learned'
        self.beta=c.physical.beta;self.epsilon_squared=c.physical.epsilon**2
        self.initial_feature=c.arch.initial_feature if 'initial_feature' in c.arch else False
        self.net=make_pinn(2+4*len(self.modes)+int(self.initial_feature),1 if self.chemical_mode=='eliminated' else 2,hidden_dim=c.arch.hidden_dim,
            num_layers=c.arch.num_layers,activation=c.arch.activation,use_spatial_emb=False,
            use_temporal_emb=True,temporal_emb_dim=c.arch.temporal_emb_dim,
            temporal_scale=c.arch.temporal_scale,key=key)

    def inputs(self,x):
        """Shared coordinate features and prescribed initial field."""
        z=(x-jnp.asarray(self.center))/jnp.asarray(self.axes)
        angles=jnp.pi*z[:,None]*jnp.asarray(self.modes)
        features=jnp.concatenate((z,jnp.sin(angles).ravel(),jnp.cos(angles).ravel()))
        # Same prescribed IC formula as problem.initial_field, via static metadata.
        wall=.5*min(self.axes)*(1-jnp.sum(z*z));w=self.wall_width
        envelope=wall**4/(wall**4+w**4)
        for cx,cy,r in self.holes:
            wall=(jnp.sum((x-jnp.asarray((cx,cy)))**2)-r*r)/(2*r)
            envelope=envelope*wall**4/(wall**4+w**4)
        if self.initial_kind=='multimode':
            waves=jnp.asarray(self.initial_waves)
            angles=jnp.pi*jnp.sum(z[None,:]*waves[:,:2],axis=-1)+waves[:,3]
            p0=self.amplitude*envelope*jnp.tanh(jnp.sum(waves[:,2]*jnp.cos(angles))/self.initial_scale)
        else:
            p0=self.amplitude*envelope*(jnp.cos(2*jnp.pi*z[0])+.6*jnp.sin(1.5*jnp.pi*z[1]))
        if self.initial_feature:features=jnp.concatenate((features,jnp.atleast_1d(p0)))
        return features,p0

    def __call__(self,x,t):
        features,p0=self.inputs(x)
        normalized_time=t/self.duration
        if self.lifting in ('logarithmic_additive','logarithmic_blend','logarithmic_increment'):
            normalized_time=jnp.log1p(t/self.initial_time_scale)/jnp.log1p(self.duration/self.initial_time_scale)
        raw=self.net(features,normalized_time)
        if self.lifting in ('exponential_blend','logarithmic_blend'):
            elapsed=t/self.initial_time_scale
            phi=jnp.exp(-elapsed)*p0-jnp.expm1(-elapsed)*raw[0]
        elif self.lifting=='logarithmic_increment':phi=p0-jnp.expm1(-t/self.initial_time_scale)*raw[0]
        elif self.lifting=='logarithmic_additive':phi=p0+normalized_time*raw[0]
        else:phi=p0+(t/self.duration)*raw[0]
        if self.chemical_mode=='eliminated':return jnp.stack((phi,))
        return jnp.stack((phi,raw[1]))


class GlobalBasisNetwork(PhaseNetwork):
    """One spatial MLP with a globally supported continuous time expansion."""
    rank: int=eqx.field(static=True)

    def __init__(self,c,key):
        super().__init__(c,key);self.rank=c.arch.rank
        spatial=make_pinn(2+4*len(self.modes)+int(self.initial_feature),self.rank,
            hidden_dim=c.arch.hidden_dim,num_layers=c.arch.num_layers,
            activation=c.arch.activation,use_spatial_emb=False,use_temporal_emb=False,
            static=True,key=key)
        self.net=eqx.tree_at(lambda m:(m.output.weight,m.output.bias),spatial,
            (spatial.output.weight*c.arch.output_scale,spatial.output.bias*c.arch.output_scale))

    def __call__(self,x,t):
        features,p0=self.inputs(x);coefficients=self.net(features)
        s=jnp.log1p(t/self.initial_time_scale)/jnp.log1p(self.duration/self.initial_time_scale)
        u=2*s-1;basis=[jnp.ones_like(u)]
        if self.rank>1:basis.append(u)
        for k in range(2,self.rank):basis.append(2*u*basis[-1]-basis[-2])
        return jnp.stack((p0+s*jnp.dot(coefficients,jnp.stack(basis)),))


class TanhPhaseNetwork(GlobalBasisNetwork):
    """Global latent phase with physical mass projection outside the point call."""
    tanh_amplitude: float=eqx.field(static=True)

    def __init__(self,c,key):
        super().__init__(c,key);self.tanh_amplitude=c.phase.output_map.amplitude

    def latent(self,x,t):
        features,p0=self.inputs(x)
        p0=eqx.error_if(p0,jnp.abs(p0)>=self.tanh_amplitude,'Initial phase must lie strictly inside tanh amplitude')
        coefficients=self.net(features)
        s=jnp.log1p(t/self.initial_time_scale)/jnp.log1p(self.duration/self.initial_time_scale)
        u=2*s-1;basis=[jnp.ones_like(u)]
        if self.rank>1:basis.append(u)
        for k in range(2,self.rank):basis.append(2*u*basis[-1]-basis[-2])
        return jnp.arctanh(p0/self.tanh_amplitude)+s*jnp.dot(coefficients,jnp.stack(basis))

    def __call__(self,x,t):
        # Unprojected physical point value. Full-grid prediction always includes
        # the scalar offset through tanh_mapping.phase_state.
        from examples.confined_ch2d.tanh_mapping import stable_tanh
        return jnp.atleast_1d(self.tanh_amplitude*stable_tanh(self.latent(x,t)))


class GlobalMixedNetwork(PhaseNetwork):
    """Jointly learned phi/mu global time expansions; no temporal state marching."""
    chemical_net: object
    rank: int=eqx.field(static=True)
    field_layout: str=eqx.field(static=True)
    chemical_scale: float=eqx.field(static=True)

    def __init__(self,c,key):
        super().__init__(c,key)
        self.rank=c.arch.rank;self.field_layout=c.arch.field_layout
        self.chemical_scale=c.arch.chemical_scale
        phase_key,chemical_key=jax.random.split(key)
        def spatial(outputs,key):
            model=make_pinn(2+4*len(self.modes)+int(self.initial_feature),outputs,
                hidden_dim=c.arch.hidden_dim,num_layers=c.arch.num_layers,
                activation=c.arch.activation,use_spatial_emb=False,
                use_temporal_emb=False,static=True,key=key)
            return eqx.tree_at(lambda m:(m.output.weight,m.output.bias),model,
                (model.output.weight*c.arch.output_scale,model.output.bias*c.arch.output_scale))
        self.net=spatial(self.rank if self.field_layout=='separate' else 2*self.rank,phase_key)
        self.chemical_net=spatial(self.rank,chemical_key) if self.field_layout=='separate' else None

    def __call__(self,x,t):
        features,p0=self.inputs(x);coefficients=self.net(features)
        if self.field_layout=='separate':
            phase_coefficients=coefficients;chemical_coefficients=self.chemical_net(features)
        else:phase_coefficients,chemical_coefficients=jnp.split(coefficients,2)
        s=jnp.log1p(t/self.initial_time_scale)/jnp.log1p(self.duration/self.initial_time_scale)
        u=2*s-1;basis=[jnp.ones_like(u)]
        if self.rank>1:basis.append(u)
        for k in range(2,self.rank):basis.append(2*u*basis[-1]-basis[-2])
        basis=jnp.stack(basis)
        return jnp.stack((p0+s*jnp.dot(phase_coefficients,basis),
            self.chemical_scale*jnp.dot(chemical_coefficients,basis)))


def predict_fields(net,points,t,mean,volumes,D=None):
    if isinstance(net,TanhPhaseNetwork):
        if D is None:raise ValueError('Projected tanh phase currently requires the nonlocal chemical operator')
        from examples.confined_ch2d.tanh_mapping import phase_state
        phi,_,_=phase_state(net,points,t,mean,volumes)
        mu=net.beta*(phi**3-phi)+net.epsilon_squared*(D@phi)
        return jnp.stack((phi,mu),axis=-1)
    values=jax.vmap(net,in_axes=(0,None))(points,t)
    shift=mean-jnp.dot(volumes,values[:,0])/jnp.sum(volumes)
    if net.chemical_mode=='eliminated':
        if D is None:raise ValueError('Eliminated chemical potential requires the explicit regional operator')
        phi=values[:,0]+shift
        mu=net.beta*(phi**3-phi)+net.epsilon_squared*(D@phi)
        return jnp.stack((phi,mu),axis=-1)
    return values.at[:,0].add(shift)
