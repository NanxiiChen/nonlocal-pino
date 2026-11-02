"""Uniform BE-started BDF2 rates linking one network's neighboring states."""
import jax
import jax.numpy as jnp
import equinox as eqx


def step_count(c):
    return int(round(c.physical.terminal_time/c.time_discretization.dt))


def sample_intervals(key,c):
    """One uniform interval from each equal integer stratum; zero-based IDs."""
    per_stratum=step_count(c)//c.sample.Nt
    ids=jnp.arange(c.sample.Nt,dtype=jnp.int32)*per_stratum+jax.random.randint(
        key,(c.sample.Nt,),0,per_stratum,dtype=jnp.int32)
    times=(ids+1).astype(c.train.compute_dtype)*c.time_discretization.dt
    return ids,times


def sample_continuous(key,c):
    """Fresh shared times: one continuous uniform draw per equal time stratum."""
    draws=jax.random.uniform(key,(c.sample.Nt,),dtype=jnp.float32)
    return (jnp.arange(c.sample.Nt)+draws)*c.physical.terminal_time/c.sample.Nt


def previous_times(times,dt):
    nodes=jnp.rint(times/dt)
    tolerance=8*jnp.finfo(times.dtype).eps*jnp.maximum(1,jnp.abs(nodes))
    nodes=eqx.error_if(nodes,jnp.any(~jnp.isfinite(times))|jnp.any(nodes<1)
        |jnp.any(jnp.abs(times/dt-nodes)>tolerance),
        'BDF2 requires positive uniform-grid time nodes')
    return jnp.maximum(nodes-1,0.)*dt,jnp.maximum(nodes-2,0.)*dt


def backward_rate(current,previous,earlier,times,dt):
    """Euler for n=1, BDF2 for n>=2; all three physical states differentiated."""
    if current.shape!=previous.shape or current.shape!=earlier.shape or times.shape!=(len(current),):
        raise ValueError('Backward rates require aligned physical states and one time per state')
    first=(times<1.5*dt).reshape((len(times),)+(1,)*(current.ndim-1))
    return jnp.where(first,(current-previous)/dt,
        (1.5*current-2*previous+.5*earlier)/dt)
