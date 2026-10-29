"""Explicit residual-time measures; each update has a fixed number of nodes."""
import jax
import jax.numpy as jnp


def temporal_samples(c,iteration,key):
    if 'temporal' in c.sample and c.sample.temporal.kind=='fixed_nodes':
        return jnp.asarray(c.sample.temporal.nodes,dtype=jnp.float64),c.physical.terminal_time
    draws=jax.random.uniform(key,(c.sample.Nt,),dtype=jnp.float64,minval=1e-6,maxval=1-1e-6)
    if 'temporal' not in c.sample:
        # Preserve the historical expression/RNG path exactly.
        return (jnp.arange(c.sample.Nt)+draws)*c.physical.terminal_time/c.sample.Nt,c.physical.terminal_time
    s=c.sample.temporal;T=c.physical.terminal_time
    if s.curriculum:
        progress=min((iteration-1)/(s.warmup_iterations-1),1.)
        horizon=s.initial_horizon*(T/s.initial_horizon)**progress
    else:horizon=T
    nodes=(jnp.arange(c.sample.Nt)+draws)/c.sample.Nt
    times=s.time_scale*jnp.expm1(nodes*jnp.log1p(horizon/s.time_scale))
    return times,horizon
