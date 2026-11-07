"""Explicit global-time measures, retaining the historical uniform RNG path."""
import numpy as np


def horizon_fractions(c):
    """One global model: data warmup, expanding PDE coverage, then full time."""
    rule=c.sample.horizon_curriculum
    if (set(rule)!={'kind','initial_fraction','ramp_steps'} or rule.kind!='exponential'
            or isinstance(rule.initial_fraction,bool) or not np.isfinite(rule.initial_fraction)
            or not 0<rule.initial_fraction<1
            or isinstance(rule.ramp_steps,bool) or not isinstance(rule.ramp_steps,int)
            or rule.ramp_steps<2 or 'data_warmup_steps' not in c.loss):
        raise ValueError('Horizon curriculum needs exponential, initial_fraction in(0,1), ramp_steps>=2 and explicit data warmup')
    warmup=c.loss.data_warmup_steps
    if (isinstance(warmup,bool) or not isinstance(warmup,int) or warmup<0
            or warmup+rule.ramp_steps>=c.train.iterations
            or 'temporal' not in c.sample or c.sample.temporal.kind!='initial_layer'
            or rule.initial_fraction*c.sample.t_max<c.sample.temporal.early_end):
        raise ValueError('Horizon curriculum needs an initial-layer rule inside its first horizon and a full-time tail')
    result=np.ones(c.train.iterations,dtype=np.float32)
    result[warmup:warmup+rule.ramp_steps]=np.exp(
        np.linspace(np.log(rule.initial_fraction),0.,rule.ramp_steps)).astype(np.float32)
    return result


def apply_horizon_curriculum(times,c):
    """Keep the original early slots; change only the global physical queries."""
    if times.shape!=(c.train.iterations,c.sample.Nt) or times.dtype!=np.float32:
        raise ValueError('Horizon curriculum requires the complete float32 time schedule')
    if not np.isfinite(times).all() or np.any(times<0) or np.any(times>c.sample.t_max):
        raise ValueError('Horizon curriculum source times outside physical interval')
    result=times.copy()
    result[:,c.sample.temporal.early_count:]*=horizon_fractions(c)[:,None]
    return result


def initial_layer_times(draws, c):
    """Keep global draws, explicitly replace only the declared early-time slots."""
    rule=c.sample.temporal
    if (set(rule)!={'kind','early_count','early_end'} or rule.kind not in ('initial_layer','integral_initial_layer')
            or isinstance(rule.early_count,bool) or not isinstance(rule.early_count,int)
            or not 2<=rule.early_count<c.sample.Nt
            or not np.isfinite(rule.early_end) or not 0<rule.early_end<c.sample.t_max):
        raise ValueError('initial_layer needs 2<=early_count<Nt and 0<early_end<t_max')
    values=np.asarray(draws,dtype=np.float32)
    if values.shape!=(c.train.iterations,c.sample.Nt) or not np.isfinite(values).all():
        raise ValueError('Initial-layer draws must have the complete declared shape')
    if np.any(values<0) or np.any(values>np.float32(c.sample.t_max)):
        raise ValueError('Initial-layer source draws are outside physical time')
    out=values.copy();count=rule.early_count-1
    coordinate=(np.arange(count)[None,:]+values[:,1:rule.early_count].astype(np.float64)/c.sample.t_max)/count
    out[:,0]=0.
    out[:,1:rule.early_count]=(rule.early_end*coordinate).astype(np.float32)
    if rule.kind=='integral_initial_layer':
        count_global=c.sample.Nt-rule.early_count-1
        if count_global<1:raise ValueError('Integral time sampling needs positive global strata')
        out[:,rule.early_count:-1]=((np.arange(count_global)[None,:]
            +values[:,rule.early_count:-1].astype(np.float64)/c.sample.t_max)
            *c.sample.t_max/count_global).astype(np.float32)
        out[:,-1]=c.sample.t_max
    return out


def sample_times(rng, c):
    draws = rng.uniform(0, c.sample.t_max,
                        size=(c.train.iterations, c.sample.Nt))
    if 'temporal' not in c.sample:
        return draws.astype(np.float32)
    rule = c.sample.temporal
    if rule.kind in ('initial_layer','integral_initial_layer'):
        times=initial_layer_times(draws,c)
        return apply_horizon_curriculum(times,c) if 'horizon_curriculum' in c.sample else times
    if (set(rule) != {'kind', 'time_scale'} or rule.kind != 'logarithmic_stratified'
            or not np.isfinite(rule.time_scale) or rule.time_scale <= 0 or c.sample.Nt < 2):
        raise ValueError('Temporal rule requires logarithmic_stratified, positive time_scale and Nt>=2')
    coordinate = (np.arange(c.sample.Nt-1)[None, :]+draws[:, 1:]/c.sample.t_max)/(c.sample.Nt-1)
    times = rule.time_scale*np.expm1(coordinate*np.log1p(c.sample.t_max/rule.time_scale))
    # One explicit initial PDE node; the others sample the complete global interval.
    # This is a declared log-time residual measure, not uniform-time importance sampling.
    return np.concatenate((np.zeros((c.train.iterations, 1)), times), axis=1).astype(np.float32)
