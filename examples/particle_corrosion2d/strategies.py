"""Established stagger/causal policies and explicit cold-prefix controls."""
import copy
import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import ConfigDict

from nonlocal_pinn.training.causal import CausalWeightor


def active_losses(c, iteration, count):
    """Repository stagger semantics: AC then CH blocks; all data terms stay on."""
    active = np.ones(count, dtype=bool)
    warmup=c.loss.data_warmup_steps if 'data_warmup_steps' in c.loss else 0
    if iteration<warmup:
        active[:2]=False
        return active
    relative=iteration-warmup
    if 'stagger_steps' in c.loss and relative < c.loss.stagger_steps:
        phase = (relative//c.loss.stagger_switch_every) % 2
        active[0], active[1] = phase == 0, phase == 1
    return active


def validate_strategies(c):
    if 'horizon_curriculum' in c.sample:
        if (c.nl.method!='nonlocal' or 'phase' in c or 'boundary' in c
                or ('use_causal' in c.loss and c.loss.use_causal)
                or ('stagger_steps' in c.loss and c.loss.stagger_steps)
                or 'time_form' in c.loss or 'transport_residual' in c.loss):
            raise ValueError('Horizon curriculum qualifies only original joint strong nonlocal residuals')
        from .sampling import horizon_fractions
        horizon_fractions(c)
    if (isinstance(c.train.gradnorm_update_every,bool)
            or not isinstance(c.train.gradnorm_update_every,int)
            or c.train.gradnorm_update_every<1):
        raise ValueError('Gradient balancing requires a positive integer refresh interval')
    if 'gradnorm_norm' in c.loss:
        if c.loss.gradnorm_norm not in ('l1','l2') or c.nl.method!='nonlocal':
            raise ValueError('Gradient norm control requires l1/l2 on the qualified nonlocal path')
    if 'transport_residual' in c.loss:
        if (c.loss.transport_residual!='ac_eliminated' or c.nl.method!='nonlocal'
                or 'phase' in c or 'boundary' in c or 'time_form' in c.loss):
            raise ValueError('AC-eliminated transport requires original strong nonlocal fields and soft data')
    if 'time_form' in c.loss:
        if (c.loss.time_form!='integral_simpson' or c.nl.method!='nonlocal'
                or 'phase' in c or 'boundary' in c
                or ('use_causal' in c.loss and c.loss.use_causal)
                or 'temporal' not in c.sample or c.sample.temporal.kind!='integral_initial_layer'):
            raise ValueError('Integral Simpson requires original nonlocal fields,its endpoint rule and no causal flag')
    if 'data_warmup_steps' in c.loss:
        if (isinstance(c.loss.data_warmup_steps,bool) or not isinstance(c.loss.data_warmup_steps,int)
                or not 0<=c.loss.data_warmup_steps<c.train.iterations or c.nl.method!='nonlocal'):
            raise ValueError('Data warmup needs explicit integer steps below budget on nonlocal path')
        if c.loss.data_warmup_steps and ('phase' in c or 'boundary' in c):
            raise ValueError('Data warmup requires the original outputs with soft IC/BC losses')
    if 'initial_metric' in c.loss:
        if (c.loss.initial_metric not in ('physical','phase_liquid') or c.nl.method!='nonlocal'
                or 'phase' in c):
            raise ValueError('initial_metric requires original nonlocal outputs and physical/phase_liquid')
    bias=np.asarray(c.loss.gradnorm_bias)
    if not np.isfinite(bias).all() or np.any(bias<=0):
        raise ValueError('Loss priorities must be positive finite')
    enabled=(('use_causal' in c.loss and c.loss.use_causal)
             or ('stagger_steps' in c.loss and c.loss.stagger_steps))
    if enabled and c.nl.method!='nonlocal':
        raise ValueError('These strategy controls qualify the explicit nonlocal path')
    if 'use_causal' in c.loss and not isinstance(c.loss.use_causal, bool):
        raise ValueError('loss.use_causal must be an explicit boolean')
    if 'causal_steps' in c.loss:
        if ('use_causal' not in c.loss or not c.loss.use_causal
                or isinstance(c.loss.causal_steps,bool) or not isinstance(c.loss.causal_steps,int)
                or not 0<c.loss.causal_steps<c.train.iterations):
            raise ValueError('Causal release requires causal enabled and an integer step inside the budget')
    if 'stagger_steps' in c.loss:
        if (isinstance(c.loss.stagger_steps, bool) or not isinstance(c.loss.stagger_steps, int)
                or not 0 <= c.loss.stagger_steps <= c.train.iterations
                or 'stagger_switch_every' not in c.loss
                or isinstance(c.loss.stagger_switch_every, bool)
                or not isinstance(c.loss.stagger_switch_every, int)
                or c.loss.stagger_switch_every < 1):
            raise ValueError('Stagger needs explicit integer steps in budget and positive switch interval')
    if 'use_causal' in c.loss and c.loss.use_causal:
        required = {'eps_init', 'max_eps', 'step_size', 'max_last_weight', 'min_mean_weight'}
        if 'causal' not in c or set(c.causal) != required:
            raise ValueError('Causal policy requires its five explicit controller settings')
        s = c.causal
        if (not np.isfinite([s[k] for k in required]).all() or not 0 < s.eps_init <= s.max_eps
                or s.step_size <= 1 or not 0 < s.min_mean_weight < s.max_last_weight < 1):
            raise ValueError('Invalid adaptive causal controller ranges')
        if 'stagger_steps' in c.loss and c.loss.stagger_steps:
            raise ValueError('Initial controlled study tests stagger and causal separately')


def load_strategy_control(c, run, inputs, reference):
    """Reproduce a completed original-output run, optionally using its prefix."""
    run = Path(run).resolve(strict=True)
    prior = json.loads((run/'config.json').read_text())
    if 'horizon_curriculum' in prior['sample']:
        raise ValueError('Strategy source must not already have a horizon curriculum')
    runtime = json.loads((run/'runtime.json').read_text())
    summary = json.loads((run/'summary.json').read_text())
    current = json.loads(c.to_json())
    if (current['train']['optimizer'] not in ('adam','soap')
            or prior['train']['optimizer'] not in ('adam','soap')):
        raise ValueError('Strategy optimizer controls qualify only explicit Adam/SOAP choices')
    if summary['steps'] != prior['train']['iterations'] or c.train.iterations > summary['steps']:
        raise ValueError('Strategy control must be completed and cover the requested prefix')
    for name in ('model_initial.eqx', 'optimizer_initial.eqx', 'model_final.eqx'):
        if not (run/name).is_file():
            raise ValueError('Incomplete strategy control: '+name)
    if runtime['inputs'] != str(Path(inputs).resolve()) or runtime['reference'] != str(Path(reference).resolve()):
        raise ValueError('Strategy control requires identical input bank and evaluation reference')
    if any(name in current or name in prior for name in ('phase', 'boundary')):
        raise ValueError('Strategy controls require the original output and soft IC/BC')

    def canonical(value):
        result = copy.deepcopy(value)
        result['train'].pop('iterations')
        result['train'].pop('optimizer')
        result['train'].pop('gradnorm_update_every')
        if 'time_coordinate' in result['arch']:
            result['arch'].pop('time_coordinate')
        for name in ('gradnorm_bias', 'ac_scale', 'ch_scale', 'stagger_steps',
                     'stagger_switch_every', 'use_causal', 'initial_metric','data_warmup_steps',
                     'pde_reduction','time_form','transport_residual','gradnorm_norm','causal_steps'):
            if name in result['loss']:
                result['loss'].pop(name)
        if 'causal' in result:
            result.pop('causal')
        if 'temporal' in result['sample']:
            result['sample'].pop('temporal')
        if 'horizon_curriculum' in result['sample']:
            result['sample'].pop('horizon_curriculum')
        return result

    if canonical(current) != canonical(prior):
        raise ValueError('Strategy control changed non-strategy numerical/model settings')
    prior_temporal='temporal' in prior['sample']
    if prior_temporal and (prior['sample']['temporal']['kind']!='initial_layer'
            or 'temporal' not in current['sample']
            or prior['sample']['temporal']!=current['sample']['temporal']):
        raise ValueError('A temporal strategy source requires the same initial-layer sampling rule')
    if 'time_coordinate' in prior['arch']:
        raise ValueError('Strategy source must retain the original linear time coordinate')
    if 'time_coordinate' in c.arch:
        coordinate=c.arch.time_coordinate
        if (set(coordinate)!={'kind','time_scale'} or coordinate.kind!='logarithmic'
                or not np.isfinite(coordinate.time_scale) or coordinate.time_scale<=0):
            raise ValueError('Strategy time coordinate must declare logarithmic and a positive scale')
    if ('temporal' in current['sample']
            and current['sample']['temporal']['kind'] not in ('initial_layer','integral_initial_layer')):
        raise ValueError('Only explicit initial-layer sampling may change source times')
    with np.load(run/'sampling.npz', allow_pickle=False) as source:
        trace = {k:source[k] for k in source.files}
    if set(trace) != {'condition_ids', 'pool_ids', 'times'}:
        raise ValueError('Unexpected source sampling schema')
    rng = np.random.default_rng(runtime['seed'])
    shape = (summary['steps'], prior['operator']['condition_batch'])
    expected = dict(condition_ids=rng.integers(prior['operator']['geometry_count'], size=shape, dtype=np.int32),
                    pool_ids=rng.integers(prior['sample']['K_pool'], size=shape, dtype=np.int32),
                    times=rng.uniform(0, prior['sample']['t_max'],
                                      size=(summary['steps'], prior['sample']['Nt'])).astype(np.float32))
    if prior_temporal:
        from .sampling import initial_layer_times
        expected['times']=initial_layer_times(expected['times'],ConfigDict(prior))
    for key in trace:
        if trace[key].dtype!=expected[key].dtype or not np.array_equal(trace[key], expected[key]):
            raise ValueError('Source sampling fails independent replay: '+key)
    selected={key:value[:c.train.iterations] for key,value in trace.items()}
    if 'temporal' in c.sample and not prior_temporal:
        from .sampling import initial_layer_times
        selected['times']=initial_layer_times(selected['times'],c)
    if 'horizon_curriculum' in c.sample:
        from .sampling import apply_horizon_curriculum
        selected['times']=apply_horizon_curriculum(selected['times'],c)
    return runtime, selected


def verify_cold_initialization(net,state,c,control,allow_optimizer_change):
    """Check cold leaves; different optimizers retain their own fresh state."""
    prior=json.loads((control/'config.json').read_text())
    same_optimizer=prior['train']['optimizer']==c.train.optimizer
    if not same_optimizer and not allow_optimizer_change:
        raise ValueError('This paired control requires the same optimizer')
    if not same_optimizer and {prior['train']['optimizer'],c.train.optimizer}!={'adam','soap'}:
        raise ValueError('Only an explicit Adam/SOAP cold-state comparison is qualified')
    old_net=eqx.tree_deserialise_leaves(control/'model_initial.eqx',net)
    pairs=[(net,old_net)]
    if same_optimizer:
        old_state=eqx.tree_deserialise_leaves(control/'optimizer_initial.eqx',state)
        pairs.append((state,old_state))
    for new_tree,old_tree in pairs:
        for new,old in zip(jax.tree.leaves(new_tree),jax.tree.leaves(old_tree),strict=True):
            if eqx.is_array(new) and not np.array_equal(np.asarray(new),np.asarray(old)):
                raise ValueError('Cold paired model/optimizer do not reproduce saved initialization')
    return same_optimizer


def parameter_gradient_norm(gradient,kind):
    """Historical L1,or Euclidean norm without squaring unscaled large leaves."""
    if kind=='l1':
        return sum(jnp.sum(jnp.abs(a)) for a in jax.tree.leaves(gradient) if eqx.is_array(a))
    if kind!='l2':raise ValueError('Unknown parameter gradient norm: '+str(kind))
    arrays=[a for a in jax.tree.leaves(gradient) if eqx.is_array(a) and a.size]
    if not arrays:return jnp.asarray(0.)
    scale=jnp.max(jnp.stack([jnp.max(jnp.abs(a)) for a in arrays]))
    divisor=jnp.where(scale>0,scale,jnp.ones_like(scale))
    return scale*jnp.sqrt(sum(jnp.sum((a/divisor)**2) for a in arrays))


def make_causal_step(components, optimizer, c, loss_names):
    """Existing adaptive causal controller; IC/BC remain outside time weighting."""
    count = len(loss_names)
    bias = jnp.asarray(c.loss.gradnorm_bias, dtype=jnp.float32)
    if bias.shape != (count,):
        raise ValueError('Causal loss needs one explicit bias per component')
    controller = CausalWeightor()
    norm_kind=c.loss.gradnorm_norm if 'gradnorm_norm' in c.loss else 'l1'

    @eqx.filter_jit
    def step(net, state, weights, epsilon, times, payload, refresh, active=None, causal_active=None):
        if causal_active is not None and (causal_active.shape!=() or causal_active.dtype!=jnp.bool_):
            raise ValueError('Causal release selector requires a scalar boolean')
        effective_epsilon=epsilon if causal_active is None else jnp.where(causal_active,epsilon,0.)
        values, pullback, aux = eqx.filter_vjp(
            lambda model: components(model, times, payload, effective_epsilon), net, has_aux=True)

        def update(_):
            def norm(i):
                gradient = pullback(jax.nn.one_hot(i, count, dtype=values.dtype))[0]
                return parameter_gradient_norm(gradient,norm_kind)
            raw_norms=jax.lax.map(norm,jnp.arange(count,dtype=jnp.int32))
            norms=jnp.clip(raw_norms,1e-6,1e6)
            proposal = jnp.clip(jnp.mean(norms)/norms, 1e-6, c.loss.gradnorm_weight_cap)
            return (jax.lax.stop_gradient(c.train.alpha_gradnorm*proposal+(1-c.train.alpha_gradnorm)*weights),
                    jnp.all(jnp.isfinite(raw_norms)))

        weights,norm_finite=jax.lax.cond(refresh,update,lambda _:(weights,jnp.asarray(True)),operand=None)
        cotangent=weights*bias
        if active is not None:
            if active.shape!=(count,) or active.dtype!=jnp.bool_:
                raise ValueError('Causal objective selector requires a fixed boolean loss vector')
            cotangent=cotangent*active
        gradient = pullback(cotangent)[0]
        s = c.causal
        next_epsilon = controller.update_eps(aux[0], epsilon, s.step_size, s.max_eps,
                                              s.max_last_weight, s.min_mean_weight)
        if active is not None:
            next_epsilon=jnp.where(jnp.any(active[:2]),next_epsilon,epsilon)
        if causal_active is not None:
            next_epsilon=jnp.where(causal_active,next_epsilon,epsilon)
        finite = (norm_finite & jnp.all(jnp.isfinite(values)) & jnp.all(jnp.isfinite(aux))
                  & jnp.all(jnp.isfinite(weights)) & jnp.all(weights > 0)
                  & jnp.isfinite(next_epsilon) & (next_epsilon > 0)
                  & jnp.all(jnp.stack([jnp.all(jnp.isfinite(a)) for a in jax.tree.leaves(gradient) if eqx.is_array(a)])))
        updates, state = optimizer.update(gradient, state, net)
        return eqx.apply_updates(net, updates), state, weights, next_epsilon, values, aux, finite

    return step
