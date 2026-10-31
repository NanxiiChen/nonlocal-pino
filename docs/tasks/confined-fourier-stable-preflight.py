"""Full input-bank/spawn/PDE-gradient preflight; no optimizer updates or labels."""
import json
from pathlib import Path
import time
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from examples.confined_ch2d.configs.train_operator_smallscale10_512_stable import get_config
from examples.confined_ch2d.operator_problem import validate
from examples.confined_ch2d.operator_training import load_bank,build_network,make_loss,fields


def main():
    jax.config.update('jax_enable_x64',True);jax.config.update('jax_default_matmul_precision','highest')
    c=get_config();validate(c);start=time.perf_counter();manifest,refs,bank=load_bank(c)
    assert set(bank)=={'points','volumes','initial','neighbors','coefficients','metric','condition','condition_weights'}
    setup=time.perf_counter()-start
    net=build_network(c,jax.random.split(jax.random.key(42),3)[0])
    batch=jax.tree.map(lambda a:a[jnp.array([0,1,2,3])],bank)
    loss=eqx.filter_jit(eqx.filter_value_and_grad(make_loss(c)))
    value,gradient=loss(net,batch,jnp.array([0.,.1,.5,1.5,2.8]));value.block_until_ready()
    arrays=[np.asarray(a) for a in jax.tree.leaves(gradient) if eqx.is_array(a)]
    assert np.isfinite(float(value)) and all(np.isfinite(a).all() for a in arrays)
    assert all(np.linalg.norm(np.asarray(a.layers[0].weight))>0 for a in gradient.lifts)
    item=jax.tree.map(lambda a:a[0],bank);phi=np.asarray(fields(net,item,jnp.array([0.,1.5,3.]),c))[...,0]
    initial_gap=float(abs(phi[0]-np.asarray(item['initial'])).max())
    mass_gap=float(abs(phi@np.asarray(item['volumes'])-np.asarray(item['initial'])@np.asarray(item['volumes'])).max())
    assert initial_gap==0 and mass_gap<1e-12
    reports=manifest['training_metric']['conditions']
    result=dict(status='passed',backend=jax.default_backend(),conditions=len(reports),setup_seconds=setup,
        wall_seconds=time.perf_counter()-start,metric=c.operator.metric,workers=c.operator.metric_workers,
        minimum_spectral_weight=min(r['minimum_weight'] for r in reports),
        maximum_constant_mode_error=max(r['constant_mode_error'] for r in reports),
        input_payload_keys=sorted(bank),loss=float(value),finite_pde_gradients=True,encoder_gradients_nonzero=True,
        exact_initial_max_difference=initial_gap,physical_mass_max_difference=mass_gap,
        optimizer_updates=0,future_fields_in_loss=False)
    Path('docs/tasks/confined-fourier-stable-bank-preflight.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
