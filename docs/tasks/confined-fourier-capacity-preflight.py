"""Input-only parameter/SOAP-state sizes for ordinary backbone-width controls."""
import json
from pathlib import Path
import equinox as eqx
import jax
import numpy as np
from examples.confined_ch2d.configs.train_operator_smallscale10_512_batch import get_config
from examples.confined_ch2d.operator_problem import validate
from examples.confined_ch2d.operator_training import build_network
from nonlocal_pinn.training.optimizers import make_optimizer


def main():
    jax.config.update('jax_enable_x64',True)
    rows=[]
    for width in (64,96,128):
        c=get_config();c.arch.width=width;validate(c)
        model=build_network(c,jax.random.split(jax.random.key(42),3)[0])
        state=make_optimizer(c).init(eqx.filter(model,eqx.is_array))
        parameters=[np.asarray(a) for a in jax.tree.leaves(model) if eqx.is_array(a)]
        arrays=[np.asarray(a) for a in jax.tree.leaves(state) if eqx.is_array(a)]
        assert all(np.isfinite(a).all() for a in parameters+arrays)
        rows.append(dict(width=width,trainable_parameters=sum(a.size for a in parameters)-model.frequencies.size,
            model_array_bytes=sum(a.nbytes for a in parameters),soap_state_bytes=sum(a.nbytes for a in arrays),
            spatial_fourier_channels=2*c.arch.spatial_features,time_fourier_channels=2*c.arch.time_features,
            heads=c.arch.heads,condition_depth=c.arch.condition_depth,decoder_depth=c.arch.decoder_depth))
    result=dict(scope='Zero updates, no evolved fields, JAX CPU. Model/state bytes exclude bank, activations and GPU allocator/workspace; not a GPU memory or accuracy benchmark.',controls=rows)
    output=Path('docs/tasks/confined-fourier-capacity-preflight.json')
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
