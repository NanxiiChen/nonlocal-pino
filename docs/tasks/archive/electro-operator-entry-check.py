"""Public c.arch configuration/CLI dispatch and original pilot template parity."""
import json
from pathlib import Path
from unittest.mock import patch
import equinox as eqx
import jax
import numpy as np
from ml_collections import ConfigDict

from examples.electropolish2d import train
from examples.electropolish2d.configs.train_operator import get_config
from examples.electropolish2d.configs.train_operator_5k import get_config as pilot_config
from examples.electropolish2d.configs.train_base import get_config as single_config
from examples.electropolish2d.operator_training import _build_network
from examples.electropolish2d.operator_data import validate_config

train.FLAGS(['check','--config=examples/electropolish2d/configs/train_operator.py','--seed=2026'])
with patch('examples.electropolish2d.operator_training.train_operator',return_value={'success':True}) as run:
    assert train._main(None) is None
    c=run.call_args.args[0]
    assert c.train.seed==2026 and c.arch.model=='conditional_field'
    assert 'seed' not in get_config().train and 'seed' not in get_config().surface
    validate_config(c)
assert 'operator' not in single_config()
path=Path('/root/autodl-tmp/tf-logs/electropolish2d/20261028-electro-operator-cross-01')
old=ConfigDict(json.loads((path/'config.json').read_text()));new=pilot_config()
k=jax.random.split(jax.random.PRNGKey(42),3)[0]
a=_build_network(old.operator,old.operator.image_shape,k)
b=_build_network(new.arch,new.operator.image_shape,k)
for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)):
    if eqx.is_array(x):np.testing.assert_array_equal(x,y)
loaded=eqx.tree_deserialise_leaves(path/'model_initial.eqx',b)
for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(loaded)):
    if eqx.is_array(x):
        assert x.shape==y.shape and x.dtype==y.dtype
print('PASS canonical operator CLI/seed/success return, c.arch/explicit pilot template parity, single-IVP default unchanged')
