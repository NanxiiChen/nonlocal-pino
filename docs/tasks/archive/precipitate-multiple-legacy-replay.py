"""Check old single-particle checkpoint compatibility without editing its run."""
from pathlib import Path
import json
import sys
ROOT = next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
sys.path.insert(0,str(ROOT))
import equinox as eqx
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict
import numpy as np
from examples.precipitate2d.problem import load_reference, initial_field, validate_config
from examples.precipitate2d.train import _build_network
from examples.precipitate2d.network import predict_fields
from examples.precipitate2d.mechanics import ElasticEquilibrium


def main():
    if jax.default_backend() != 'cpu':raise ValueError('Use CPU')
    jax.config.update('jax_enable_x64',True)
    run = Path('/root/autodl-tmp/tf-logs/precipitate2d/20261027-periodic-elastic-pilot-01')
    c = ConfigDict(json.loads((run/'config.json').read_text())); validate_config(c)
    ref = load_reference(c)
    key = jax.random.split(jax.random.PRNGKey(c.train.seed))[0]
    net = eqx.tree_deserialise_leaves(run/'model_best.eqx',_build_network(c,key))
    points = jnp.asarray(ref['points']); mean = jnp.mean(initial_field(points,c,xp=jnp))
    mechanics = ElasticEquilibrium(c)
    def at(t):
        pq = predict_fields(net,points,t,mean)
        _,u,stress = mechanics(pq[:,0])
        return pq,u,stress
    pq,u,stress = (np.asarray(v) for v in eqx.filter_jit(jax.vmap(at))(jnp.asarray(ref['time'])))
    shape = (len(ref['time']),c.sample.Nx,c.sample.Ny)
    fields = dict(phi=pq[...,0].reshape(shape),q=pq[...,1].reshape(shape),
                  u=u.reshape(shape+(2,)),stress=stress.reshape(shape+(3,)))
    errors = {}
    with np.load(run/'prediction_best.npz',allow_pickle=False) as saved:
        for k,v in fields.items():
            np.testing.assert_allclose(v,saved[k],atol=1e-8,rtol=1e-8)
            errors[k] = float(np.max(np.abs(v-saved[k])))
    record = dict(status='passed',run=str(run),saved_schema='legacy single particle',
                  checkpoint=str(run/'model_best.eqx'),maximum_absolute_differences=errors,
                  run_files_written=False)
    Path(__file__).with_name('precipitate-multiple-legacy-replay.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
