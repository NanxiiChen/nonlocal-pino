"""Replay the retained SOAP stage of the interrupted Newton cost control."""
from pathlib import Path
import sys,json
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
sys.path.insert(0,str(ROOT))
import numpy as np
import jax,jax.numpy as jnp,equinox as eqx
from ml_collections import ConfigDict
from examples.confined_ch2d.train import _build_network
from examples.confined_ch2d.problem import load_reference,initial_field
from examples.confined_ch2d.operators import RegionalLaplacian
from examples.confined_ch2d.network import predict_fields


def main():
    jax.config.update('jax_enable_x64',True)
    if jax.default_backend()!='cpu':raise ValueError('CPU replay required')
    run=Path('/root/autodl-tmp/tf-logs/confined_ch2d/20261028-chglobal-newton-01')
    c=ConfigDict(json.loads((run/'config.json').read_text()));ref=load_reference(c)
    points=jnp.asarray(ref['points']);v=jnp.asarray(ref['volumes'])
    D=RegionalLaplacian(ref['points'],ref['volumes'],c);mean=jnp.dot(v,initial_field(points,c,xp=jnp))/v.sum()
    key,_=jax.random.split(jax.random.PRNGKey(c.train.seed))
    model=eqx.tree_deserialise_leaves(run/'model_soap.eqx',_build_network(c,key))
    if (run/'model_soap.eqx').read_bytes()!=(run/'model_best.eqx').read_bytes():raise ValueError('Require identical SOAP and saved-best leaves for this interrupted control')
    @eqx.filter_jit
    def query(net):return jax.vmap(lambda t:predict_fields(net,points,t,mean,v,D))(jnp.asarray(ref['time']))
    actual=np.asarray(query(model));errors={}
    with np.load(run/'prediction_best.npz',allow_pickle=False) as stored:
        for j,field in enumerate(('phi','mu')):
            np.testing.assert_allclose(actual[...,j],stored[field],atol=1e-8,rtol=1e-8)
            errors[field]=float(np.max(abs(actual[...,j]-stored[field])))
    info=json.loads((run/'interruption.json').read_text());info['cpu_soap_replay_max_absolute']=errors
    (run/'interruption.json').write_text(json.dumps(info,indent=2,allow_nan=False)+'\n')
    Path(__file__).with_name('chglobal-newton-interruption.json').write_text(json.dumps(info,indent=2,allow_nan=False)+'\n')
    print(errors,flush=True)


if __name__=='__main__':main()
