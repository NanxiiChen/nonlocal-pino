"""Input-only initial range check for the existing unit decoder gain option."""
import json
import argparse
import importlib
from pathlib import Path
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from examples.confined_ch2d.operator_problem import validate,fixed_initial
from examples.confined_ch2d.operator_training import build_network,fields
from examples.confined_ch2d.operator_conditioning import initial_rhs_enabled,initial_operator_fields


def main(preset,output):
    jax.config.update('jax_enable_x64',True)
    c=importlib.import_module(preset).get_config();validate(c);root=Path(c.operator.dataset_path)
    m=json.loads((root/'manifest.json').read_text());i=512
    with np.load(root/m['conditions'][i]['file'],allow_pickle=False) as f:
        data={k:jnp.asarray(f[k]) for k in ('points','volumes','neighbors','coefficients','condition','condition_weights')}
    data['initial']=fixed_initial(data['points'],c,xp=jnp)
    if initial_rhs_enabled(c):data['domain_fields']=jnp.asarray(initial_operator_fields(data,c))
    net=build_network(c,jax.random.split(jax.random.key(42),3)[0]);times=jnp.linspace(0,3,5)
    result=np.asarray(eqx.filter_jit(lambda n,d:fields(n,d,times,c))(net,data));phi=result[...,0]
    assert np.isfinite(result).all();np.testing.assert_array_equal(phi[0],data['initial'])
    drift=float(abs(phi@np.asarray(data['volumes'])-np.asarray(data['initial'])@np.asarray(data['volumes'])).max())
    assert drift<1e-12
    report=dict(config=preset,status='passed',index=i,output_scale=c.arch.output_scale,
        trainable_parameters=sum(np.size(a) for a in jax.tree.leaves(net) if eqx.is_array(a))-net.frequencies.size,
        initial_rhs_condition=bool(initial_rhs_enabled(c)),
        phase_range=[float(phi.min()),float(phi.max())],mass_drift=drift,
        phase_rms=np.sqrt(np.sum(phi**2*np.asarray(data['volumes']),axis=1)/np.asarray(data['volumes']).sum()).tolist(),
        optimizer_updates=0,evolved_reference_fields_used=False)
    if initial_rhs_enabled(c):
        tangent=np.asarray(data['domain_fields'])[:,1];v=np.asarray(data['volumes'])
        report['known_initial_tangent']=dict(rms=float(np.sqrt(v@(tangent**2)/v.sum())),maximum_absolute=float(abs(tangent).max()))
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.config,a.output)
