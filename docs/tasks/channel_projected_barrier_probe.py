"""Frozen query-context diagnostic; test a materialization barrier without training."""
import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

import equinox as eqx
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict

from examples.channel_nsch2d.network import build_network
from examples.channel_nsch2d.projected_fields import make_decoder_fn
from examples.channel_nsch2d.chemical_projection import project_state
from docs.tasks.channel_projected_query_probe import check
from nonlocal_pinn.training.runtime import configure_gpu_policy


def main(control,output,variants):
    control,output=Path(control),Path(output);output.mkdir(parents=True,exist_ok=False)
    c=ConfigDict(json.loads((control/'config.json').read_text()))
    c.chemical_constraint=dict(kind='interior_gmls_projection',boundary='decoder_trace',query='mapped_bilinear')
    configure_gpu_policy(c.train.gpu_policy)
    jax.config.update('jax_enable_x64',False);jax.config.update('jax_default_matmul_precision',c.train.matmul_precision)
    key,_=jax.random.split(jax.random.PRNGKey(c.train.seed))
    model=eqx.tree_deserialise_leaves(control/'model_final.eqx',build_network(c,key))
    def barrier_domain(config):
        decoder=make_decoder_fn(config)
        def value(net,e,d,ts):
            raw=decoder(net,e,d['points'],ts,d['amplitudes'])
            physical=project_state(jax.lax.optimization_barrier(raw),d,config,xp=jnp)
            return jax.lax.optimization_barrier(physical)if kind=='materialized_state'else physical
        return value
    results=[]
    for kind in variants:
        def execute():
            try:return dict(passed=True,report=check(model,c,control))
            except ValueError as error:return dict(passed=False,error=str(error))
        if kind=='original_query':result=execute()
        else:
            with patch('examples.channel_nsch2d.projected_fields.make_domain_fn',barrier_domain):result=execute()
        result['kind']=kind;results.append(result);print(json.dumps(result),flush=True)
    report=dict(status='frozen_query_context_barrier_diagnostic_completed',source_run=str(control),
        optimizer_updates=0,reference_labels_used=0,production_source_changed=False,results=results)
    (output/'validation.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--control',required=True);p.add_argument('--output',required=True)
    p.add_argument('--variants',nargs='+',choices=('original_query','barrier_query','materialized_state'),required=True)
    a=p.parse_args();main(a.control,a.output,a.variants)
