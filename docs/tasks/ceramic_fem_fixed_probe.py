"""Fixed known geometry, full weak rows; pure-physics feasibility, no labels."""
import argparse
import json
from pathlib import Path
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np


def run(reference,output):
    from examples.particle_corrosion2d.configs.train_fem_comparison_weak import get_config
    from examples.particle_corrosion2d.evaluate import load_references,make_evaluator
    from examples.particle_corrosion2d.mesh import build_mesh
    from examples.particle_corrosion2d.fem_weak import diffusion_matrix,signed_payload
    from examples.particle_corrosion2d.inputs import conditioning,initial_payload,pad_graph
    from examples.particle_corrosion2d.network import build_network
    from examples.particle_corrosion2d.losses import make_loss_vector,make_step,LOSS_NAMES
    from nonlocal_pinn.training.optimizers import make_optimizer
    from nonlocal_pinn.training.runtime import configure_gpu_policy
    c=get_config();c.train.iterations=2000;c.operator.condition_batch=1
    configure_gpu_policy(c.train.gpu_policy)
    jax.config.update('jax_enable_x64',False)
    jax.config.update('jax_default_matmul_precision',c.train.matmul_precision)
    references,manifest=load_references(reference,c)
    record=manifest['cases'][0]['geometry']
    mesh=build_mesh(c,record,c.mesh.spacing)
    c.operator.online.halo=len(mesh['points'])
    free=np.flatnonzero(~mesh['mouth']);c.sample.K_core=len(free)
    graph=pad_graph(signed_payload(c,mesh,diffusion_matrix(mesh),free,mesh['volumes'][free]),
                    len(mesh['points']),c.operator.online.neighbors)
    rng=np.random.default_rng(42)
    data={**conditioning(c,record),**initial_payload(c,mesh,rng),**graph}
    output.mkdir(parents=True,exist_ok=False)
    (output/'config.json').write_text(c.to_json(indent=2)+'\n')
    np.savez_compressed(output/'known_inputs.npz',**data)
    payload={k:jnp.asarray(v,dtype=jnp.int32 if k in ('neighbors','core_indices') else jnp.float32)[None]
             for k,v in data.items()}
    net=build_network(c,jax.random.PRNGKey(42));optimizer=make_optimizer(c)
    state=optimizer.init(eqx.filter(net,eqx.is_array));weights=jnp.ones(5)
    step=make_step(make_loss_vector(c),optimizer,c,LOSS_NAMES)
    evaluator=make_evaluator(c,[references[0]])
    times=rng.uniform(0,c.sample.t_max,(2000,c.sample.Nt)).astype(np.float32)
    np.save(output/'times.npy',times)
    eqx.tree_serialise_leaves(output/'model_initial.eqx',net)
    evaluations=[];updates=[];started=time.perf_counter()
    def evaluate(index):
        pred,score=evaluator(net)
        row=dict(step=index,elapsed=time.perf_counter()-started,metrics=score)
        evaluations.append(row)
        with (output/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
        return pred
    evaluate(0)
    for i in range(2000):
        tick=time.perf_counter()
        net,state,weights,parts,finite=step(net,state,weights,jnp.asarray(times[i]),payload,jnp.asarray(i%25==0))
        jax.block_until_ready((net,state,weights,parts,finite));updates.append(time.perf_counter()-tick)
        if not bool(finite):raise FloatingPointError(f'Nonfinite FEM feasibility update{i+1}')
        if i==0 or (i+1)%100==0:
            row=dict(step=i+1,parts=np.asarray(parts).tolist(),weights=np.asarray(weights).tolist(),seconds=updates[-1])
            with (output/'loss.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            print(json.dumps(row),flush=True)
        if (i+1)%500==0:pred=evaluate(i+1)
    eqx.tree_serialise_leaves(output/'model_final.eqx',net)
    np.savez_compressed(output/'prediction.npz',prediction=pred[0],points=references[0]['points'],times=references[0]['times'])
    report=dict(status='passed',steps=2000,geometry=record,reference=str(reference),
        warm_seconds=float(np.median(updates[25:])),total_seconds=time.perf_counter()-started,
        initial=evaluations[0]['metrics'],final=evaluations[-1]['metrics'],
        peak_bytes=jax.devices()[0].memory_stats()['peak_bytes_in_use'],
        training_uses_reference_targets=False,scope='One fixed geometry, full FEM rows; feasibility only, not generalization')
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.reference,a.output)
