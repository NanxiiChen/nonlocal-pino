"""Optional matrix-free Newton-CG on the unchanged global PDE objective."""
import time,json
import numpy as np
import jax,jax.numpy as jnp,equinox as eqx
from jax.flatten_util import ravel_pytree
from scipy.optimize import minimize


def refine(net,loss_fn,initial_mean,config,run,callback):
    s=config.train.refinement;times=jnp.asarray(s.time_nodes,dtype=jnp.float64)
    params,static=eqx.partition(net,eqx.is_inexact_array);x0,unravel=ravel_pytree(params)
    def objective(z):return loss_fn(eqx.combine(unravel(z),static),times,initial_mean)[0]
    value_grad=jax.jit(jax.value_and_grad(objective));gradient=jax.grad(objective)
    product=jax.jit(lambda z,d:jax.jvp(gradient,(z,),(d,))[1])
    counts=dict(iterations=0,value_gradient_calls=0,hessian_vector_products=0)
    start=time.perf_counter()
    def evaluate(z):
        value,g=value_grad(jnp.asarray(z));a,b=float(value),np.asarray(g,dtype=np.float64)
        if not np.isfinite(a) or not np.isfinite(b).all():raise FloatingPointError('Nonfinite Newton-CG objective/gradient')
        counts['value_gradient_calls']+=1
        return a,b
    def hessian(z,d):
        result=np.asarray(product(jnp.asarray(z),jnp.asarray(d)),dtype=np.float64)
        if not np.isfinite(result).all():raise FloatingPointError('Nonfinite Newton-CG Hessian product')
        counts['hessian_vector_products']+=1
        return result
    with (run/'refinement.jsonl').open('x') as log:
        def accepted(z):
            counts['iterations']+=1;i=counts['iterations']
            if i==1 or i%25==0:
                value,g=evaluate(z)
                row=dict(iteration=i,loss=value,gradient_inf_norm=float(np.max(abs(g))),**{k:v for k,v in counts.items() if k!='iterations'})
                log.write(json.dumps(row,allow_nan=False)+'\n');log.flush()
                print(f'[Newton-CG {i}] loss={value:.7g} hvp={counts["hessian_vector_products"]}',flush=True)
            if i%config.log.eval_every==0:callback(eqx.combine(unravel(jnp.asarray(z)),static),i)
        result=minimize(evaluate,np.asarray(x0),method='Newton-CG',jac=True,hessp=hessian,callback=accepted,
            options=dict(maxiter=s.iterations,xtol=s.step_tolerance,c1=s.c1,c2=s.c2))
    final=eqx.combine(unravel(jnp.asarray(result.x)),static)
    record=dict(optimizer='Newton-CG',success=bool(result.success),status=int(result.status),message=str(result.message),
        iterations=int(result.nit),scipy_function_evaluations=int(result.nfev),
        total_function_evaluations=counts['value_gradient_calls'],hessian_vector_products=counts['hessian_vector_products'],
        final_loss=float(result.fun),final_gradient_inf_norm=float(np.max(abs(result.jac))),
        wall_seconds=time.perf_counter()-start,time_points=np.asarray(times).tolist(),
        training_targets='Unchanged global PDE-only objective; no reference values')
    (run/'refinement_summary.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    return final,record
