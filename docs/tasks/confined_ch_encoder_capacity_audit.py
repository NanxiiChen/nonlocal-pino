"""CPU-only shape and dot-work audit; no GPU timing or training claims."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict

from examples.confined_ch2d.operator_training import (
    apply_operator, build_network, encode, make_loss, phase,
)
from nonlocal_pinn.training.optimizers import make_optimizer


def dot_work(closed):
    """Count scalar multiply/add dot work; nested branches are not GPU timings."""
    counts = Counter()
    def nested(value):
        if hasattr(value, "jaxpr") and hasattr(value.jaxpr, "eqns"):
            walk(value.jaxpr)
        elif hasattr(value, "eqns"):
            walk(value)
        elif isinstance(value, dict):
            for item in value.values():
                nested(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                nested(item)
    def walk(graph):
        for equation in graph.eqns:
            counts[equation.primitive.name] += 1
            if equation.primitive.name == "dot_general":
                axes = equation.params["dimension_numbers"][0][0]
                contraction = math.prod(equation.invars[0].aval.shape[i] for i in axes)
                counts["dot_flops"] += 2 * contraction * math.prod(equation.outvars[0].aval.shape)
            nested(equation.params)
    nested(closed)
    return dict(counts)


def audit(run, output):
    if output.exists():
        raise FileExistsError(output)
    jax.config.update("jax_enable_x64", True)
    if jax.default_backend() != "cpu":
        raise RuntimeError("Run this audit with JAX_PLATFORMS=cpu; training owns GPU")
    config = ConfigDict(json.loads((run / "config.json").read_text()))
    probe = json.loads(Path("docs/tasks/confined-ch-width128-preflight.json").read_text())
    batch_size = config.operator.condition_batch
    shapes = {name:jax.ShapeDtypeStruct((batch_size, *shape[1:]),
              jnp.int64 if name == "neighbors" else jnp.float64)
              for name,shape in probe["bank_shapes"].items()}
    nodes = shapes["points"].shape[1]
    nt = config.sample.Nt + 1
    times = jax.ShapeDtypeStruct((nt,), jnp.float64)
    records = []
    for width in (128, 192, 256):
        c = ConfigDict(config.to_dict())
        c.arch.width = width
        net = build_network(c, jax.random.key(c.train.seed))
        parameters, static = eqx.partition(net, eqx.is_array)
        def combine(p):
            return eqx.combine(p, static)
        groups = {}
        for name in ("lifts", "spatial", "cross", "time", "layers", "output"):
            groups[name] = sum(a.size for a in jax.tree.leaves(getattr(net,name)) if eqx.is_array(a))
        skipped = []
        # SOAP's cap applies independently to each tensor axis.
        for path, array in jax.tree_util.tree_leaves_with_path(parameters):
            if eqx.is_array(array) and array.ndim > 1:
                for axis, size in enumerate(array.shape):
                    if size > c.train.soap_max_precond_dim:
                        skipped.append(dict(path=jax.tree_util.keystr(path), shape=list(array.shape), axis=axis))
        opt = make_optimizer(c)
        state_shapes = jax.eval_shape(opt.init, parameters)
        state_bytes = sum(math.prod(a.shape)*a.dtype.itemsize
                          for a in jax.tree.leaves(state_shapes) if hasattr(a,"shape"))
        record = dict(width=width, heads=c.arch.heads, head_features=width//c.arch.heads,
            parameter_groups=groups, trainable_parameters=sum(groups.values()),
            optimizer_state_bytes=state_bytes, soap_axis_cap=c.train.soap_max_precond_dim,
            skipped_matrix_axes=skipped)
        larger_cap = ConfigDict(c.to_dict())
        larger_cap.train.soap_max_precond_dim = 512
        wider_state = jax.eval_shape(make_optimizer(larger_cap).init, parameters)
        record["cap512_optimizer_state_bytes"] = sum(math.prod(a.shape)*a.dtype.itemsize
            for a in jax.tree.leaves(wider_state) if hasattr(a,"shape"))
        record["cap512_skipped_matrix_axes"] = [row for row in skipped if row["shape"][row["axis"]]>512]
        if width == 128:
            def encoder(p, data):
                return jax.vmap(lambda d:encode(combine(p),d))(data)
            encoded_shape = jax.eval_shape(encoder, parameters, shapes)
            def spatial(p, data, contexts):
                return jax.vmap(lambda d,z:combine(p).prepare_encoded(z,d["points"]))(data,contexts)
            prepared_shape = jax.ShapeDtypeStruct((batch_size,nodes,width),jnp.float64)
            def decoder(p, data, prepared, t):
                model=combine(p)
                def one(d,h):
                    return jax.vmap(lambda tt:jax.jvp(
                        lambda s:phase(model,h,s,d,c), (tt,), (jnp.ones_like(tt),)))(t)
                return jax.vmap(one)(data,prepared)
            field_shape=jax.ShapeDtypeStruct((batch_size,nt,nodes),jnp.float64)
            def physics(data, fields, derivatives):
                def one(d, values, dt):
                    def at(phi, derivative):
                        mu=c.physical.beta*(phi**3-phi)+c.physical.epsilon**2*apply_operator(phi,d)
                        residual=c.physical.terminal_time*(derivative+c.physical.mobility*apply_operator(mu,d))
                        residual=d["metric"]@residual
                        return jnp.dot(d["volumes"],residual**2)/jnp.sum(d["volumes"])
                    return jax.vmap(at)(values,dt)
                return jax.vmap(one)(data,fields,derivatives)
            loss=make_loss(c)
            full=lambda p,d,t:loss(combine(p),d,t)
            traces = {
                "input_lifting":jax.make_jaxpr(encoder)(parameters,shapes),
                "spatial_condition_reads":jax.make_jaxpr(spatial)(parameters,shapes,encoded_shape),
                "decoder_and_time_jvp":jax.make_jaxpr(decoder)(parameters,shapes,prepared_shape,times),
                "pde_and_metric":jax.make_jaxpr(physics)(shapes,field_shape,field_shape),
                "full_loss_forward":jax.make_jaxpr(full)(parameters,shapes,times),
                "full_loss_value_and_gradient":jax.make_jaxpr(jax.value_and_grad(full))(parameters,shapes,times),
            }
            record["traced_dot_work"]={name:dot_work(trace) for name,trace in traces.items()}
            summed=sum(record["traced_dot_work"][name]["dot_flops"] for name in
                ("input_lifting","spatial_condition_reads","decoder_and_time_jvp","pde_and_metric"))
            assert summed==record["traced_dot_work"]["full_loss_forward"]["dot_flops"]
        records.append(record)
    toy=jax.make_jaxpr(lambda x,y:x@y)(jax.ShapeDtypeStruct((3,4),jnp.float64),
                                      jax.ShapeDtypeStruct((4,5),jnp.float64))
    assert dot_work(toy)["dot_flops"] == 2*3*4*5
    result=dict(status="passed", run=str(run), batch=batch_size, times=nt, padded_nodes=nodes,
        widths=records, scope="CPU abstract-shape traces and parameter/state sizes. Dot FLOPs omit elementwise/gather/scatter, fusion, bandwidth, optimizer update and QR execution. No GPU stage timings, accuracy, peak-memory or larger-network training measurements.")
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps(result,indent=2),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    audit(args.run.resolve(),args.output.resolve())
