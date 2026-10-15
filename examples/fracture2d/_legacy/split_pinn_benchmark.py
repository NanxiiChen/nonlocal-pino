"""Paired FP64 full-graph update benchmarks at identical learned checkpoint fields.

Warmup/compilation are separate from synchronized GPU timings. Reference fields
do not enter benchmark objectives. Compiled buffer estimates are not total
device/allocator memory; both figures and input identities are recorded.
"""

import argparse
import hashlib
import json
from pathlib import Path
import runpy
import time
import zipfile

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np
import optax

from examples.fracture2d._legacy.coupling_diagnostics import atomic_json
from examples.fracture2d._legacy.fp64 import FP64A2Backend
from examples.fracture2d._legacy.losses import FractureResidual
from examples.fracture2d._legacy.split_horizon_model import build_model, read_cloud
from examples.fracture2d._legacy.split_pinn import SCHEMA, neural_state_root
from examples.fracture2d._legacy.split_pinn_gradients import (
    make_analytic_energy, make_analytic_matvec, make_padded_energy, make_padded_matvec,
)
from examples.fracture2d._legacy.stagger_checkpoint import read_checkpoint
from examples.fracture2d._legacy.train import _build_network, _nonnegative_damage


def relative_difference(actual, expected):
    actual, expected = map(np.asarray, (actual, expected))
    if not all(np.all(np.isfinite(x)) for x in (actual, expected)) or actual.shape != expected.shape:
        raise ValueError("benchmark comparison fields must match and be finite")
    norm = np.linalg.norm(expected)
    return {"absolute_inf": float(np.max(np.abs(actual - expected))),
            "relative_l2": float(np.linalg.norm(actual - expected) / norm) if norm > 0 else None}


def benchmark_kernel(objective, parameters, optimizer, *, warmup, repeats):
    state = optimizer.init(parameters)

    def step(p, s):
        value, gradient = jax.value_and_grad(objective)(p)
        updates, s = optimizer.update(gradient, s, p)
        return optax.apply_updates(p, updates), s, value, gradient

    start = time.perf_counter()
    compiled = jax.jit(step).lower(parameters, state).compile()
    compile_seconds = time.perf_counter() - start
    first = jax.block_until_ready(compiled(parameters, state))
    p, s = parameters, state
    for _ in range(warmup):
        p, s, _, _ = jax.block_until_ready(compiled(p, s))
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        p, s, value, gradient = jax.block_until_ready(compiled(p, s))
        times.append(time.perf_counter() - start)
        if not np.isfinite(float(value)):
            raise FloatingPointError("benchmark optimizer update became nonfinite")
    memory = compiled.memory_analysis()
    result = {"compile_seconds": compile_seconds, "mean_step_seconds": float(np.mean(times)),
              "median_step_seconds": float(np.median(times)), "minimum_step_seconds": min(times),
              "argument_bytes": memory.argument_size_in_bytes, "output_bytes": memory.output_size_in_bytes,
              "temporary_bytes": memory.temp_size_in_bytes, "alias_bytes": memory.alias_size_in_bytes,
              "generated_code_bytes": memory.generated_code_size_in_bytes,
              "first_loss": float(first[2]), "first_gradient_norm": float(np.linalg.norm(np.asarray(first[3])))}
    return result, first


def run(args):
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    if args.warmup < 1 or args.repeats < 2 or args.state_index < 1:
        raise ValueError("benchmark needs a positive checkpoint index/warmup and at least two measured updates")
    if args.output.exists() or not args.output.parent.is_dir():
        raise ValueError("benchmark output must be new with an existing parent")
    c = runpy.run_path(str(args.config))["get_config"]()
    device = jax.devices()[0]
    if device.platform != "gpu":
        raise RuntimeError("fine-cloud update timing requires GPU")
    start = time.perf_counter()
    model = build_model(read_cloud(c.data.cloud, c.data.cloud_sha256),
        delta_u=c.nl.delta_u, delta_d=c.nl.delta_d, grip_extent=c.geometry.grip_extent,
        h_grip=c.geometry.h_grip, visibility_tolerance=c.nl.visibility_tolerance,
        E=c.physical.E, nu=c.physical.nu, k_res=c.physical.k_res,
        rank_rtol=c.physical.rank_rtol, Gc=c.physical.Gc, l0=c.physical.l0)
    backend = FP64A2Backend(model.static, model.constraints, device)
    residual = FractureResidual(model, backend)
    build_seconds = time.perf_counter() - start
    q_net, d_net = (jax.tree.map(lambda x: x.astype(jnp.float64),
        _build_network(c, jax.random.PRNGKey(c.train.seed + offset), out_dim=dimension))
        for offset, dimension in ((0, 2), (1, 1)))
    optimizer = optax.adam(c.train.learning_rate)
    template = (q_net, optimizer.init(eqx.filter(q_net, eqx.is_inexact_array)),
                d_net, optimizer.init(eqx.filter(d_net, eqx.is_inexact_array)),
                jnp.zeros(model.n_dofs, dtype=jnp.float64),
                jnp.zeros(model.n_material, dtype=jnp.float64), jnp.zeros(model.n_material, dtype=jnp.float64))
    path = neural_state_root(args.root, args.state_index) / "state.zip"
    with zipfile.ZipFile(path) as archive:
        identity = json.loads(archive.read("metadata.json"))["identity"]
    if identity["model_digest"] != model.model_digest or identity["index"] != args.state_index:
        raise ValueError("benchmark checkpoint model/index differs")
    for name in ("arch", "physical", "nl", "geometry"):
        if identity["config"][name] != json.loads(c.to_json_best_effort())[name]:
            raise ValueError(f"benchmark checkpoint {name} differs")
    _, restored = read_checkpoint(path, identity, template, schema=SCHEMA)
    q_net, _, d_net, _, _, d_fixed, H_parent = restored
    points = jnp.asarray(model.geometry.coordinates)
    load = jnp.asarray((args.state_index + 1) * c.data.load_step, dtype=jnp.float64)
    gc0, gc1 = model.Gc / model.l0, model.Gc * model.l0
    energy_scale = c.physical.E * load**2 * float(np.sum(model.geometry.union_volumes))
    analytic_energy, analytic_D = make_analytic_energy(backend), make_analytic_matvec(model.material_operator)
    padded_energy, mechanics_layout = make_padded_energy(backend)
    padded_D, damage_layout = make_padded_matvec(model.material_operator)
    factor = backend._edge_factor(d_fixed)

    def displacement(net):
        base = jnp.stack((jnp.zeros_like(points[:, 1]), load * points[:, 1]), axis=1)
        return residual.lift((base + load * jax.vmap(net)(points)).reshape(-1), load)

    H_fixed = residual.evaluate(displacement(q_net), d_fixed, None, H_parent).trial_history
    damage_scale = 2 * jnp.sqrt(jnp.mean(H_fixed**2))
    if not np.isfinite(float(damage_scale)) or float(damage_scale) <= 0:
        raise ValueError("benchmark damage scale must be positive finite")
    report = {"checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "model_digest": model.model_digest, "cloud_sha256": c.data.cloud_sha256,
              "checkpoint_index": args.state_index, "trial_load": float(load), "build_seconds": build_seconds,
              "material_nodes": model.n_material, "mechanics_edges": model.static.n_edges,
              "damage_edges": len(model.geometry.material_damage_edges),
              "support_occurrences": len(backend.occurrence_particle),
              "warmup": args.warmup, "repeats": args.repeats, "jax_version": jax.__version__,
              "device": device.device_kind, "optimizer_state": "fresh Adam, identical between paired variants",
              "memory_scope": "compiled arguments/outputs/temporaries, not total captured constants or allocator memory",
              "mechanics_padded_layout": mechanics_layout, "damage_padded_layout": damage_layout,
              "kernels": {}}
    atomic_json(args.output, report)
    for field, network in (("mechanics", q_net), ("damage", d_net)):
        dynamic, static = eqx.partition(network, eqx.is_inexact_array)
        parameters, unravel = ravel_pytree(dynamic)
        first_outputs = {}
        for mode in ("autodiff", "analytic", "padded"):
            energy = {"autodiff": backend._energy_from_factor, "analytic": analytic_energy, "padded": padded_energy}[mode]
            D = {"autodiff": model.material_operator.deterministic_matvec, "analytic": analytic_D, "padded": padded_D}[mode]

            def objective(p):
                net = eqx.combine(unravel(p), static)
                if field == "mechanics":
                    return energy(factor, displacement(net)) / energy_scale
                raw = c.physical.E / gc0 * load**2 * jax.vmap(net)(points[:model.n_material])[:, 0]
                d = _nonnegative_damage(raw, c.physical.damage_floor)
                rho = gc0 * d + gc1 * D(d) + 2 * H_fixed * (d - 1)
                return jnp.mean((rho / damage_scale)**2)

            measurements, first = benchmark_kernel(objective, parameters, optimizer,
                warmup=args.warmup, repeats=args.repeats)
            report["kernels"][f"{field}_{mode}"] = measurements
            first_outputs[mode] = first
            atomic_json(args.output, report)
            print(json.dumps({"kernel": f"{field}_{mode}", **measurements}), flush=True)
        baseline = first_outputs["autodiff"]
        for mode in ("analytic", "padded"):
            candidate = first_outputs[mode]
            agreement = {"loss_absolute": abs(float(baseline[2]) - float(candidate[2])),
                         "gradient": relative_difference(candidate[3], baseline[3]),
                         "one_update_parameters": relative_difference(candidate[0], baseline[0])}
            report[f"{field}_{mode}_agreement"] = agreement
            if (agreement["loss_absolute"] > 1e-12 or agreement["gradient"]["relative_l2"] is None
                    or agreement["gradient"]["relative_l2"] > 1e-8
                    or agreement["one_update_parameters"]["absolute_inf"] > 1e-9):
                atomic_json(args.output, report)
                raise ValueError(f"{mode} {field} update does not meet declared equivalence: {agreement}")
            report[f"{field}_{mode}_speedup"] = (report["kernels"][f"{field}_autodiff"]["median_step_seconds"]
                                               / report["kernels"][f"{field}_{mode}"]["median_step_seconds"])
            atomic_json(args.output, report)
    report["complete"] = True
    atomic_json(args.output, report)
    print(json.dumps({name: value for name, value in report.items() if name != "kernels"}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--state-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, required=True)
    parser.add_argument("--repeats", type=int, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
