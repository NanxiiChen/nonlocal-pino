"""Isolated late-time accuracy pilots for the declared six-neighbour graph.

Run reference/check on CPU with JAX_ENABLE_X64=1; train uses the selected JAX
backend. References are used only for evaluation, never as training labels.
"""

import argparse
import importlib.util
import json
import time
from decimal import Decimal, localcontext
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from nonlocal_pinn.nn.arch import make_pinn
from nonlocal_pinn.training.optimizers import make_optimizer


_METHODS = ("phase_soft", "latent_phase", "latent_balanced", "latent_curriculum",
            "ic_phase", "ic_balanced", "ic_curriculum")
_IC_ONLY = ("ic_phase", "ic_balanced", "ic_curriculum")
_TRANSFORMED = ("latent_balanced", "latent_curriculum", "ic_balanced", "ic_curriculum")


def _load_config(path):
    spec = importlib.util.spec_from_file_location("accuracy_config", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_config()


def _parameters(c):
    n = c.sample.Nx
    if not (n == c.sample.Ny == c.sample.Nz and n >= 4 and n % 2 == 0):
        raise ValueError("Pilot requires an even cubic grid with at least 4 points")
    h = 1.0 / (n - 1)
    if not (h < c.nl.delta < np.sqrt(2.0) * h):
        raise ValueError("Pilot implements exactly the six-neighbour horizon")
    if c.nl.method != "sparse" or c.nl.kernel != "cubic":
        raise ValueError("Pilot requires the declared sparse cubic operator")
    physical = [c.physical.M, c.physical.eps_star, c.physical.lam,
                c.physical.R0, c.nondim.T_c, c.nondim.L_c, c.nl.delta]
    if not np.all(np.isfinite(physical)):
        raise ValueError("Pilot physical parameters and scales must be finite")
    if c.physical.M <= 0 or c.physical.eps_star <= 0:
        raise ValueError("M and eps_star must be positive")
    if c.nondim.T_c <= 0 or c.nondim.L_c <= 0 or c.physical.lam <= 0:
        raise ValueError("Time, length scales and melting speed must be positive")
    if tuple(c.nondim.xc) != (0.5, 0.5, 0.5):
        raise ValueError("Octant reference requires the centred sphere")
    if not 0 < c.physical.R0 / c.nondim.L_c < 0.5:
        raise ValueError("Initial sphere must lie strictly inside the unit cube")
    d = np.sqrt(2.0) * c.physical.eps_star
    a = c.physical.M * c.nondim.T_c / c.nondim.L_c**2
    b = a / c.physical.eps_star**2
    velocity = c.physical.lam * c.nondim.T_c / c.nondim.L_c
    g = velocity / d
    w = 315.0 / (2 * np.pi * c.nl.delta**5) * (1 - h / c.nl.delta)**2 * h**3
    return n, h, d, a, b, g, velocity, c.physical.R0 / c.nondim.L_c, w


def _identity(c):
    identity = {
        "physical": c.physical.to_dict(), "nondim": c.nondim.to_dict(),
        "nl": c.nl.to_dict(), "grid": [c.sample.Nx, c.sample.Ny, c.sample.Nz],
    }
    return json.loads(json.dumps(identity))


def _write_json(path, value):
    with path.open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def _direct_reference(c, dt):
    n, h, d, a, b, g, _, radius, w = _parameters(c)
    count = c.experiment.reference_snapshots
    if count < 2:
        raise ValueError("Need at least two reference snapshots")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("reference_dt must be finite and positive")
    interval = 1.0 / (count - 1)
    steps = round(interval / dt)
    if steps < 1 or abs(steps * dt - interval) > 1e-12:
        raise ValueError("reference_dt must divide the snapshot interval")
    axis = (np.arange(n // 2) + 0.5) * h
    r = np.sqrt(sum(v**2 for v in np.meshgrid(axis, axis, axis, indexing="ij")))
    u0 = jnp.asarray(np.tanh((radius - r) / d), dtype=jnp.float64)

    def rhs(u):
        p = jnp.pad(u, 1, mode="edge")
        lap = (p[:-2, 1:-1, 1:-1] + p[2:, 1:-1, 1:-1]
               + p[1:-1, :-2, 1:-1] + p[1:-1, 2:, 1:-1]
               + p[1:-1, 1:-1, :-2] + p[1:-1, 1:-1, 2:] - 6 * u)
        return a * w * lap - b * (u**3 - u) - g * (1 - u * u)

    @jax.jit
    def integrate(u):
        def step(u, _):
            k1 = rhs(u)
            k2 = rhs(u + dt * k1 / 2)
            k3 = rhs(u + dt * k2 / 2)
            k4 = rhs(u + dt * k3)
            return u + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4), None

        def interval_step(u, _):
            u, _ = jax.lax.scan(step, u, None, length=steps)
            return u, u

        return jax.lax.scan(interval_step, u, None, length=count - 1)[1]

    fields = np.concatenate([np.asarray(u0)[None], np.asarray(integrate(u0))])
    if not np.all(np.isfinite(fields)):
        raise FloatingPointError("Nonfinite direct reference")
    return axis, np.linspace(0, 1, count), fields


def reference(c, output):
    if jax.default_backend() != "cpu" or not jax.config.jax_enable_x64:
        raise ValueError("Run reference with JAX_PLATFORMS=cpu JAX_ENABLE_X64=1")
    paths = [output / "reference.npz", output / "reference_half_dt.npz"]
    if any(p.exists() for p in paths):
        raise FileExistsError("Reference outputs already exist; choose a new output directory")
    fields = []
    for path, dt in zip(paths, [c.experiment.reference_dt, c.experiment.reference_dt / 2]):
        axis, times, phi = _direct_reference(c, dt)
        np.savez_compressed(path, axis=axis, time=times, phi=phi,
                            identity=json.dumps(_identity(c)), dt=dt)
        fields.append(phi)
        print(f"reference dt={dt}: phase range [{phi.min():.9g}, {phi.max():.9g}]", flush=True)
    report = {"max_field_dt_difference": float(np.max(np.abs(fields[0] - fields[1]))),
              "identity": _identity(c), "reference_dt": c.experiment.reference_dt}
    _write_json(output / "reference_check.json", report)
    print(json.dumps(report), flush=True)


def _ratio(ui, uj):
    """(tanh(ui)-tanh(uj))/sech(ui)^2, without subtracting saturated tanh."""
    log_ratio = jnp.logaddexp(ui, -ui) - jnp.logaddexp(uj, -uj)
    return jnp.sinh(ui - uj) * jnp.exp(log_ratio)


def check(c, output):
    if not jax.config.jax_enable_x64:
        raise ValueError("Run check with JAX_ENABLE_X64=1")
    _, _, _, a, b, g, _, _, w = _parameters(c)
    rng = np.random.default_rng(c.experiment.seed)
    u = rng.uniform(-4, 4, (100, 7))
    ut = rng.normal(size=100)
    phi = np.tanh(u)
    original = ((1 - phi[:, 0]**2) * ut
                + a * w * np.sum(phi[:, :1] - phi[:, 1:], axis=1)
                + b * (phi[:, 0]**3 - phi[:, 0]) + g * (1 - phi[:, 0]**2))
    transformed = (ut + a * w * np.sum(np.asarray(_ratio(u[:, :1], u[:, 1:])), axis=1)
                   - b * phi[:, 0] + g)
    err = float(np.max(np.abs(original - (1 - phi[:, 0]**2) * transformed)))
    if err > 1e-10:
        raise AssertionError(f"Residual transformation error: {err}")
    tail_errors = []
    with localcontext() as ctx:
        ctx.prec = 80
        for ui in [-30, -20, 0, 20, 30]:
            uj = ui + 0.2
            x, y = Decimal(str(ui)), Decimal(str(uj))
            ex, ey = (2 * x).exp(), (2 * y).exp()
            tx, ty = (ex - 1) / (ex + 1), (ey - 1) / (ey + 1)
            truth = float((tx - ty) / (1 - tx * tx))
            value = float(_ratio(jnp.array(ui, dtype=jnp.float64), jnp.array(uj)))
            tail_errors.append(abs(value - truth))
    if max(tail_errors) > 1e-12:
        raise AssertionError("Stable ratio failed independent high-precision check")
    # A nonzero network verifies hard IC structurally, rather than only because
    # the pilot's last layer starts at zero.
    net = _network(c, jax.random.PRNGKey(c.experiment.seed), latent=False)
    points = jnp.asarray(rng.uniform(0, 1, (100, 3)))
    _, _, d, _, _, _, _, radius, _ = _parameters(c)
    target = jnp.tanh((radius - jnp.linalg.norm(points - 0.5, axis=1)) / d)
    ic_error = 0.0
    for method in ("latent_phase",) + _IC_ONLY:
        phi0 = _field(net, points, jnp.zeros(1), c, method)[0][:, 0]
        ic_error = max(ic_error, float(jnp.max(jnp.abs(phi0 - target))))
    if ic_error != 0:
        raise AssertionError(f"Hard initial condition error: {ic_error}")
    times = jnp.array([0.0, 0.3, 1.0])
    changed_speed = c.copy_and_resolve_references()
    changed_speed.physical.lam = 2 * c.physical.lam
    zero_net = _network(c, jax.random.PRNGKey(c.experiment.seed), latent=True)
    speed_error = stationary_error = 0.0
    for method in _IC_ONLY:
        field = _field(net, points, times, c, method)[0]
        changed = _field(net, points, times, changed_speed, method)[0]
        speed_error = max(speed_error, float(jnp.max(jnp.abs(field - changed))))
        initial = _field(zero_net, points, times, c, method)[0]
        stationary_error = max(stationary_error, float(jnp.max(jnp.abs(initial - target[:, None]))))
    if speed_error != 0 or stationary_error != 0:
        raise AssertionError("IC-only trial function contains an unintended motion prior")
    report = {"residual_equivalence_max_abs": err,
              "ratio_decimal80_max_abs": max(tail_errors),
              "hard_ic_max_abs": ic_error,
              "ic_only_trial_speed_dependence": speed_error,
              "ic_only_untrained_motion": stationary_error,
              "time_map": {"physical_endpoint_s": c.nondim.T_c, "normalized_endpoint": 1.0}}
    _write_json(output / "derivation_check.json", report)
    print(json.dumps(report), flush=True)


def _network(c, key, latent):
    p = c.arch
    if p.type != "mlp" or p.out_dim != 1 or p.use_hard_constraint:
        raise ValueError("Pilot requires a scalar raw MLP; constraints are defined explicitly here")
    net = make_pinn(spatial_dim=3, out_dim=1, hidden_dim=p.hidden_dim,
                    num_layers=p.num_layers, activation=p.activation,
                    use_spatial_emb=p.use_spatial_emb, use_temporal_emb=p.use_temporal_emb,
                    spatial_emb_dim=p.spatial_emb_dim, spatial_scale=p.spatial_scale,
                    temporal_emb_dim=p.temporal_emb_dim, temporal_scale=p.temporal_scale,
                    tensor_split=p.tensor_split, key=key)
    if latent:
        net = eqx.tree_at(lambda n: (n.output.weight, n.output.bias), net,
                          (jnp.zeros_like(net.output.weight), jnp.zeros_like(net.output.bias)))
    return net


def _raw_and_dt(net, points, times):
    def at_times(ts):
        return jax.vmap(lambda t: jax.vmap(net, in_axes=(0, None))(points, t)[:, 0])(ts)
    raw, raw_t = jax.jvp(at_times, (times,), (jnp.ones_like(times),))
    return raw.T, raw_t.T


def _field(net, points, times, c, method):
    raw, raw_t = _raw_and_dt(net, points, times)
    if method == "phase_soft":
        return raw, raw_t, raw, raw_t
    _, _, d, _, _, _, velocity, radius, _ = _parameters(c)
    r = jnp.linalg.norm(points - jnp.asarray(c.nondim.xc), axis=1)
    if method in _IC_ONLY:
        u = (radius - r[:, None]) / d + times[None, :] * raw
        u_t = raw + times[None, :] * raw_t
    else:
        u = (radius - r[:, None] - velocity * times[None, :]) / d + times[None, :] * raw
        u_t = -velocity / d + raw + times[None, :] * raw_t
    phi = jnp.tanh(u)
    return phi, (1 - phi * phi) * u_t, u, u_t


def _batch(key, c):
    n, h, *_ = _parameters(c)
    ijk = jax.random.randint(key, (c.sample.K_core, 3), 0, n)
    offsets = jnp.concatenate([jnp.eye(3, dtype=jnp.int32), -jnp.eye(3, dtype=jnp.int32)])
    neighbours = ijk[:, None, :] + offsets[None, :, :]
    mask = jnp.all((neighbours >= 0) & (neighbours < n), axis=2)
    neighbours = jnp.clip(neighbours, 0, n - 1)
    indices = jnp.concatenate([ijk[:, None, :], neighbours], axis=1)
    return indices.reshape(-1, 3).astype(jnp.float32) * h, mask


def _pde(net, points, mask, times, c, method):
    _, _, _, a, b, g, _, _, w = _parameters(c)
    phi, phi_t, u, u_t = _field(net, points, times, c, method)
    phi = phi.reshape(-1, 7, len(times))
    if method in _TRANSFORMED:
        u = u.reshape(-1, 7, len(times))
        ratios = _ratio(u[:, :1], u[:, 1:])
        D_ratio = w * jnp.sum(ratios * mask[..., None], axis=1)
        residual = u_t.reshape(-1, 7, len(times))[:, 0] + a * D_ratio - b * phi[:, 0] + g
    else:
        D_phi = w * jnp.sum((phi[:, :1] - phi[:, 1:]) * mask[..., None], axis=1)
        center = phi[:, 0]
        residual = (phi_t.reshape(-1, 7, len(times))[:, 0] + a * D_phi
                    + b * (center**3 - center) + g * (1 - center * center))
    return jnp.mean(jnp.abs(residual))


def _crossing(axis, phi):
    k = np.flatnonzero((phi[:-1] > 0) & (phi[1:] <= 0))
    if not len(k):
        return (0.0, None) if not np.any(phi > 0) else (None, None)
    k = k[-1]
    slope = (phi[k + 1] - phi[k]) / (axis[k + 1] - axis[k])
    return float(axis[k] - phi[k] / slope), float(-1 / slope)


def evaluate(net, c, method, ref):
    """Full grid, all faces included; stream inference in fixed-size chunks."""
    n, h, _, a, b, g, _, _, w = _parameters(c)
    snapshot_indices = np.linspace(0, len(ref["time"]) - 1, 6).astype(int)
    if not np.allclose(ref["time"][snapshot_indices], np.linspace(0, 1, 6)):
        raise ValueError("Reference must include each of the six evaluation times")
    times = jnp.asarray(ref["time"][snapshot_indices], dtype=jnp.float32)
    chunk = c.experiment.eval_chunk
    if n**3 % chunk:
        raise ValueError("eval_chunk must divide the full grid size for fixed shapes")
    fn = eqx.filter_jit(lambda net, x: _field(net, x, times, c, method)[:2])
    pred = np.empty((6, n**3), dtype=np.float32)
    pred_t = np.empty_like(pred)
    for start in range(0, n**3, chunk):
        indices = np.stack(np.unravel_index(np.arange(start, start + chunk), (n, n, n)), axis=1)
        x = jnp.asarray(indices * h, dtype=jnp.float32)
        phi, phi_t = fn(net, x)
        pred[:, start:start + chunk] = np.asarray(phi).T
        pred_t[:, start:start + chunk] = np.asarray(phi_t).T
    mirror = (np.abs(np.arange(n) - (n - 1) / 2) - 0.5).astype(int)
    axis = ref["axis"]
    records = []
    for j, index in enumerate(snapshot_indices):
        p = pred[j].reshape(n, n, n).astype(np.float64)
        pt = pred_t[j].reshape(n, n, n).astype(np.float64)
        truth = ref["phi"][index][np.ix_(mirror, mirror, mirror)]
        error = p - truth
        padded = np.pad(p, 1, mode="edge")
        Dp = w * (6 * p - padded[:-2, 1:-1, 1:-1] - padded[2:, 1:-1, 1:-1]
                  - padded[1:-1, :-2, 1:-1] - padded[1:-1, 2:, 1:-1]
                  - padded[1:-1, 1:-1, :-2] - padded[1:-1, 1:-1, 2:])
        residual = pt + a * Dp + b * (p**3 - p) + g * (1 - p * p)
        iface = np.abs(truth) < 0.9
        near_axis = p[n // 2:, n // 2, n // 2]
        radius, width = _crossing(axis, near_axis)
        ref_radius, ref_width = _crossing(axis, ref["phi"][index, :, 0, 0])
        ref_volume = float(np.mean(truth > 0))
        records.append({
            "time_star": float(ref["time"][index]),
            "physical_time_s": float(ref["time"][index] * c.nondim.T_c),
            "rel_l2": float(np.linalg.norm(error) / np.linalg.norm(truth)),
            "interface_mae": float(np.mean(np.abs(error[iface]))),
            "phase_max_error": float(np.max(np.abs(error))),
            "radius": radius, "reference_radius": ref_radius,
            "width": width, "reference_width": ref_width,
            "positive_fraction": float(np.mean(p > 0)),
            "reference_positive_fraction": ref_volume,
            "volume_relative_error": float(np.mean(p > 0) / ref_volume - 1),
            "pde_mae_full_grid": float(np.mean(np.abs(residual))),
            "pde_mae_reference_interface": float(np.mean(np.abs(residual[iface]))),
            "phase_min": float(p.min()), "phase_max": float(p.max()),
        })
    return records


def train(c, output, selected_method):
    _parameters(c)
    if min(c.sample.K_core, c.sample.Nt, c.sample.N_ic, c.experiment.log_every,
           c.experiment.curriculum_stages, c.experiment.eval_chunk) < 1:
        raise ValueError("Batch sizes, chunk size and log/curriculum counts must be positive")
    if c.train.iterations < c.experiment.curriculum_stages:
        raise ValueError("Need at least one step per curriculum stage")
    if c.sample.t_min <= 0 or c.sample.t_min >= 1 / c.experiment.curriculum_stages:
        raise ValueError("Invalid t_min for the curriculum")
    if tuple(c.loss.gradnorm_bias) != (1.0, 1.0):
        raise ValueError("This pilot explicitly compares unit GradNorm biases")
    jax.config.update("jax_default_matmul_precision", c.train.matmul_precision)
    with np.load(output / "reference.npz", allow_pickle=False) as f:
        ref = {key: f[key] for key in f.files}
    if json.loads(str(ref["identity"])) != _identity(c):
        raise ValueError("Reference parameters do not match the pilot")
    methods = [selected_method] if selected_method is not None else list(c.experiment.methods)
    for method in methods:
        if method not in _METHODS or method not in c.experiment.methods:
            raise ValueError(f"Method is absent from the explicit configuration: {method}")
        directory = output / method
        directory.mkdir(exist_ok=False)
        _write_json(directory / "config.json", json.loads(c.to_json_best_effort()))
        (directory / "source.py").write_text(Path(__file__).read_text())
        repo = Path(__file__).resolve().parents[2]
        sources = list((repo / "src" / "nonlocal_pinn").rglob("*.py"))
        sources += list((repo / "examples" / "icemelting3d" / "configs").glob("*.py"))
        for source in sources:
            snapshot = directory / "source" / source.relative_to(repo)
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            snapshot.write_bytes(source.read_bytes())
        key = jax.random.PRNGKey(c.experiment.seed)
        k_net, k_ic = jax.random.split(key)
        net = _network(c, k_net, method != "phase_soft")
        n, h, d, _, _, _, _, radius, _ = _parameters(c)
        x_ic = jax.random.randint(k_ic, (c.sample.N_ic, 3), 0, n).astype(jnp.float32) * h
        target_ic = jnp.tanh((radius - jnp.linalg.norm(x_ic - 0.5, axis=1)) / d)
        optimizer = make_optimizer(c)
        state = optimizer.init(eqx.filter(net, eqx.is_array))
        weights = jnp.ones(2)

        def ic_loss(net):
            prediction = jax.vmap(net, in_axes=(0, None))(x_ic, jnp.array(0.0))[:, 0]
            return jnp.mean((prediction - target_ic)**2)

        @eqx.filter_jit
        def step(net, state, weights, key_x, key_t, horizon):
            points, mask = _batch(key_x, c)
            times = jax.random.uniform(key_t, (c.sample.Nt,), minval=c.sample.t_min, maxval=horizon)
            pde, gp = eqx.filter_value_and_grad(_pde)(net, points, mask, times, c, method)
            if method == "phase_soft":
                ic, gi = eqx.filter_value_and_grad(ic_loss)(net)
                norms = jnp.array([jnp.sqrt(sum(jnp.sum(x*x) for x in jax.tree.leaves(grad)))
                                   for grad in (gp, gi)])
                raw_weights = jnp.where(norms > 1e-12, jnp.mean(norms) / norms, 1.0)
                raw_weights = jnp.clip(raw_weights, 0, c.loss.gradnorm_weight_cap)
                weights = c.train.alpha_gradnorm * raw_weights + (1-c.train.alpha_gradnorm) * weights
                grad = jax.tree.map(lambda x, y: weights[0]*x + weights[1]*y, gp, gi)
            else:
                ic = jnp.array(0.0)
                grad = gp
            updates, state = optimizer.update(grad, state, net)
            return eqx.apply_updates(net, updates), state, weights, pde, ic

        initial = evaluate(net, c, method, ref)
        _write_json(directory / "initial_metrics.json", initial)
        start = time.monotonic()
        history = []
        for iteration in range(c.train.iterations):
            key_x, key_t = jax.random.split(jax.random.fold_in(key, iteration + 1000))
            if method in ("latent_curriculum", "ic_curriculum"):
                stage = min(c.experiment.curriculum_stages,
                            iteration * c.experiment.curriculum_stages // c.train.iterations + 1)
                horizon = stage / c.experiment.curriculum_stages
            else:
                horizon = 1.0
            net, state, weights, pde, ic = step(net, state, weights, key_x, key_t,
                                               jnp.array(horizon, dtype=jnp.float32))
            if iteration % c.experiment.log_every == 0 or iteration == c.train.iterations - 1:
                values = [float(pde), float(ic)]
                if not np.all(np.isfinite(values)):
                    raise FloatingPointError(f"Nonfinite loss in {method} at {iteration}")
                row = {"step": iteration + 1, "pde": values[0], "ic": values[1],
                       "horizon": horizon, "elapsed_s": time.monotonic() - start}
                history.append(row)
                print(method, json.dumps(row), flush=True)
        final = evaluate(net, c, method, ref)
        eqx.tree_serialise_leaves(directory / "model_final.eqx", net)
        report = {"method": method, "backend": jax.default_backend(),
                  "steps": c.train.iterations, "history": history,
                  "initial": initial, "final": final,
                  "reference_path": str((output / "reference.npz").resolve())}
        _write_json(directory / "metrics.json", report)
        print(method, "terminal", json.dumps(final[-1]), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["reference", "check", "train"])
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).parent / "configs" / "accuracy_weak_prior.py")
    parser.add_argument("--method", choices=_METHODS)
    args = parser.parse_args()
    c = _load_config(args.config)
    output = Path(c.experiment.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if args.action == "reference":
        reference(c, output)
    elif args.action == "check":
        check(c, output)
    else:
        train(c, output, args.method)


if __name__ == "__main__":
    main()
