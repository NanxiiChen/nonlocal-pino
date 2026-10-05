"""Read-only fork from strict state 59 into explicitly non-strict PINN trials.

Warning-region direct fields/reactions are comparators only. Eligible trials
are chained by separate diagnostic snapshots; no post-59 strict accepted load
is ever written or inferred.
"""

import hashlib
import json
import os
from pathlib import Path
import zipfile

import equinox as eqx
import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

import jax.numpy as jnp
from ml_collections import ConfigDict
import numpy as np
import optax
from absl import app as _absl_app, flags as _absl_flags

from examples.fracture2d.continuation import (
    _canonical, _check_fork_source, _digest, _load_accepted,
    _source_root_for_state,
)
from examples.fracture2d.stagger import _validate_stagger, run as run_stagger
from examples.fracture2d.stagger_checkpoint import (
    SCHEMA_ACCEPTED, SCHEMA_DIAGNOSTIC, read_checkpoint,
)
from examples.fracture2d.pinn_paths import physical_pinn_path
from examples.fracture2d.train import _build_grid, _build_network, _reference


def _check_warning_endpoint_extension(requested, stored_text):
    """Bind all settings except a nondecreasing archived warning-region end."""
    stored = json.loads(stored_text)
    requested_values = json.loads(_canonical(requested))
    old_end = stored["data"]["end_state_index"]
    new_end = requested_values["data"]["end_state_index"]
    if (type(old_end) is not int or type(new_end) is not int
            or not 60 <= old_end <= new_end <= 75):
        raise ValueError("warning diagnostic endpoint may only increase within 60..75")
    requested_values["data"]["end_state_index"] = old_end
    if json.dumps(requested_values, sort_keys=True) != stored_text:
        raise ValueError("warning diagnostic settings other than the endpoint differ")
    return old_end


def _strict_parent(config, model):
    """Re-verify all 59 immutable accepted snapshots and their parent hashes."""
    source_root = physical_pinn_path(config.output.fork_from).resolve()
    source_file = source_root / "config.json"
    if not source_file.is_file():
        raise FileNotFoundError(f"strict source root has no config: {source_file}")
    source_config = ConfigDict(json.loads(source_file.read_text(encoding="utf-8")))
    _check_fork_source(config, source_config)
    if (physical_pinn_path(source_config.output.dir).resolve() != source_root
            or int(source_config.data.end_state_index) != 59):
        raise ValueError("warning trials require a completed strict-59 source root")
    parent_digest = None
    parent_state = None
    for index in range(1, 60):
        if index <= source_config.output.fork_after_state_index:
            ancestor = _source_root_for_state(
                source_config, source_config.output.fork_from, index
            )
        else:
            ancestor = source_root
        state_dir = ancestor / f"state_{index:04d}"
        path = state_dir / "config.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing strict accepted state config: {path}")
        state_config = ConfigDict(json.loads(path.read_text(encoding="utf-8")))
        _check_fork_source(config, state_config)
        if (int(state_config.data.state_index) != index
                or physical_pinn_path(state_config.output.dir).resolve() != state_dir.resolve()):
            raise ValueError(f"strict source state {index} has a mismatched identity")
        parent_state, parent_digest = _load_accepted(state_config, model, parent_digest)
    return parent_state, parent_digest


def _trial_template(config, model):
    q_net = _build_network(config, jax.random.PRNGKey(int(config.train.seed)), out_dim=2)
    d_net = _build_network(config, jax.random.PRNGKey(int(config.train.seed) + 1), out_dim=1)
    q_net = jax.tree.map(lambda leaf: leaf.astype(np.float64), q_net)
    d_net = jax.tree.map(lambda leaf: leaf.astype(np.float64), d_net)
    optimizer = optax.adam(float(config.train.learning_rate))
    return (
        q_net, optimizer.init(eqx.filter(q_net, eqx.is_inexact_array)),
        d_net, optimizer.init(eqx.filter(d_net, eqx.is_inexact_array)),
        jnp.zeros(model.n_dofs, jnp.float64),
        jnp.zeros(model.n_material, jnp.float64),
        jnp.zeros(model.n_material, jnp.float64),
    )


def _load_trial(config, model, parent_digest, parent_schema):
    path = physical_pinn_path(config.output.dir) / "diagnostic.ckpt.zip"
    if (physical_pinn_path(config.output.dir) / "config.json").read_text(encoding="utf-8") != _canonical(config):
        raise ValueError("warning trial stored numerical config differs")
    index = int(config.data.state_index)
    identity = {
        "config_sha256": hashlib.sha256(_canonical(config).encode()).hexdigest(),
        "cloud_sha256": str(config.data.cloud_sha256),
        "reference_sha256": str(config.data.reference_sha256),
        "direct_model_digest": model.model_digest,
        "precision": "float64",
        "warning_diagnostic": True,
        "parent_schema": parent_schema,
        "state_index": index,
        ("parent_accepted_sha256" if index == 60 else "parent_diagnostic_sha256"): parent_digest,
    }
    metadata, state = read_checkpoint(
        path, identity, _trial_template(config, model), schema=SCHEMA_DIAGNOSTIC,
    )
    if (metadata["phase"] != "complete"
            or metadata["sweep"] != int(config.stagger.sweeps)
            or not 1 <= metadata["step"] <= int(config.stagger.mechanics_steps)):
        raise ValueError("warning trial checkpoint is not a final completed sweep")
    q, d, H = map(np.asarray, state[4:])
    if (q.shape != (model.n_dofs,) or d.shape != (model.n_material,)
            or H.shape != (model.n_material,)
            or any(x.dtype != np.float64 or not np.all(np.isfinite(x)) for x in (q, d, H))):
        raise ValueError("warning trial fields have invalid shape, dtype or values")
    load = float(_reference(config, model, diagnostic_warning=True)[0])
    fixed = model.constraints.constrained_dofs
    if not np.array_equal(q[fixed], model.constraints.prescribed_vector(load)[fixed]):
        raise ValueError("warning trial violates hard displacement constraints")
    if (np.min(d) < -float(config.stagger.damage_range_atol)
            or np.max(d) > 1.0 or np.min(H) < 0.0):
        raise ValueError("warning trial violates damage/history bounds")
    report = json.loads((physical_pinn_path(config.output.dir) / "result.json").read_text(encoding="utf-8"))
    if (report["accepted"] or not report["diagnostic_only"]
            or not report["diagnostic_usable"] or not all(report["gates"].values())):
        raise ValueError("warning trial result is not a usable non-strict diagnostic")
    return state, _digest(path)


def run(config, *, resume=False):
    _validate_stagger(config)
    if config.train.dtype != "float64" or int(config.data.state_index) != 1:
        raise ValueError("warning continuation requires FP64 and an explicit state-1 root")
    end = config.data.end_state_index
    if (isinstance(end, (bool, np.bool_)) or not np.isfinite(end)
            or int(end) != end or not 60 <= end <= 75):
        raise ValueError("warning diagnostic end_state_index must be in 60..75")
    if (not isinstance(config.output.fork_from, str) or not config.output.fork_from
            or config.output.fork_after_state_index != 59):
        raise ValueError("warning diagnostic must fork exactly after strict state 59")
    root = physical_pinn_path(config.output.dir)
    if root.resolve() == physical_pinn_path(config.output.fork_from).resolve():
        raise ValueError("warning diagnostics require a new, separate output root")
    model = _build_grid(config)
    committed, parent_digest = _strict_parent(config, model)
    if resume:
        if not root.is_dir():
            raise FileNotFoundError(f"warning diagnostic root is missing: {root}")
        previous_end = _check_warning_endpoint_extension(
            config, (root / "config.json").read_text(encoding="utf-8")
        )
        if previous_end < end:
            temporary = root / "config.json.tmp"
            if temporary.exists():
                raise FileExistsError(f"incomplete warning root update: {temporary}")
            temporary.write_text(_canonical(config), encoding="utf-8")
            os.replace(temporary, root / "config.json")
    else:
        if not root.parent.is_dir():
            raise FileNotFoundError(f"warning diagnostic root parent is missing: {root.parent}")
        root.mkdir()
        (root / "config.json").write_text(_canonical(config), encoding="utf-8")
    summary = []
    parent_state = committed
    for index in range(60, int(end) + 1):
        state_config = ConfigDict(json.loads(config.to_json_best_effort()))
        state_config.data.state_index = index
        state_config.output.dir = str(Path(config.output.dir) / f"state_{index:04d}")
        state_config.output.tensorboard_root = str(
            Path(config.output.tensorboard_root) / root.name
        )
        state_config.output.resume = False
        state_dir = physical_pinn_path(state_config.output.dir)
        stored_file = state_dir / "config.json"
        if stored_file.is_file():
            old_end = _check_warning_endpoint_extension(
                state_config, stored_file.read_text(encoding="utf-8")
            )
            if old_end < index:
                raise ValueError(f"warning load {index} exceeds its stored endpoint")
            state_config.data.end_state_index = old_end
        _reference(state_config, model, diagnostic_warning=True)
        parent_schema = SCHEMA_ACCEPTED if index == 60 else SCHEMA_DIAGNOSTIC
        trial_path = state_dir / "diagnostic.ckpt.zip"
        if not trial_path.is_file():
            checkpoint = state_dir / "latest.ckpt.zip"
            if state_dir.exists() and not checkpoint.is_file():
                raise RuntimeError(f"warning load {index} has no checkpoint to resume: {state_dir}")
            if checkpoint.is_file():
                with zipfile.ZipFile(checkpoint) as archive:
                    metadata = json.loads(archive.read("metadata.json"))
                if metadata["phase"] == "complete":
                    raise RuntimeError(f"warning load {index} completed without a usable trial")
            result, trial_state = run_stagger(
                state_config, resume=state_dir.exists(),
                initial_state=committed, parent_sha256=parent_digest,
                return_state=True, warning_diagnostic=True,
                parent_schema=parent_schema,
            )
            report_path = state_dir / "result.json"
            temporary = report_path.with_name(report_path.name + ".tmp")
            temporary.write_text(json.dumps(result, sort_keys=True), encoding="utf-8")
            os.replace(temporary, report_path)
            if trial_state is None or not result["diagnostic_usable"] or result["accepted"]:
                raise RuntimeError(
                    f"warning load {index} stopped at physical gates: {result['gates']}"
                )
        committed, parent_digest = _load_trial(
            state_config, model, parent_digest, parent_schema,
        )
        if index > 60:
            previous_d, previous_H = map(np.asarray, parent_state[5:])
            new_d, new_H = map(np.asarray, committed[5:])
            if np.min(new_d - previous_d) < -float(config.stagger.healing_atol):
                raise ValueError("warning diagnostic exceeds bounded healing")
            if np.min(new_H - previous_H) < 0.0:
                raise ValueError("warning diagnostic decreased cumulative Amor history")
        parent_state = committed
        summary.append({"state_index": index, "diagnostic_sha256": parent_digest})
        print(json.dumps({"event": "warning_trial_usable", **summary[-1]}, sort_keys=True), flush=True)
    return summary


def main(_):
    config = _absl_flags.FLAGS.config
    run(config, resume=bool(config.output.resume))


if __name__ == "__main__":
    _absl_app.run(main)
