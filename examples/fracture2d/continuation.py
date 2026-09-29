"""Strict-prefix, accepted-state-only continuation of the FP64 stagger.

Every load has its own identity-bound in-progress checkpoint and an atomic
accepted-state checkpoint. Later loads start from the latter, never from the
archived current-load solution. A rejected load leaves all earlier accepted
checkpoints intact and cannot be skipped on resume.
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
import numpy as np
import optax
from absl import app as _absl_app, flags as _absl_flags
from ml_collections import ConfigDict

from examples.fracture2d.stagger import _validate_stagger, run as run_stagger
from examples.fracture2d.stagger_checkpoint import SCHEMA_ACCEPTED, read_checkpoint
from examples.fracture2d.train import _build_grid, _build_network, _reference


def _canonical(config):
    values = json.loads(config.to_json_best_effort())
    values["output"]["resume"] = False
    return json.dumps(values, sort_keys=True)


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _check_endpoint_extension(requested, stored_text):
    """Verify every setting except an explicit nondecreasing endpoint."""
    stored = json.loads(stored_text)
    requested_values = json.loads(_canonical(requested))
    old_end = stored["data"]["end_state_index"]
    new_end = requested_values["data"]["end_state_index"]
    if (type(old_end) is not int or not 1 <= old_end <= 59
            or type(new_end) is not int or not old_end <= new_end <= 59):
        raise ValueError("continuation endpoint may only increase within the strict prefix")
    requested_values["data"]["end_state_index"] = old_end
    if json.dumps(requested_values, sort_keys=True) != stored_text:
        raise ValueError("continuation numerical settings other than the endpoint differ")
    return old_end


def _check_fork_source(requested, source):
    """Allow only the approved scale and acceptance-policy differences."""
    if (source.stagger.mechanics_scale_policy not in (
            "first_load_initial", "first_load_times_load_ratio")
            or requested.stagger.mechanics_scale_policy != "first_load_times_load_ratio"):
        raise ValueError("fork requires first-load or load-ratio source and load-ratio target policies")
    old = json.loads(_canonical(source))
    new = json.loads(_canonical(requested))
    for values in (old, new):
        values["data"]["end_state_index"] = 0
        values["data"]["state_index"] = 1
        values["stagger"]["mechanics_scale_policy"] = "first_load_initial"
        values["output"]["dir"] = ""
        values["output"]["tensorboard_root"] = ""
        values["output"].pop("fork_from", None)
        values["output"].pop("fork_after_state_index", None)
        values["stagger"].pop("gate_displacement_l2", None)
        values["stagger"].pop("damage_rmse_limit", None)
    if old != new:
        raise ValueError("fork source differs from target outside approved scale/gate settings")


def _source_root_for_state(requested, source_root, index):
    """Resolve nested, read-only fork ancestry for a previously accepted load."""
    root = Path(source_root).resolve()
    visited = set()
    while True:
        if root in visited:
            raise ValueError("continuation fork ancestry contains a cycle")
        visited.add(root)
        path = root / "config.json"
        if not path.is_file():
            raise FileNotFoundError(f"fork ancestor has no root config: {path}")
        ancestor = ConfigDict(json.loads(path.read_text(encoding="utf-8")))
        _check_fork_source(requested, ancestor)
        if (Path(ancestor.output.dir).resolve() != root
                or int(ancestor.data.end_state_index) < index):
            raise ValueError("fork ancestor root identity or endpoint differs")
        if "fork_from" not in ancestor.output or not ancestor.output.fork_from:
            return root
        cutoff = ancestor.output.fork_after_state_index
        if (isinstance(cutoff, (bool, np.bool_)) or not np.isfinite(cutoff)
                or int(cutoff) != cutoff or not 1 <= cutoff < int(ancestor.data.end_state_index)):
            raise ValueError("fork ancestor has an invalid prefix cutoff")
        if index > cutoff:
            return root
        root = Path(ancestor.output.fork_from).resolve()


def _load_accepted(config, model, parent_digest):
    path = Path(config.output.dir) / "accepted.ckpt.zip"
    if (Path(config.output.dir) / "config.json").read_text(encoding="utf-8") != _canonical(config):
        raise ValueError(f"stored numerical config differs at load {config.data.state_index}")
    q_net = _build_network(config, jax.random.PRNGKey(int(config.train.seed)), out_dim=2)
    d_net = _build_network(config, jax.random.PRNGKey(int(config.train.seed) + 1), out_dim=1)
    q_net = jax.tree.map(lambda leaf: leaf.astype(np.float64), q_net)
    d_net = jax.tree.map(lambda leaf: leaf.astype(np.float64), d_net)
    opt = optax.adam(float(config.train.learning_rate))
    template = (
        q_net, opt.init(eqx.filter(q_net, eqx.is_inexact_array)),
        d_net, opt.init(eqx.filter(d_net, eqx.is_inexact_array)),
        jnp.zeros(model.n_dofs, jnp.float64),
        jnp.zeros(model.n_material, jnp.float64),
        jnp.zeros(model.n_material, jnp.float64),
    )
    identity = {
        "config_sha256": hashlib.sha256(_canonical(config).encode()).hexdigest(),
        "cloud_sha256": str(config.data.cloud_sha256),
        "reference_sha256": str(config.data.reference_sha256),
        "direct_model_digest": model.model_digest,
        "precision": "float64",
        "state_index": int(config.data.state_index),
    }
    if parent_digest is not None:
        identity["parent_accepted_sha256"] = parent_digest
    metadata, state = read_checkpoint(path, identity, template, schema=SCHEMA_ACCEPTED)
    if (metadata["phase"] != "complete"
            or metadata["sweep"] != int(config.stagger.sweeps)
            or not 1 <= metadata["step"] <= int(config.stagger.mechanics_steps)):
        raise ValueError("accepted checkpoint is not the final completed sweep")
    q, d, H = map(np.asarray, state[4:])
    if (q.shape != (model.n_dofs,) or d.shape != (model.n_material,)
            or H.shape != (model.n_material,)
            or any(x.dtype != np.float64 or not np.all(np.isfinite(x)) for x in (q, d, H))):
        raise ValueError("accepted checkpoint fields have invalid shape, dtype or values")
    load, *_ = _reference(config, model)
    fixed = model.constraints.constrained_dofs
    if not np.array_equal(q[fixed], model.constraints.prescribed_vector(float(load))[fixed]):
        raise ValueError("accepted checkpoint violates load constraints")
    if np.min(d) < -float(config.stagger.damage_range_atol) or np.max(d) > 1.0 or np.min(H) < 0:
        raise ValueError("accepted checkpoint violates damage/history bounds")
    return state, _digest(path)


def run(config, *, resume=False):
    """Run loads 1..end_state_index, stopping before any rejected successor."""
    _validate_stagger(config)
    if config.train.dtype != "float64":
        raise ValueError("strict continuation requires the adopted float64 graph")
    if int(config.data.state_index) != 1:
        raise ValueError("strict continuation must start at state 1 (zero-load parent)")
    end = config.data.end_state_index
    if (isinstance(end, (bool, np.bool_)) or not np.isfinite(end)
            or int(end) != end or not (1 <= end <= 59)):
        raise ValueError("data.end_state_index must be a strict prefix index in 1..59")
    root = Path(config.output.dir)
    fork_path = config.output.fork_from
    fork_after = config.output.fork_after_state_index
    if not isinstance(fork_path, str):
        raise ValueError("output.fork_from must be an explicit path or empty string")
    if (isinstance(fork_after, (bool, np.bool_)) or not np.isfinite(fork_after)
            or int(fork_after) != fork_after
            or (fork_path and not 1 <= fork_after < end)
            or (not fork_path and fork_after != 0)):
        raise ValueError("output.fork_after_state_index must name an accepted prefix before the endpoint")
    if fork_path:
        source_root = Path(fork_path).resolve()
        if (source_root == root.resolve() or not source_root.is_dir()
                or not (source_root / "config.json").is_file()):
            raise ValueError("fork source must be a separate existing continuation root")
        _source_root_for_state(config, source_root, int(fork_after))
    if resume:
        if not root.is_dir():
            raise FileNotFoundError(f"continuation root is missing: {root}")
        previous_end = _check_endpoint_extension(
            config, (root / "config.json").read_text(encoding="utf-8")
        )
        if previous_end < end:
            temporary = root / "config.json.tmp"
            if temporary.exists():
                raise FileExistsError(f"incomplete continuation config update: {temporary}")
            temporary.write_text(_canonical(config), encoding="utf-8")
            os.replace(temporary, root / "config.json")
    else:
        if not root.parent.is_dir():
            raise FileNotFoundError(f"continuation parent does not exist: {root.parent}")
        root.mkdir()
        (root / "config.json").write_text(_canonical(config), encoding="utf-8")
    model = _build_grid(config)
    committed = None
    parent_digest = None
    summary = []
    parent_state = None
    for index in range(1, int(end) + 1):
        if fork_path and index <= fork_after:
            ancestor_root = _source_root_for_state(config, source_root, index)
            old_dir = ancestor_root / f"state_{index:04d}"
            old_file = old_dir / "config.json"
            if not old_file.is_file():
                raise FileNotFoundError(f"fork source lacks accepted load config: {old_file}")
            old_config = ConfigDict(json.loads(old_file.read_text(encoding="utf-8")))
            _check_fork_source(config, old_config)
            if int(old_config.data.state_index) != index:
                raise ValueError("fork source has a mismatched load index")
            if Path(old_config.output.dir).resolve() != old_dir.resolve():
                raise ValueError("fork source config points outside its state directory")
            state_config = old_config
        else:
            state_config = ConfigDict(json.loads(config.to_json_best_effort()))
            state_config.data.state_index = index
            state_config.output.dir = str(root / f"state_{index:04d}")
            state_config.output.tensorboard_root = str(
                Path(config.output.tensorboard_root) / root.name
            )
            state_config.output.resume = False
        state_dir = Path(state_config.output.dir)
        state_config_path = state_dir / "config.json"
        if state_config_path.is_file():
            stored_text = state_config_path.read_text(encoding="utf-8")
            old_end = _check_endpoint_extension(state_config, stored_text)
            if old_end < index:
                raise ValueError(f"load {index} exceeds its stored continuation endpoint")
            state_config.data.end_state_index = old_end
        _reference(state_config, model)
        accepted_path = state_dir / "accepted.ckpt.zip"
        if fork_path and index <= fork_after and not accepted_path.is_file():
            raise RuntimeError(f"fork source load {index} is not accepted")
        if accepted_path.is_file():
            if not resume and not (fork_path and index <= fork_after):
                raise FileExistsError(f"accepted load already exists: {accepted_path}")
        else:
            checkpoint = state_dir / "latest.ckpt.zip"
            if state_dir.exists() and not checkpoint.is_file():
                raise RuntimeError(f"load {index} has no checkpoint to resume: {state_dir}")
            if checkpoint.is_file():
                with zipfile.ZipFile(checkpoint) as archive:
                    metadata = json.loads(archive.read("metadata.json"))
                if metadata["phase"] == "complete":
                    raise RuntimeError(f"load {index} completed without acceptance; committed parent unchanged")
            result, accepted_state = run_stagger(
                state_config, resume=state_dir.exists(),
                initial_state=committed, parent_sha256=parent_digest,
                return_state=True,
            )
            report_path = state_dir / "result.json"
            temporary = report_path.with_name(report_path.name + ".tmp")
            temporary.write_text(json.dumps(result, sort_keys=True), encoding="utf-8")
            os.replace(temporary, report_path)
            if not result["accepted"] or accepted_state is None:
                raise RuntimeError(f"load {index} rejected; committed parent unchanged: {result['gates']}")
        committed, parent_digest = _load_accepted(state_config, model, parent_digest)
        if index > 1:
            previous_d, previous_H = map(np.asarray, parent_state[5:])
            new_d, new_H = map(np.asarray, committed[5:])
            if np.min(new_d - previous_d) < -float(config.stagger.healing_atol):
                raise ValueError("accepted continuation exceeds the healing tolerance")
            if np.min(new_H - previous_H) < 0.0:
                raise ValueError("accepted continuation decreased Amor history")
        parent_state = committed
        summary.append({"state_index": index, "load": float(_reference(state_config, model)[0]),
                        "accepted_sha256": parent_digest})
        print(json.dumps({"event": "load_committed", **summary[-1]}, sort_keys=True), flush=True)
    return summary


def main(_):
    config = _absl_flags.FLAGS.config
    run(config, resume=bool(config.output.resume))


if __name__ == "__main__":
    _absl_app.run(main)
