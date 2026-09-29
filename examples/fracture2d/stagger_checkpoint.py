"""Atomic, identity-bound in-progress checkpoints for long staggers."""

import hashlib
import io
import json
import os
from pathlib import Path
import zipfile

import equinox as eqx


SCHEMA = "fracture2d-fp32-stagger-checkpoint-v1"
SCHEMA_FP64 = "fracture2d-fp64-stagger-checkpoint-v2"
SCHEMA_ACCEPTED = "fracture2d-fp64-accepted-load-v2"


def write_checkpoint(path, identity, sweep, phase, step, state, *, schema=SCHEMA):
    """Atomically replace a ZIP containing JSON metadata and Equinox leaves."""
    path = Path(path)
    if not path.parent.is_dir():
        raise FileNotFoundError(f"checkpoint parent does not exist: {path.parent}")
    if phase not in ("before", "damage", "after", "complete"):
        raise ValueError(f"invalid checkpoint phase {phase!r}")
    if sweep < 1 or step < 0:
        raise ValueError("checkpoint sweep/step must be positive/nonnegative")
    payload_file = io.BytesIO()
    eqx.tree_serialise_leaves(payload_file, state)
    payload = payload_file.getvalue()
    metadata = {
        "schema": schema,
        "identity": identity,
        "sweep": int(sweep),
        "phase": phase,
        "step": int(step),
        "state_sha256": hashlib.sha256(payload).hexdigest(),
    }
    temp = path.with_name(path.name + ".tmp")
    if temp.exists():
        raise FileExistsError(f"refusing to overwrite incomplete checkpoint: {temp}")
    with zipfile.ZipFile(temp, "x") as archive:
        archive.writestr("metadata.json", json.dumps(metadata, sort_keys=True))
        archive.writestr("state.eqx", payload)
    os.replace(temp, path)
    return metadata


def read_checkpoint(path, expected_identity, template, *, schema=SCHEMA):
    """Validate a checkpoint's identity and digest before restoring its leaves."""
    with zipfile.ZipFile(path) as archive:
        if set(archive.namelist()) != {"metadata.json", "state.eqx"}:
            raise ValueError("checkpoint has unexpected entries")
        metadata = json.loads(archive.read("metadata.json"))
        payload = archive.read("state.eqx")
    if metadata["schema"] != schema or metadata["identity"] != expected_identity:
        raise ValueError("checkpoint identity/schema mismatch")
    if hashlib.sha256(payload).hexdigest() != metadata["state_sha256"]:
        raise ValueError("checkpoint numerical state digest mismatch")
    if metadata["phase"] not in ("before", "damage", "after", "complete"):
        raise ValueError("checkpoint phase is invalid")
    if type(metadata["sweep"]) is not int or type(metadata["step"]) is not int:
        raise ValueError("checkpoint sweep/step must be integers")
    if metadata["sweep"] < 1 or metadata["step"] < 0:
        raise ValueError("checkpoint sweep/step outside valid range")
    restored = eqx.tree_deserialise_leaves(io.BytesIO(payload), template)
    return metadata, restored
