"""Bounded trial-generation retention after atomic progress publication."""

import hashlib
import json
from pathlib import Path
import re
import stat


FIELDS = ("trial_retention", "trial_tail", "trial_stride")
GENERATION = re.compile(r"trial_(\d{4,})\.npz$")


def _validate(tail, stride):
    if type(tail) is not int or tail < 4 or type(stride) is not int or stride < 1:
        raise ValueError("trial retention requires tail >= 4 and stride >= 1")


def resolve_storage_policy(args):
    """Fresh roots are sparse; resume inherits policy and preserves legacy bytes."""
    requested = {name: vars(args)[name] if name in vars(args) else None for name in FIELDS}
    stored = json.loads((Path(args.output) / "policy.json").read_text()) if args.resume else {}
    if "trial_retention" in stored and not all(name in stored for name in FIELDS):
        raise ValueError("stored trial retention policy is incomplete")
    if args.resume and "trial_retention" not in stored and all(value is None for value in requested.values()):
        return None  # Existing numerical identities omitted storage fields and saved all.
    defaults = dict(trial_retention="sparse", trial_tail=4, trial_stride=50)
    policy = {name: requested[name] if requested[name] is not None else
              stored[name] if name in stored else defaults[name] for name in FIELDS}
    if policy["trial_retention"] not in ("sparse", "all"):
        raise ValueError("trial retention must be sparse or all")
    _validate(policy["trial_tail"], policy["trial_stride"])
    return policy


def prune_trial_fields(state_dir, binding, *, tail, stride):
    """Keep first/tail/periodic fields; refuse to prune before valid publication.

    Committed fields and reports are never candidates. Generations newer than
    the pinned progress pointer are also retained, including interrupted writes.
    """
    _validate(tail, stride)
    state_dir = Path(state_dir)
    if state_dir.is_symlink():
        raise ValueError("trial storage cannot traverse a symlinked state directory")
    name = binding["trial_file"]
    match = GENERATION.fullmatch(name)
    if match is None:
        raise ValueError("trial retention requires a local generation filename")
    progress = json.loads((state_dir / "progress.json").read_text())
    if progress["trial_file"] != name or progress["trial_sha256"] != binding["trial_sha256"]:
        raise ValueError("progress pointer must publish the new generation before pruning")
    pinned = state_dir / name
    if not stat.S_ISREG(pinned.lstat().st_mode) or hashlib.sha256(pinned.read_bytes()).hexdigest() != binding["trial_sha256"]:
        raise ValueError("published trial generation is not a valid regular snapshot")
    current = int(match[1])
    generations = []
    for path in state_dir.glob("trial_*.npz"):
        match = GENERATION.fullmatch(path.name)
        if match is None:
            continue
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("trial retention cannot unlink nonregular generations")
        generations.append((int(match[1]), path, info.st_blocks * 512))
    published = sorted(row for row in generations if row[0] <= current)
    keep = {row[1] for row in published[:1] + published[-tail:]}
    keep.update(path for generation, path, _ in generations if generation % stride == 0 or generation > current)
    keep.add(pinned)
    removed = []
    for _, path, size in generations:
        if path not in keep:
            path.unlink()
            removed.append({"file": path.name, "allocated_bytes": size})
    if removed:
        with (state_dir / "storage_pruning.jsonl").open("a") as stream:
            stream.write(json.dumps({"pinned": name, "tail": tail, "stride": stride, "removed": removed}) + "\n")
    return {"removed_count": len(removed), "retained_count": len(generations) - len(removed)}
