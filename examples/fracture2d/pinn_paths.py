"""Resolve relocated legacy PINN artifacts without changing stored identities.

Old checkpoint/config payloads deliberately retain their original absolute
paths and SHA256 identity. Only filesystem access is redirected; reference
inputs in ``data/fracture2d`` are never redirected.
"""

from pathlib import Path


LEGACY_ROOT = Path("/root/autodl-tmp/data/fracture2d")
ARCHIVE_ROOT = Path("/root/autodl-tmp/tf-logs/fracture2d/pinn_runs")


def physical_pinn_path(path):
    """Map old ``pinn_*`` run paths to their immutable tf-logs archive."""
    path = Path(path)
    try:
        relative = path.relative_to(LEGACY_ROOT)
    except ValueError:
        return path
    if not relative.parts or not relative.parts[0].startswith("pinn_"):
        return path
    archived = ARCHIVE_ROOT / relative
    if archived.exists() and path.exists():
        raise ValueError(f"ambiguous legacy PINN path exists in both roots: {path}")
    return archived if archived.exists() or not path.exists() else path
