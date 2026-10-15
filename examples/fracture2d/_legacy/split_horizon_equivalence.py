"""Real frozen-cloud common-horizon equivalence, without checkpoint rewriting."""

import argparse
from dataclasses import fields
import json
from pathlib import Path

import numpy as np

from examples.fracture2d._legacy import nonlocal_baseline as direct
from examples.fracture2d._legacy.inner_probe import load_snapshot
from examples.fracture2d._legacy.split_horizon_model import build_model, read_cloud


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--parent-digest", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config, frozen, _ = load_snapshot(args.base, args.parent_digest)
    cloud = read_cloud(config.data.cloud, config.data.cloud_sha256)
    split = build_model(cloud, delta_u=config.nl.delta, delta_d=config.nl.delta,
        grip_extent=config.nl.delta, h_grip=config.nl.h_g,
        visibility_tolerance=config.nl.visibility_tolerance, E=config.physical.E,
        nu=config.physical.nu, k_res=config.physical.k_res, rank_rtol=config.physical.rank_rtol,
        Gc=config.physical.Gc, l0=config.physical.l0)
    for field in fields(direct.UnionGripGeometry):
        np.testing.assert_array_equal(getattr(frozen.geometry, field.name), getattr(split.geometry, field.name))
    if frozen.static_digest != split.static_digest:
        raise ValueError("common-horizon split adapter changes the frozen static bytes")
    original, adapted = frozen.material_operator.to_scipy(), split.material_operator.to_scipy()
    for name in ("data", "indices", "indptr"):
        np.testing.assert_array_equal(getattr(original, name), getattr(adapted, name))
    if not args.output.parent.is_dir() or args.output.exists():
        raise ValueError("equivalence report output parent/ownership differs")
    report = {"common_horizon_operator_equivalence": True, "static_digest": frozen.static_digest,
              "frozen_model_digest": frozen.model_digest, "split_schema_model_digest": split.model_digest,
              "cloud_sha256": config.data.cloud_sha256, "material_nodes": frozen.n_material,
              "geometry_arrays_equal": True, "damage_CSR_bytes_equal": True,
              "accepted_snapshot_rewritten": False}
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
