"""Generate centre-refined point cloud for 3D ice melting PINN.

Samples points in the unit cube [0,1]³ with density proportional to
1 / lc(r)³, where *lc(r)* is a radial mesh-size function that ramps
from *lc_center* at the cube centre to *lc_far* at the corners.

No tetrahedral meshing — just direct rejection sampling.  ~100 K points
in under 10 s.

Usage:
  python examples/icemelting3d/create_mesh.py
  python examples/icemelting3d/create_mesh.py -n 80000 --lc_center 0.005
"""

import argparse
import os

import numpy as np


def _mesh_size(r, lc_center, lc_far, r_max, power):
    """Target mesh size at radial distance *r*."""
    return lc_center + (lc_far - lc_center) * (r / r_max) ** power


def generate(output_path, n_target=100_000, lc_center=0.004, lc_far=0.025,
             power=2.0, seed=42):
    """Rejection-sample points with centre-refined density.

    Density at radius *r* is proportional to 1 / lc(r)³, so that the
    local point spacing matches *lc(r)*.

    Args:
        output_path: output ``.npy`` file path.
        n_target: approximate number of output points.
        lc_center: target spacing at cube centre (r=0).
        lc_far: target spacing at cube corners (r=r_max).
        power: radial exponent; higher → refinement stays tighter to centre.
        seed: RNG seed.
    """
    r_max = 3**0.5 / 2                    # centre → corner distance
    vol = 1.0                              # unit cube
    _rng = np.random.default_rng(seed)

    # Estimate required candidates:
    #   acceptance prob at r is (lc_center / lc(r))³ ≤ 1.
    # Integrate over unit cube to get average acceptance rate.
    # Use Monte-Carlo estimate on a quick 20 K sample.
    x_test = _rng.uniform(size=(20000, 3))
    r_test = np.sqrt(np.sum((x_test - 0.5)**2, axis=1))
    lc_test = _mesh_size(r_test, lc_center, lc_far, r_max, power)
    p_test = (lc_center / lc_test) ** 3
    acc_rate = float(np.mean(p_test))

    n_candidates = int(n_target / acc_rate * 1.2)  # 20 % safety margin
    print(f"Estimated acceptance rate: {acc_rate:.3f}  "
          f"→ {n_candidates:,} candidates for ~{n_target:,} points")

    # Rejection sampling in chunks (keep memory low)
    points = []
    remaining = n_candidates
    chunk = 200_000

    while remaining > 0 and len(points) < n_target * 2:
        n = min(chunk, remaining)
        x = _rng.uniform(size=(n, 3))
        r = np.sqrt(np.sum((x - 0.5)**2, axis=1))
        lc = _mesh_size(r, lc_center, lc_far, r_max, power)
        p_accept = (lc_center / lc) ** 3
        keep = _rng.random(n) < p_accept
        points.append(x[keep])
        remaining -= n

    points = np.concatenate(points)[:n_target]
    np.save(output_path, points.astype(np.float32))

    # Quick distribution report
    r = np.sqrt(np.sum((points - 0.5)**2, axis=1))
    bins = [(0., 0.10), (0.10, 0.20), (0.20, 0.35), (0.35, 0.50), (0.50, 0.87)]
    print(f"Saved {len(points):,} points → {output_path}")
    for lo, hi in bins:
        n_shell = ((r >= lo) & (r < hi)).sum()
        vol_shell = _shell_vol(lo, hi)
        dx_eff = (vol_shell / max(n_shell, 1)) ** (1/3)
        print(f"  r ∈ [{lo:.2f}, {hi:.2f}): {n_shell:6,} pts  "
              f"dx_eff≈{dx_eff:.4f}")


def _shell_vol(r_lo, r_hi):
    """Volume of spherical shell [r_lo, r_hi] ∩ unit cube, approximated."""
    # Cube of side 1 centred at (0.5, 0.5, 0.5).  Shell volume ≈ sphere shell
    # corrected by the fraction that falls inside the cube.
    v_sphere = (4/3) * np.pi * (r_hi**3 - r_lo**3)
    # For small r the whole shell is inside; for r > 0.5 it tapers off.
    # A rough correction: treat each shell midpoint and check whether a sphere
    # of that radius would fit in the cube.
    r_mid = (r_lo + r_hi) / 2
    if r_mid < 0.01:
        return v_sphere
    # Ratio of sphere cap outside cube to sphere volume ≈ proportional to
    # (sphere_radius - 0.5) / sphere_radius when r_mid > 0.5
    if r_mid <= 0.5:
        return v_sphere  # whole spherical shell fits in cube
    # r_mid > 0.5: only fraction of shell lies inside
    frac = (0.5 / r_mid) ** 2  # rough 3D correction
    return v_sphere * frac


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="Generate centre-refined point cloud for ice melting")
    p.add_argument("-o", "--output", default="",
                   help="output .npy path")
    p.add_argument("-n", "--n_points", type=int, default=100_000,
                   help="target number of points (default 100000)")
    p.add_argument("--lc_center", type=float, default=0.004,
                   help="target spacing at centre (default 0.004)")
    p.add_argument("--lc_far", type=float, default=0.025,
                   help="target spacing at corners (default 0.025)")
    p.add_argument("--power", type=float, default=2.0,
                   help="radial exponent, >1 keeps refinement tighter (default 2.0)")
    p.add_argument("--seed", type=int, default=42, help="RNG seed")
    args = p.parse_args()

    out = args.output or os.path.join(
        os.path.dirname(__file__),
        f"mesh_n{args.n_points // 1000}k_lc{args.lc_center:.3f}.npy")
    generate(out, args.n_points, args.lc_center, args.lc_far, args.power, args.seed)


if __name__ == "__main__":
    main()
