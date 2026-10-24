"""Plot non-uniform core + halo sampling for elasticity2d ElasticD.

Generates a 3-panel figure showing:
  1. Full domain with all grid points (tiny light squares), halo (blue), core (red)
  2. Hole zoom
  3. Radial distribution: core vs all-grid density

Usage:
  source .venv/bin/activate
  python examples/elasticity2d/plot_sampling.py
  python examples/elasticity2d/plot_sampling.py -o /path/to/output.png
  python examples/elasticity2d/plot_sampling.py --K_core 2000 --refine_weight 2.0 --weight_power=2.0
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import jax.numpy as jnp
from nonlocal_pinn.operators import ElasticD


def main():
    p = argparse.ArgumentParser(description="Plot non-uniform sub-sampling for elasticity2d")
    p.add_argument('-o', '--output', default='',
                   help='output PNG path (default: same dir as script)')
    p.add_argument('--K_core', type=int, default=2000,
                   help='number of core points (default: 2000)')
    p.add_argument('--refine_weight', type=float, default=2.0,
                   help='0=uniform; >0 concentrates near hole (default: 2.0)')
    p.add_argument('--weight_power', type=float, default=2.0,
                   help='exponent in w ∝ 1/r^p (default: 2.0)')
    p.add_argument('--delta', type=float, default=0.0075,
                   help='peridynamic horizon (default: 0.0075)')
    p.add_argument('--seed', type=int, default=42,
                   help='RNG seed (default: 42)')
    p.add_argument('--dpi', type=int, default=200,
                   help='output DPI (default: 200)')
    args = p.parse_args()

    # ── Grid (same as train._build_grid) ────────────────────────────────────
    Lx, Ly = 1.0, 0.75
    a_x, a_y = 0.10, 0.15
    cx, cy = Lx / 2.0, Ly / 2.0
    Nx, Ny = 400, 300

    xs = jnp.linspace(0.0, Lx, Nx)
    ys = jnp.linspace(0.0, Ly, Ny)
    Xv, Yv = jnp.meshgrid(xs, ys, indexing='xy')
    x_full = jnp.stack([Xv.ravel(), Yv.ravel()], axis=-1)

    dx = (x_full[:, 0] - cx) / a_x
    dy = (x_full[:, 1] - cy) / a_y
    mask = dx ** 2 + dy ** 2 >= 1.0
    x_base = x_full[mask]
    x_np = np.array(x_base)
    print(f"Grid: {Nx}×{Ny} → {len(x_np):,} points (hole excluded)")

    # ── Sampling weights (same as train._build_sampling_weights) ────────────
    r = np.sqrt(((x_np[:, 0] - cx) / a_x) ** 2 + ((x_np[:, 1] - cy) / a_y) ** 2)
    w = 1.0 + args.refine_weight / (r ** args.weight_power)
    w = w / w.sum()

    # ── Build ElasticD + sub-sample ────────────────────────────────────────
    print(f"Building ElasticD (δ={args.delta})...")
    dV = (Lx / (Nx - 1)) * (Ly / (Ny - 1))
    D = ElasticD(x_base, args.delta, dV=dV, kernel='cubic')
    print(f"  N={D._N:,}, k_avg={D._k_avg:.1f}")

    all_idx, sub_edges, core_mask, K_all = D.sub_sample(args.seed, args.K_core, weights=w)
    print(f"  K_core={args.K_core}, K_all={K_all:,} (core+halo1+halo2)")

    core_pts = x_np[all_idx][core_mask]
    halo_pts = x_np[all_idx][~core_mask]

    r_core = np.sqrt(((core_pts[:, 0] - cx) / a_x) ** 2 + ((core_pts[:, 1] - cy) / a_y) ** 2)

    # ── Plot ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(24, 7),
                             gridspec_kw={'width_ratios': [1, 1, 0.5]})

    BG_COLOR = '#F4F4F4'

    # --- Panel 1: Full view ---
    ax = axes[0]
    ax.scatter(x_np[:, 0], x_np[:, 1],
               c=BG_COLOR, marker='s', s=0.15, alpha=0.6,
               edgecolors='none', rasterized=True, zorder=1)
    ax.scatter(halo_pts[:, 0], halo_pts[:, 1],
               c='#64B5F6', marker='s', s=4, alpha=0.55,
               edgecolors='none', rasterized=True, zorder=2,
               label=f'halo ({len(halo_pts):,})')
    ax.scatter(core_pts[:, 0], core_pts[:, 1],
               c='#D32F2F', marker='o', s=8, alpha=0.85,
               edgecolors='#B71C1C', linewidths=0.3, rasterized=True, zorder=3,
               label=f'core ({len(core_pts):,})')
    hole = Ellipse((cx, cy), 2 * a_x, 2 * a_y, fill=True,
                   facecolor='white', edgecolor='#333333', linewidth=1.5, zorder=4)
    ax.add_patch(hole)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    ax.set_title('Full domain')
    ax.legend(loc='upper right', fontsize=9)

    # --- Panel 2: Hole zoom ---
    ax = axes[1]
    zoom = 0.25
    near = (np.abs(x_np[:, 0] - cx) < zoom) & (np.abs(x_np[:, 1] - cy) < zoom)
    bg_zoom = x_np[near]
    ax.scatter(bg_zoom[:, 0], bg_zoom[:, 1],
               c=BG_COLOR, marker='s', s=0.3, alpha=0.6,
               edgecolors='none', rasterized=True, zorder=1)
    halo_zoom = halo_pts[(np.abs(halo_pts[:, 0] - cx) < zoom) &
                         (np.abs(halo_pts[:, 1] - cy) < zoom)]
    ax.scatter(halo_zoom[:, 0], halo_zoom[:, 1],
               c='#64B5F6', marker='s', s=10, alpha=0.6,
               edgecolors='none', rasterized=True, zorder=2)
    core_zoom = core_pts[(np.abs(core_pts[:, 0] - cx) < zoom) &
                         (np.abs(core_pts[:, 1] - cy) < zoom)]
    n_core_zoom = len(core_zoom)
    ax.scatter(core_zoom[:, 0], core_zoom[:, 1],
               c='#D32F2F', marker='o', s=18, alpha=0.9,
               edgecolors='#B71C1C', linewidths=0.4, rasterized=True, zorder=3)
    hole2 = Ellipse((cx, cy), 2 * a_x, 2 * a_y, fill=True,
                    facecolor='white', edgecolor='#333333', linewidth=1.5, zorder=4)
    ax.add_patch(hole2)
    ax.set_xlim(cx - zoom, cx + zoom)
    ax.set_ylim(cy - zoom, cy + zoom)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    ax.set_title(f'Hole zoom — {n_core_zoom} core pts in view')

    # --- Panel 3: Radial distribution ---
    ax = axes[2]
    bins = np.linspace(1.0, r.max(), 40)
    ax.hist(r, bins=bins, density=True, alpha=0.4, color='gray',
            edgecolor='none', label='all grid')
    ax.hist(r_core, bins=bins, density=True, alpha=0.7, color='#D32F2F',
            edgecolor='none', label=f'core (n={args.K_core})')
    ax.axvline(x=1.0, color='#333333', linestyle='--', linewidth=1.2)
    ax.set_xlabel('elliptic radius r')
    ax.set_ylabel('density')
    ax.set_title('Radial distribution')
    ax.legend(fontsize=9)

    fig.suptitle(
        f"Non-uniform core sampling  |  refine_weight={args.refine_weight}, "
        f"power={args.weight_power}  |  "
        f"δ={args.delta}  |  K_core={args.K_core}, K_all={K_all:,}",
        fontsize=11, y=0.98)

    fig.tight_layout(rect=[0, 0, 1, 0.94])

    out = args.output or os.path.join(os.path.dirname(__file__),
                                       'sampling_weights.png')
    fig.savefig(out, dpi=args.dpi, bbox_inches='tight')
    print(f"Saved: {out}")
    plt.close(fig)


if __name__ == '__main__':
    main()
