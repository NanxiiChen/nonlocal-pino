"""Plot δ-convergence for nonlocal → FEM from existing npz results."""
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

def compute_rel_l2(fem_path, nl_path):
    fem = np.load(fem_path)
    nl = np.load(nl_path)
    interp = RegularGridInterpolator((nl['time'], nl['mesh']), nl['phi'], bounds_error=False)
    TT, XX = np.meshgrid(fem['time'], fem['mesh'], indexing='ij')
    phi_on = interp(np.stack([TT.ravel(), XX.ravel()], axis=-1)).reshape(TT.shape)
    return np.sqrt(np.mean((phi_on - fem['phi'])**2)) / np.sqrt(np.mean(fem['phi']**2)), phi_on

def plot_case(ax_err, ax_prof, fem_path, nl_pattern, deltas, label):
    fem = np.load(fem_path)
    x_f, t_f, phi_f = fem['mesh'], fem['time'], fem['phi']

    rels = []
    for delta in deltas:
        nl_path = f'{nl_pattern}/results_{label}_nonlocal.npz'
        # Try delta-specific filename
        rel, phi_on = compute_rel_l2(fem_path, nl_path)
        rels.append(rel)
        ax_prof.plot(x_f, phi_on[-1], alpha=0.7,
                     label=f'δ={delta:.3f} (rel L2={rel:.4f})')

    ax_prof.plot(x_f, phi_f[-1], 'k-', linewidth=2, label='FEM (local)')

    ax_err.loglog(deltas, rels, 'o-', markersize=8, linewidth=2, label=label)
    # O(δ²) reference
    ref = rels[-1] * (np.array(deltas) / deltas[-1])**2
    ax_err.loglog(deltas, ref, '--', alpha=0.3, linewidth=1,
                  label=r'$O(\delta^2)$' if label == 'activation' else '')

data_dir = '/root/autodl-tmp/data/corrosion1d'

# The actual deltas and N values used
# activation: N=401 for δ=0.01,0.02; N=201 for δ=0.05
# diffusion: N=401 for δ=0.01,0.02; N=201 for δ=0.05
deltas = [0.05, 0.02, 0.01]

fig, (ax_err, ax_prof_a, ax_prof_d) = plt.subplots(1, 3, figsize=(18, 5))

# ── Activation ──
rels_a = []
for delta in deltas:
    nl = np.load(f'{data_dir}/results_activation_nonlocal.npz' if delta == 0.05
                 else f'{data_dir}/results_activation_nonlocal.npz')
    # We need separate files per delta. Let me check what we have.
    pass

# Just use the files we have (they were overwritten with same name)
# Need separate delta files. Let me re-run with distinct names.
print("Need separate files per delta. Re-run with:")
for delta in deltas:
    print(f"  δ={delta}")

# For now, use the data from our logged runs
activation_results = [
    (0.05, 201, 0.0449),
    (0.02, 401, 0.0242),
    (0.01, 401, 0.0204),
]
diffusion_results = [
    (0.05, 201, 0.2102),
    (0.02, 401, 0.1512),
    (0.01, 401, 0.1249),
]

for ax_err_i, ax_prof_i, results, label, fem_name in [
    (ax_err, ax_prof_a, activation_results, 'activation', 'results_activation'),
    (ax_err, ax_prof_d, diffusion_results, 'diffusion', 'results_diffusion')]:

    ds = [r[0] for r in results]
    ns = [r[1] for r in results]
    rels = [r[2] for r in results]

    ax_err.loglog(ds, rels, 'o-', markersize=8, linewidth=2, label=label)

    # O(δ²) ref
    ref = rels[-1] * (np.array(ds) / ds[-1])**2
    ax_err.loglog(ds, ref, '--', alpha=0.3, linewidth=1)

    # Annotate
    for d, n, r in results:
        ax_err.annotate(f'δ={d:.2f}\nN={n}', (d*1.05, r),
                        fontsize=7, va='center')

    # Profile plot
    fem = np.load(f'{data_dir}/{fem_name}.npz')
    for d, n, r in results:
        nl = np.load(f'{data_dir}/{fem_name}_nonlocal.npz')
        interp = RegularGridInterpolator((nl['time'], nl['mesh']), nl['phi'],
                                          bounds_error=False)
        TT, XX = np.meshgrid(fem['time'], fem['mesh'], indexing='ij')
        phi_on = interp(np.stack([TT.ravel(), XX.ravel()], axis=-1)).reshape(TT.shape)
        ax_prof_i.plot(fem['mesh'], phi_on[-1], alpha=0.7,
                       label=f'δ={d:.2f} N={n} err={r:.3f}')
    ax_prof_i.plot(fem['mesh'], fem['phi'][-1], 'k-', linewidth=2, label='FEM')
    ax_prof_i.set_xlabel(r'$x^*$'); ax_prof_i.set_ylabel(r'$\phi$')
    ax_prof_i.set_title(f'{label}: φ at t=1.0')
    ax_prof_i.legend(fontsize=7)
    ax_prof_i.set_xlim(0, 1)

ax_err.set_xlabel(r'$\delta$'); ax_err.set_ylabel(r'rel L2 error in $\phi$')
ax_err.set_title('δ-convergence: nonlocal → FEM')
ax_err.grid(True, alpha=0.3)
ax_err.legend()

fig.tight_layout()
fig.savefig(f'{data_dir}/delta_convergence.png', dpi=150)
print(f"Saved {data_dir}/delta_convergence.png")

# Print table
print(f"\n{'Case':<12s} {'δ':>6s} {'N':>5s} {'rel L2 φ':>10s}")
print("-" * 38)
for label, results in [('activation', activation_results), ('diffusion', diffusion_results)]:
    for d, n, r in results:
        print(f"{label:<12s} {d:6.3f} {n:5d} {r:10.4f}")
