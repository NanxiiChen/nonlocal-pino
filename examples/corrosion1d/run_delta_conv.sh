#!/usr/bin/env bash
set -e

LOG=/root/autodl-tmp/data/corrosion1d/delta_conv_log.txt
echo "case delta nx rel_L2_phi" > $LOG

for case in activation diffusion; do
    for delta in 0.05 0.02 0.01 0.005; do

        if [ "$delta" = "0.05" ]; then nx=101; fi
        if [ "$delta" = "0.02" ]; then nx=251; fi
        if [ "$delta" = "0.01" ]; then nx=501; fi
        if [ "$delta" = "0.005" ]; then nx=501; fi

        echo "=== $case δ=$delta N=$nx ==="

        config=examples/corrosion1d/configs/solver_${case}.py

        python examples/corrosion1d/nonlocal_solver.py \
            --config=$config \
            --config.mesh.n_cells=$((nx - 1)) \
            --config.nl.delta=$delta \
            --config.output.filename=_tmp_nl_dconv.npz 2>&1 | grep -E "Saved|Total|rel"

        python -c "
import numpy as np
from scipy.interpolate import RegularGridInterpolator
fem = np.load('/root/autodl-tmp/data/corrosion1d/results_${case}.npz')
nl = np.load('/root/autodl-tmp/data/corrosion1d/_tmp_nl_dconv.npz')
interp = RegularGridInterpolator((nl['time'], nl['mesh']), nl['phi'], bounds_error=False)
TT, XX = np.meshgrid(fem['time'], fem['mesh'], indexing='ij')
pts = np.stack([TT.ravel(), XX.ravel()], axis=-1)
phi_on = interp(pts).reshape(TT.shape)
rel = np.sqrt(np.mean((phi_on-fem['phi'])**2)) / max(np.sqrt(np.mean(fem['phi']**2)), 1e-16)
with open('${LOG}','a') as f:
    f.write(f'${case} ${delta} ${nx} {rel:.6f}\n')
print(f'  rel L2 φ = {rel:.4f}')
"
    done
done

cat $LOG
