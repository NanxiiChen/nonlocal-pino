#!/usr/bin/env bash
set -euo pipefail
cd /root/nonlocal-pinn
source /root/miniconda3/etc/profile.d/conda.sh
conda activate fenicsenv
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
# Wait for the existing one-case controller; do not interrupt its MPI child.
python - <<'PY'
from pathlib import Path
import time
path = Path('/proc/60204/cmdline')
while path.exists():
    command = path.read_bytes()
    if not command or b'docs.tasks.melting-local-fem-refine' not in command:
        break
    time.sleep(30)
PY
python -u -m docs.tasks.melting-local-fem-refine > docs/tasks/melting-local-fem-refinement-final.txt 2>&1
conda deactivate
source .venv/bin/activate
python -m unittest tests.test_melting_operator.MeltingOperatorTests.test_float32_rotated_distance_near_principal_axes > docs/tasks/melting-local-fem-jax-geometry.txt 2>&1
python -u -m docs.tasks.melting-local-fem-checkpoints > docs/tasks/melting-local-fem-checkpoint-replay.txt 2>&1
python -u -m docs.tasks.melting-local-fem-evaluate > docs/tasks/melting-local-fem-evaluation.txt 2>&1
